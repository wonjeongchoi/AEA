# -*- coding: utf-8 -*-
import copy
import functools
from typing import List
import math

import torch
import torch.nn as nn
import ttab.model_adaptation.utils as adaptation_utils
from ttab.api import Batch
from ttab.model_adaptation.base_adaptation import BaseAdaptation
from ttab.model_selection.base_selection import BaseSelection
from ttab.model_selection.metrics import Metrics
from ttab.utils.auxiliary import fork_rng_with_seed
from ttab.utils.logging import Logger
from ttab.utils.timer import Timer


class ENETTA(BaseAdaptation):
    def __init__(self, meta_conf, model: nn.Module):
        super(ENETTA, self).__init__(meta_conf, model)

        # loss weight
        self.lamb1 = meta_conf.lamb1
        self.lamb2 = meta_conf.lamb2
        self.lamb3 = meta_conf.lamb3
        self.beta = meta_conf.decay_beta
        self.lcs_thr = meta_conf.lcs_thr

        self.ss_ratio = meta_conf.ss_ratio  # sample selection ratio for energy alignment

        # self.source_model = copy.deepcopy(self._model)

    def _initialize_model(self, model: nn.Module):
        """
        Configure model for adaptation.

        configure target modules for adaptation method updates: enable grad + ...
        """
        # TODO: make this more general
        # Problem description: the naming and structure of classifier layers may change.
        # Be careful: the following list may not cover all cases when using model outside of this library.
        model.train()

        if self._meta_conf.adapt_layers == "feat_ext":
            #### except classifier
            self._freezed_module_names = ["fc", "classifier", "head"]
            for index, (name_module, module) in enumerate(model.named_children()):
                if name_module in self._freezed_module_names:
                    module.requires_grad_(False)
        elif self._meta_conf.adapt_layers == "norm":
            #### only normalization layers
            model.requires_grad_(False)
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(True)
                    # bn module always uses batch statistics, in both training and eval modes
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
                if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                    m.requires_grad_(True)
        else:
            raise NotImplementedError

        return model.to(self._meta_conf.device)

    def _initialize_trainable_parameters(self):
        """
        Params in classifier layer is freezed.
        """

        self._adapt_module_names = []
        adapt_params = []
        adapt_param_names = []

        if self._meta_conf.adapt_layers == "feat_ext":
            #### except classifier
            classifier_param_names = []
            for name_module, module in self._model.named_children():
                if name_module in self._freezed_module_names:
                    for name_param, param in module.named_parameters():
                        classifier_param_names.append(f"{name_module}.{name_param}")
                else:
                    self._adapt_module_names.append(name_module)
                    for name_param, param in module.named_parameters():
                        adapt_params.append(param)
                        adapt_param_names.append(f"{name_module}.{name_param}")

            assert (
                len(classifier_param_names) > 0
            ), "Cannot find the classifier. Please check the model structure."


        elif self._meta_conf.adapt_layers == "norm":
            #### adapt only normalization layers
            for name_module, module in self._model.named_modules():
                if isinstance(
                        module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)
                ):  # only bn is used in the paper.
                    self._adapt_module_names.append(name_module)
                    for name_parameter, parameter in module.named_parameters():
                        if name_parameter in ["weight", "bias"]:
                            adapt_params.append(parameter)
                            adapt_param_names.append(f"{name_module}.{name_parameter}")

        else:
            raise NotImplementedError

        assert len(adapt_param_names) > 0, "ENETTA needs some adaptable model parameters."
        return adapt_params, adapt_param_names

    def _post_safety_check(self):
        is_training = self._model.training
        assert is_training, "adaptation needs train mode: call model.train()."

        param_grads = [p.requires_grad for p in (self._model.parameters())]
        has_any_params = any(param_grads)
        assert has_any_params, "adaptation needs some trainable params."

    def one_adapt_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        timer: Timer,
        num_past_batches: int,
        random_seed: int = None,
    ):

        #############################################################################################
        """adapt the model in one step."""
        with timer("forward"):
            if self._meta_conf.loss_name == "em_energy_sp_wlcs":  # AEA
                with fork_rng_with_seed(random_seed):
                    y_hat = model(batch._x)
                    if self._meta_conf.src_data_name == "imagenet":
                        cls_weight = model.fc.weight.data.clone()
                    else:
                        cls_weight = model.classifier.weight.data.clone()
                em_loss = adaptation_utils.softmax_entropy(y_hat).mean(0)
                energy_loss = adaptation_utils.sample_selective_softplus_energy_alignment(y_hat, ratio=self.ss_ratio)
                energy_loss = energy_loss.mean(0)
                logit_csim_loss = adaptation_utils.weighted_lcs(y_hat, cls_weight, self.lcs_thr).mean(0)

                loss = em_loss \
                       + self.lamb2 * energy_loss \
                       + self.lamb3 * logit_csim_loss

            else:
                raise NotImplementedError





            # apply fisher regularization when enabled
            if self.fishers is not None:
                ewc_loss = 0
                for name, param in model.named_parameters():
                    if name in self.fishers:
                        ewc_loss += (
                            self._meta_conf.fisher_alpha
                            * (
                                self.fishers[name][0]  # importance of corresponding parameter
                                * (param - self.fishers[name][1]) ** 2
                            ).sum()
                        )
                loss += ewc_loss

        #############################################################################################
        with timer("backward"):
            loss.backward()
            grads = dict(
                (name, param.grad.clone().detach())
                for name, param in model.named_parameters()
                if param.grad is not None
            )
            optimizer.step()
            optimizer.zero_grad()
        return {
            "optimizer": copy.deepcopy(optimizer).state_dict(),
            "loss": loss.item(),
            "grads": grads,
            "yhat": y_hat,
        }

    def run_multiple_steps(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        model_selection_method: BaseSelection,
        nbsteps: int,
        timer: Timer,
        num_past_batches: int,
        random_seed: int = None,
    ):

        for step in range(1, nbsteps + 1):

            adaptation_result = self.one_adapt_step(
                model,
                optimizer,
                batch,
                timer,
                num_past_batches,
                random_seed=random_seed,
            )

            model_selection_method.save_state(
                {
                    "model": copy.deepcopy(model).state_dict(),
                    "step": step,
                    "lr": self._meta_conf.lr,
                    **adaptation_result,
                },
                current_batch=batch,
            )

    def adapt_and_eval(
        self,
        episodic: bool,
        metrics: Metrics,
        model_selection_method: BaseSelection,
        current_batch: Batch,
        previous_batches: List[Batch],
        logger: Logger,
        timer: Timer,
    ):
        """The key entry of test-time adaptation."""
        # some simple initialization.
        log = functools.partial(logger.log, display=self._meta_conf.debug)
        if episodic:
            log("\treset model to initial state during the test time.")
            self.reset()

        log(f"\tinitialize selection method={model_selection_method.name}.")
        model_selection_method.initialize()

        # evaluate the per batch pre-adapted performance. Different with no adaptation.
        if self._meta_conf.record_preadapted_perf:
            with timer("evaluate_preadapted_performance"):
                self._model.eval()
                with torch.no_grad():
                    yhat = self._model(current_batch._x)
                self._model.train()
                metrics.eval_auxiliary_metric(
                    current_batch._y, yhat, metric_name="preadapted_accuracy_top1"
                )

        # adaptation.
        with timer("test_time_adaptation"):
            nbsteps = self._get_adaptation_steps(index=len(previous_batches))
            log(f"\tadapt the model for {nbsteps} steps with lr={self._meta_conf.lr}.")
            self.run_multiple_steps(
                model=self._model,
                optimizer=self._optimizer,
                batch=current_batch,
                model_selection_method=model_selection_method,
                nbsteps=nbsteps,
                timer=timer,
                num_past_batches=len(previous_batches),
                random_seed=self._meta_conf.seed,
            )

        # select the optimal checkpoint, and return the corresponding prediction.
        with timer("select_optimal_checkpoint"):
            optimal_state = model_selection_method.select_state()
            log(
                f"\tselect the optimal model ({optimal_state['step']}-th step and lr={optimal_state['lr']}) for the current mini-batch.",
            )

            self._model.load_state_dict(optimal_state["model"])
            model_selection_method.clean_up()

            if self._oracle_model_selection:
                # oracle model selection needs to save steps
                self.oracle_adaptation_steps.append(optimal_state["step"])
                # update optimizer.
                self._optimizer.load_state_dict(optimal_state["optimizer"])

        with timer("evaluate_adaptation_result"):
            metrics.eval(current_batch._y, optimal_state["yhat"])
            if self._meta_conf.base_data_name in ["waterbirds"]:
                self.tta_loss_computer.loss(
                    optimal_state["yhat"],
                    current_batch._y,
                    current_batch._g,
                    is_training=False,
                )

        # stochastic restore part of model parameters if enabled.
        if self._meta_conf.stochastic_restore_model:
            self.stochastic_restore()

    @property
    def name(self):
        return "ENETTA"
