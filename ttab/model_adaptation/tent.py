# -*- coding: utf-8 -*-
import copy
import functools
from typing import List

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


class TENT(BaseAdaptation):
    """
    Tent: Fully Test-Time Adaptation by Entropy Minimization,
    https://arxiv.org/abs/2006.10726,
    https://github.com/DequanWang/tent
    """

    def __init__(self, meta_conf, model: nn.Module):
        super(TENT, self).__init__(meta_conf, model)

    def _initialize_model(self, model: nn.Module):
        """Configure model for adaptation."""
        model.train()
        # disable grad, to (re-)enable only what specified adaptation method updates
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
        return model.to(self._meta_conf.device)

    def _initialize_trainable_parameters(self):
        """
        Collect the affine scale + shift parameters from norm layers.

        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.

        Note: other choices of parameterization are possible!
        """
        self._adapt_module_names = []
        adapt_params = []
        adapt_param_names = []

        for name_module, module in self._model.named_modules():
            if isinstance(
                module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)
            ):  # only bn is used in the paper.
                self._adapt_module_names.append(name_module)
                for name_parameter, parameter in module.named_parameters():
                    if name_parameter in ["weight", "bias"]:
                        adapt_params.append(parameter)
                        adapt_param_names.append(f"{name_module}.{name_parameter}")

        assert (
            len(self._adapt_module_names) > 0
        ), "TENT needs some adaptable model parameters."
        return adapt_params, adapt_param_names

    def one_adapt_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        previous_batches: List[Batch],
        timer: Timer,
        random_seed: int = None,
    ):
        """adapt the model in one step."""
        with timer("forward"):
            with fork_rng_with_seed(random_seed):
                y_hat = model(batch._x)
            loss = adaptation_utils.softmax_entropy(y_hat).mean(0)

            # # # for feature visualization
            # adaptation_utils.feature_vis(self._meta_conf,
            #                              copy.deepcopy(self._model),
            #                              len(previous_batches),
            #                              "TENT"
            #                              )

            # # for energy/logit analysis
            # import numpy as np
            # import os
            # import time
            # save_dir = './logs/resnet26/tent_debug'
            # # file_name = 'energy_anal_' + time.strftime("%Y%m%d") + '.npz'  # time.strftime("%Y%m%d_%H%M%S")
            # file_name = f'energy_anal_seed{self._meta_conf.seed}_{self._meta_conf.data_names}.npz'
            # full_file_name = os.path.join(save_dir, file_name)
            #
            # if (len(previous_batches) == 0) or (len(previous_batches)%10 == 0):   # os.path.isfile(full_file_name):
            #     ## 1) source domain
            #     #self._src_loader = adaptation_utils.make_src_loader(self._meta_conf)
            #     self._src_loader = adaptation_utils.make_src_train_loader(self._meta_conf)
            #     src_energies = []
            #     src_gts = []
            #     src_logits = []
            #     model.eval()
            #     with torch.no_grad():
            #         for step, epoch, batch_s in self._src_loader.iterator(
            #                 batch_size=self._meta_conf.batch_size,  # apply the batch-wise or sample-wise setting.
            #                 shuffle=False,  # we will apply shuffle operation in preparing dataset.
            #                 repeat=False,
            #                 ref_num_data=None,
            #                 num_workers=self._meta_conf.num_workers
            #                 if hasattr(self._meta_conf, "num_workers")
            #                 else 2,
            #                 pin_memory=True,
            #                 drop_last=False,
            #         ):
            #             yhat_s = model(batch_s._x)
            #             eng_s = adaptation_utils.energy(yhat_s)
            #
            #             src_logits.append(yhat_s)
            #             src_gts.append(batch_s._y)
            #             src_energies.append(eng_s)
            #     model.train()
            #     src_logits = torch.concat(src_logits, dim=0)
            #     src_gts = torch.concat(src_gts, dim=0)
            #     src_energies = torch.concat(src_energies, dim=0)
            #
            #     ## 2) target domain
            #     self._trg_loader = adaptation_utils.make_trg_loader(self._meta_conf)
            #     trg_energies = []
            #     trg_gts = []
            #     trg_logits = []
            #     model.eval()
            #     with torch.no_grad():
            #         for step, epoch, batch_t in self._trg_loader.iterator(
            #                 batch_size=self._meta_conf.batch_size,  # apply the batch-wise or sample-wise setting.
            #                 shuffle=False,  # we will apply shuffle operation in preparing dataset.
            #                 repeat=False,
            #                 ref_num_data=None,
            #                 num_workers=self._meta_conf.num_workers
            #                 if hasattr(self._meta_conf, "num_workers")
            #                 else 2,
            #                 pin_memory=True,
            #                 drop_last=False,
            #         ):
            #             yhat_t = model(batch_t._x)
            #             eng_t = adaptation_utils.energy(yhat_t)
            #
            #             trg_logits.append(yhat_t)
            #             trg_gts.append(batch_t._y)
            #             trg_energies.append(eng_t)
            #     model.train()
            #     trg_logits = torch.concat(trg_logits, dim=0)
            #     trg_gts = torch.concat(trg_gts, dim=0)
            #     trg_energies = torch.concat(trg_energies, dim=0)
            #
            #
            # if len(previous_batches) == 0:
            #     assert not os.path.isfile(full_file_name)
            #     np.savez(full_file_name,
            #              logits_s=np.array(src_logits.cpu()),
            #              energys_s=np.array(src_energies.cpu()),
            #              gts_s=np.array(src_gts.cpu()),
            #
            #              logits_t=np.array(trg_logits.cpu()),
            #              energys_t=np.array(trg_energies.cpu()),
            #              gts_t=np.array(trg_gts.cpu()),
            #              )
            #
            # elif len(previous_batches) % 10 == 0:
            #     assert os.path.isfile(full_file_name)
            #     saved_array = np.load(full_file_name)
            #     logits_s_save = np.concatenate((saved_array['logits_s'], np.array(src_logits.cpu())), axis=0)
            #     energys_s_save = np.concatenate((saved_array['energys_s'], np.array(src_energies.cpu())), axis=0)
            #     gts_s_save = np.concatenate((saved_array['gts_s'], np.array(src_gts.cpu())), axis=0)
            #
            #     logits_t_save = np.concatenate((saved_array['logits_t'], np.array(trg_logits.cpu())), axis=0)
            #     energys_t_save = np.concatenate((saved_array['energys_t'], np.array(trg_energies.cpu())), axis=0)
            #     gts_t_save = np.concatenate((saved_array['gts_t'], np.array(trg_gts.cpu())), axis=0)
            #
            #     np.savez(full_file_name,
            #              logits_s=logits_s_save,
            #              energys_s=energys_s_save,
            #              gts_s=gts_s_save,
            #
            #              logits_t=logits_t_save,
            #              energys_t=energys_t_save,
            #              gts_t=gts_t_save
            #              )
            #     saved_array.close()
            # else:
            #     pass


            # # for energy/logit analysis
            # weighted_energys, neg_free_energys = adaptation_utils.softmax_entropy_decompose(y_hat)
            # total_losses = adaptation_utils.softmax_entropy(y_hat)
            # gts = batch._y
            # import numpy as np
            # import os
            # import time
            # save_dir = './logs/resnet26/tent_debug'
            # file_name = 'energy_anal_' + time.strftime("%Y%m%d") + '_tent.npz'  # time.strftime("%Y%m%d_%H%M%S")
            # full_file_name = os.path.join(save_dir, file_name)
            #
            # if not os.path.isfile(full_file_name):
            #     np.savez(full_file_name,
            #              logits=np.array(y_hat.detach().cpu()),
            #              neg_free_energys=np.array(neg_free_energys.detach().cpu()),
            #              weighted_energys = np.array(weighted_energys.detach().cpu()),
            #              total_losses=np.array(total_losses.detach().cpu()),
            #              gts=np.array(gts.detach().cpu())
            #              )
            # else:
            #     assert os.path.isfile(full_file_name)
            #     saved_array = np.load(full_file_name)
            #     logits_save = np.concatenate((saved_array['logits'], np.array(y_hat.detach().cpu())), axis=0)
            #     neg_free_energys_save = np.concatenate((saved_array['neg_free_energys'], np.array(neg_free_energys.detach().cpu())), axis=0)
            #     weighted_energys_save = np.concatenate((saved_array['weighted_energys'], np.array(weighted_energys.detach().cpu())), axis=0)
            #     total_losses_save = np.concatenate((saved_array['total_losses'], np.array(total_losses.detach().cpu())), axis=0)
            #     gts_save = np.concatenate((saved_array['gts'], np.array(gts.detach().cpu())), axis=0)
            #     np.savez(full_file_name,
            #              logits=logits_save,
            #              neg_free_energys=neg_free_energys_save,
            #              weighted_energys=weighted_energys_save,
            #              total_losses=total_losses_save,
            #              gts=gts_save
            #              )
            #     saved_array.close()


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
        previous_batches: List[Batch],
        model_selection_method: BaseSelection,
        nbsteps: int,
        timer: Timer,
        random_seed: int = None,
    ):
        for step in range(1, nbsteps + 1):
            adaptation_result = self.one_adapt_step(
                model,
                optimizer,
                batch,
                previous_batches,
                timer,
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
                previous_batches=previous_batches,
                model_selection_method=model_selection_method,
                nbsteps=nbsteps,
                timer=timer,
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
        return "tent"


