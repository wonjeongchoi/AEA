# -*- coding: utf-8 -*-
import copy
import warnings
from typing import List

import torch
import torch.nn as nn
import ttab.model_adaptation.utils as adaptation_utils
from ttab.api import Batch
from ttab.loads.define_model import load_pretrained_model
from ttab.model_adaptation.base_adaptation import BaseAdaptation
from ttab.model_selection.base_selection import BaseSelection
from ttab.model_selection.metrics import Metrics
from ttab.utils.logging import Logger
from ttab.utils.timer import Timer


class NoAdaptation(BaseAdaptation):
    """Standard test-time evaluation (no adaptation)."""

    def __init__(self, meta_conf, model: nn.Module):
        super().__init__(meta_conf, model)

    def convert_iabn(self, module: nn.Module, **kwargs):
        """
        Recursively convert all BatchNorm to InstanceAwareBatchNorm.
        """
        module_output = module
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            IABN = (
                adaptation_utils.InstanceAwareBatchNorm2d
                if isinstance(module, nn.BatchNorm2d)
                else adaptation_utils.InstanceAwareBatchNorm1d
            )
            module_output = IABN(
                num_channels=module.num_features,
                k=self._meta_conf.iabn_k,
                eps=module.eps,
                momentum=module.momentum,
                threshold=self._meta_conf.threshold_note,
                affine=module.affine,
            )

            module_output._bn = copy.deepcopy(module)

        for name, child in module.named_children():
            module_output.add_module(name, self.convert_iabn(child, **kwargs))
        del module
        return module_output

    def _initialize_model(self, model: nn.Module):
        """Configure model for adaptation."""
        if hasattr(self._meta_conf, "iabn") and self._meta_conf.iabn:
            # check BN layers
            bn_flag = False
            for name_module, module in model.named_modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    bn_flag = True
            if not bn_flag:
                warnings.warn(
                    "IABN needs bn layers, while there is no bn in the base model."
                )
            self.convert_iabn(model)
            load_pretrained_model(self._meta_conf, model)
        model.eval()
        return model.to(self._meta_conf.device)

    def _post_safety_check(self):
        pass

    def initialize(self, seed: int):
        """Initialize the algorithm."""
        self._model = self._initialize_model(model=copy.deepcopy(self._base_model))

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
        with timer("test_time_adaptation"):
            with torch.no_grad():
                y_hat = self._model(current_batch._x)

                # adaptation_utils.feature_vis(self._meta_conf,
                #                              copy.deepcopy(self._model),
                #                              len(previous_batches),
                #                              "NoAdapt"
                #                              )

                # # for energy/logit analysis
                # energy = adaptation_utils.energy(y_hat)
                # gts = current_batch._y
                # import numpy as np
                # import os
                # import time
                # save_dir = './logs/resnet26/noadapt_debug'
                # #file_name = 'energy_anal_' + time.strftime("%Y%m%d") + '.npz'  # time.strftime("%Y%m%d_%H%M%S")
                # file_name = f'energy_anal_seed{self._meta_conf.seed}_{self._meta_conf.data_names}.npz'
                # full_file_name = os.path.join(save_dir, file_name)
                #
                # if len(previous_batches) == 0:   # os.path.isfile(full_file_name):
                #     np.savez(full_file_name,
                #              logits=np.array(y_hat.cpu()),
                #              energys=np.array(energy.cpu()),
                #              gts=np.array(gts.cpu())
                #              )
                # else:
                #     assert os.path.isfile(full_file_name)
                #     saved_array = np.load(full_file_name)
                #     logits_save = np.concatenate((saved_array['logits'], np.array(y_hat.cpu())), axis=0)
                #     energys_save = np.concatenate((saved_array['energys'], np.array(energy.cpu())), axis=0)
                #     gts_save = np.concatenate((saved_array['gts'], np.array(gts.cpu())), axis=0)
                #     np.savez(full_file_name,
                #              logits=logits_save,
                #              energys=energys_save,
                #              gts=gts_save
                #              )
                #     saved_array.close()

                # ########################################################################################
                # # for energy gap visualization
                # model_for_vis = copy.deepcopy(self._model)
                # if len(previous_batches) == 0:
                #     self._src_loader = adaptation_utils.make_src_loader(self._meta_conf)
                #     src_energy = adaptation_utils.src_prediction(self._meta_conf, model_for_vis, self._src_loader)
                # else:
                #     src_energy = adaptation_utils.src_prediction(self._meta_conf, model_for_vis, self._src_loader)
                #
                # weighted_energys, neg_free_energys = adaptation_utils.softmax_entropy_decompose(y_hat)
                # tar_energy = (-1.0 * neg_free_energys).mean()
                #
                # total_losses = adaptation_utils.softmax_entropy(y_hat)
                # gts = current_batch._y
                # import numpy as np
                # import os
                # import time
                # save_dir = './logs/resnet26/noadapt_debug'
                # file_name = f'energy_anal_{time.strftime("%Y%m%d")}_{self._meta_conf.seed}_noadapt.npz'  # time.strftime("%Y%m%d_%H%M%S")
                # full_file_name = os.path.join(save_dir, file_name)
                #
                # save_arr = np.array([src_energy.cpu(), tar_energy.cpu()])
                # save_arr = np.expand_dims(save_arr, axis=0)
                #
                # if not os.path.isfile(full_file_name):
                #     np.savez(full_file_name,
                #              logits=np.array(y_hat.detach().cpu()),
                #              neg_free_energys=np.array(neg_free_energys.detach().cpu()),
                #              weighted_energys=np.array(weighted_energys.detach().cpu()),
                #              total_losses=np.array(total_losses.detach().cpu()),
                #              gts=np.array(gts.detach().cpu()),
                #              energies=save_arr,
                #              )
                # else:
                #     assert os.path.isfile(full_file_name)
                #     saved_array = np.load(full_file_name)
                #
                #     logits_save = np.concatenate((saved_array['logits'], np.array(y_hat.detach().cpu())), axis=0)
                #     neg_free_energys_save = np.concatenate(
                #         (saved_array['neg_free_energys'], np.array(neg_free_energys.detach().cpu())), axis=0)
                #     weighted_energys_save = np.concatenate(
                #         (saved_array['weighted_energys'], np.array(weighted_energys.detach().cpu())), axis=0)
                #     total_losses_save = np.concatenate(
                #         (saved_array['total_losses'], np.array(total_losses.detach().cpu())), axis=0)
                #     gts_save = np.concatenate((saved_array['gts'], np.array(gts.detach().cpu())), axis=0)
                #     energies_save = np.concatenate((saved_array['energies'], save_arr), axis=0)
                #
                #     np.savez(full_file_name,
                #              logits=logits_save,
                #              neg_free_energys=neg_free_energys_save,
                #              weighted_energys=weighted_energys_save,
                #              total_losses=total_losses_save,
                #              gts=gts_save,
                #              energies=energies_save,
                #              )
                #     saved_array.close()
                # #########################################################################################




        with timer("evaluate_adaptation_result"):
            metrics.eval(current_batch._y, y_hat)
            if self._meta_conf.base_data_name in ["waterbirds"]:
                self.tta_loss_computer.loss(
                    y_hat, current_batch._y, current_batch._g, is_training=False
                )

    @property
    def name(self):
        return "no_adaptation"
