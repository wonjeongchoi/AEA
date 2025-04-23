# -*- coding: utf-8 -*-
import copy
from typing import Any, Dict

from ttab.model_selection.base_selection import BaseSelection

import torch
import scipy as sp

class BetaMovingAVG(BaseSelection):
    """Return the beta model averaged from multiple iterations """

    def __init__(self, meta_conf, model_adaptation_method):
        super().__init__(meta_conf, model_adaptation_method)
        self.iter = 0
        self.beta_dist = sp.stats.beta(meta_conf.beta, meta_conf.beta)
        self.weight = self.get_weight(self.iter)
        self.weight_sum = self.weight

        # weight avg on/off
        self.sb_avg = meta_conf.sb_avg
        self.mb_avg = meta_conf.mb_avg

        self.start_n = meta_conf.start_n
        self.multi_batch_start_n = meta_conf.multi_batch_start_n

        # initialize
        if hasattr(self.model, "ssh"):
            self.model.ssh.eval()
            self.model.main_model.eval()
        else:
            self.model.eval()

        self.state_list = list()
        self.multi_batch_avg_state = copy.deepcopy(self.model).state_dict()

    def clean_up(self):
        self.state_list = list()

    def save_state(self, state, current_batch):
        self.state_list.append(state)
        self.current_batch = current_batch

    def select_state(self) -> Dict[str, Any]:
        """return the optimal state and sync the model defined in the model selection method."""
        # opt_state = self.average(self.state_list, self.start_n)
        ###  run_multiple_steps 함수에서 online manner로 평균 취하면 이거 할 필요 없음!

        optimal_state = self.state_list[-1]['model']
        y_hat = self.state_list[-1]['yhat']

        return optimal_state, y_hat

    def get_weight(self, it):
        return self.beta_dist.pdf((it + 0.5) / (it + 1))

    # def average(self, states, start_n=0):
    #     assert len(states) >= 1
    #
    #     swa_n = start_n
    #     swa_state = copy.deepcopy(self.model.state_dict())
    #     for state in states:
    #         self.moving_average(swa_state, state['model'], 1.0 / (swa_n + 1))
    #         swa_n += 1
    #
    #     # self.bn_update(self.current_batch, swa_state)
    #     return swa_state

    # from https://github.com/thuml/CLIPood.git
    def moving_average(self, net1, net2):
        self.iter += 1
        self.weight = self.get_weight(self.iter)
        relative_weight = self.weight / self.weight_sum

        if self.iter == 1:
            for (param1_name, param1), (param2_name, param2) in zip(net1.items(), net2.items()):
                assert param1_name == param2_name
                if param1_name.split('.')[-1] not in ['weight', 'bias']:
                    continue

                param1.data = param2.data
        else:
            for (param1_name, param1), (param2_name, param2) in zip(net1.items(), net2.items()):
                assert param1_name == param2_name
                if param1_name.split('.')[-1] not in ['weight', 'bias']:
                    continue

                param1.data = (param1 + relative_weight * param2) / (1 + relative_weight)

    def _check_bn(self, module, flag):
        if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
            flag[0] = True

    def check_bn(self, model):
        flag = [False]
        model.apply(lambda module: self._check_bn(module, flag))
        return flag[0]

    def reset_bn(self, module):
        if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)

    def _get_momenta(self, module, momenta):
        if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
            momenta[module] = module.momentum

    def _set_momenta(self, module, momenta):
        if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = momenta[module]

    def bn_update(self, batch, state):
        """
            BatchNorm buffers update (if any).
            Performs 1 epochs to estimate buffers average using train dataset.

            :param loader: train dataset loader for buffers average estimation.
            :param model: model being update
            :return: None
        """

        model = copy.deepcopy(self.model)
        model.load_state_dict(state)
        if not self.check_bn(model):
            return
        model.train()
        momenta = {}
        model.apply(self.reset_bn)
        model.apply(lambda module: self._get_momenta(module, momenta))
        n = 0

        #input = input.cuda(async=True)
        input = batch.cuda()
        input_var = torch.autograd.Variable(input)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

        model.apply(lambda module: self._set_momenta(module, momenta))


    def moving_avg_in_single_batch(self, avg_model, adapted_model, step):

        assert self.sb_avg is False # BMA do not allow single batch avg

        if self.sb_avg:
            swa_n = self.start_n + step
            self.moving_average(avg_model, adapted_model, 1.0 / (swa_n + 1))
        else: # no averaging
            avg_model = adapted_model
        return avg_model

    def moving_avg_in_multiple_batches(self, current_state):
        if self.mb_avg:
            self.moving_average(self.multi_batch_avg_state, current_state)
            self.multi_batch_start_n += 1

        else:
            return current_state
        return self.multi_batch_avg_state

    @property
    def name(self):
        return "beta_moving_avg"
