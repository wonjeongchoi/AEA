# -*- coding: utf-8 -*-

# 1. This file collects significant hyperparameters for the configuration of TTA methods.
# 2. We are only concerned about method-related hyperparameters here.
# 3. We provide default hyperparameters from the paper or official repo if users have no idea how to set up reasonable values.
import math

algorithm_defaults = {
    "no_adaptation": {"model_selection_method": "last_iterate"},
    #
    "bn_adapt": {
        "adapt_prior": 0,  # the ratio of training set statistics.
    },
    "shot": {
        "optimizer": "SGD",  # Adam for officehome
        "auxiliary_batch_size": 32,
        "threshold_shot": 0.9, # 0.9 # confidence threshold for online shot.
        "ent_par": 1.0,
        "cls_par": 0.3,  # 0.1 for officehome.
        "offline_nepoch": 10, # 10
    },
    "ttt": {
        "optimizer": "SGD",
        "entry_of_shared_layers": "layer2",
        "aug_size": 32,
        "threshold_ttt": 1.0,
        "dim_out": 4,  # For rotation prediction self-supervision task.
        "rotation_type": "rand",
    },
    "tent": {
        "optimizer": "SGD",
    },
    "t3a": {"top_M": 100},
    "cotta": {
        "optimizer": "SGD",
        "alpha_teacher": 0.999,  # weight of moving average for updating the teacher model.
        "restore_prob": 0.01,  # the probability of restoring model parameters.
        "threshold_cotta": 0.92,  # Threshold choice discussed in supplementary
        "aug_size": 32,
    },
    "eata": {
        "optimizer": "SGD",
        "eata_margin_e0": math.log(1000)
        * 0.40,  # The threshold for reliable minimization in EATA.
        "eata_margin_d0": 0.05,  # for filtering redundant samples.
        "fishers": True, # whether to use fisher regularizer.
        "fisher_size": 2000,  # number of samples to compute fisher information matrix.
        "fisher_alpha": 50,  # the trade-off between entropy and regularization loss.
    },
    "memo": {
        "optimizer": "SGD",
        "aug_size": 32,
        "bn_prior_strength": 16,
    },
    # "ttt_plus_plus": {
    #     "optimizer": "SGD",
    #     "entry_of_shared_layers": None,
    #     "batch_size_align": 256,
    #     "queue_size": 256,
    #     "offline_nepoch": 500,
    #     "bnepoch": 2,  # first few epochs to update bn stat.
    #     "delayepoch": 0,  # In first few epochs after bnepoch, we dont do both ssl and align (only ssl actually).
    #     "stopepoch": 25,
    #     "scale_ext": 0.5,
    #     "scale_ssh": 0.2,
    #     "align_ext": True,
    #     "align_ssh": True,
    #     "fix_ssh": False,
    #     "method": "align",  # choices = ['ssl', 'align', 'both']
    #     "divergence": "all",  # choices = ['all', 'coral', 'mmd']
    # },
    "note": {
        "optimizer": "SGD",  # use Adam in the paper
        "memory_size": 64,
        "update_every_x": 64,  # This param may change in our codebase.
        "memory_type": "PBRS",
        "bn_momentum": 0.01,
        "temperature": 1.0,
        "iabn": False,  # replace bn with iabn layer
        "iabn_k": 4,
        "threshold_note": 1,  # skip threshold to discard adjustment.
        "use_learned_stats": True,
    },
    "conjugate_pl": {
        "optimizer": "SGD",
        "temperature_scaling": 1.0,
        "model_eps": 0.0,  # this should be added for Polyloss model.
    },
    "sar": {
        "optimizer": "SGD",
        "sar_margin_e0": math.log(1000)
        * 0.40,  # The threshold for reliable minimization in SAR.
        "reset_constant_em": 0.2,  # threshold e_m for model recovery scheme
    },

    ##################################################################################################################
    "swa": {
        "optimizer": "SGD",
        "sb_avg":True,
        "mb_avg":True,
        "start_n": 0,
        "multi_batch_start_n": 0,
        "batch_avg_cycle": 10,
    },

    "watta": {
        "optimizer": "SGD",
        "sb_avg":False,
        "mb_avg":True,
        "start_n": 0,
        "multi_batch_start_n": 0,
        "batch_avg_cycle": 1,

        # for Beta Moving AVG
        "beta": 0.5,
        "watta_vname": "FeatureExtAdapt",

        # for Exp Moving AVG
        "exp_alpha": 0.99,
    },

    "watta_nl": {
        "optimizer": "SGD",
        "sb_avg":False,
        "mb_avg":False,
        "start_n": 0,  # from single batch
        "multi_batch_start_n": 0,
        "batch_avg_cycle": 1,  # 10, 30

        # for Beta Moving AVG
        "beta": -1,

        # for Exp Moving AVG
        "exp_alpha": -1,

        # for new loss
        "loss_name": "ent_min_div_ada_energy_decay_lcs",    # ent_min_div_energy_align, ent_min, ent_min_div, ent_min_energy, (energy_weighted_ent_min), ent_min_div_energy, ent_min_div_ada_energy, ent_min_ada_energy, ent_min_div_ada_energy_decay, ent_min_ada_energy_decay, ent_min_ada_energy_decay
        'lamb1': 1.0,
        'lamb2': 1.0,
        'lamb3': 1.0,
        'lamb4': 2.0,
    },

    "enetta": {
        "optimizer": "SGD",

        # for new loss
        "loss_name": "csa_eem_div_lcs",
        # ent_min_energy,
        # ent_min_div_energy,
        # ent_min_div_lcs,
        # ent_min_div_energy_lcs,

        # eem_div_lcs,
        # csa_eem_div_lcs,
        # esa_eem_div_lcs,

        'lamb1': 1.0,
        'lamb2': 0.05,      # Energy
        'lamb3': 5.0,      # Logit Cosine Similarity
        'decay_beta': -1,
        'lcs_thr': 0.99,
        'filter_thr': math.log(1000) * 0.40,
    },

    # "enetta": {  # decaying version
    #     "optimizer": "SGD",
    #
    #     # for new loss
    #     "loss_name": "ent_min_div_energy_lcs",
    #     # ent_min_div_energy_decay,
    #     # ent_min_div_energy_lcs_decay,

    #     # eem_div_lcs_decay,
    #     # csa_eem_div_lcs_decay,
    #     # esa_eem_div_lcs_decay,
    #
    #     'lamb1': 1.0,
    #     'lamb2': 0.9,      # Energy
    #     'lamb3': 4.0,      # Logit Cosine Similarity
    #     'decay_beta': 0.1,
    #     'lcs_thr': 0.9,
    # },
}
