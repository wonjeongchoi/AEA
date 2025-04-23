import math

class NewConf(object):
    # create the list of hyper-parameters to be replaced.
    to_be_replaced = dict(
        # general for world.
        seed=[111],
        main_file=[
            "run_exp.py",
            ],
        job_name=[
            "cifar10c_online_enetta_sp_wlcs",
        ],
        base_data_name=[
            "cifar10",
        ],
        data_names=[
            "cifar10_c_deterministic-gaussian_noise-5",
            "cifar10_c_deterministic-shot_noise-5",
            "cifar10_c_deterministic-impulse_noise-5",
            "cifar10_c_deterministic-defocus_blur-5",
            "cifar10_c_deterministic-glass_blur-5",
            "cifar10_c_deterministic-motion_blur-5",
            "cifar10_c_deterministic-zoom_blur-5",
            "cifar10_c_deterministic-snow-5",
            "cifar10_c_deterministic-frost-5",
            "cifar10_c_deterministic-fog-5",
            "cifar10_c_deterministic-brightness-5",
            "cifar10_c_deterministic-contrast-5",
            "cifar10_c_deterministic-elastic_transform-5",
            "cifar10_c_deterministic-pixelate-5",
            "cifar10_c_deterministic-jpeg_compression-5",
        ],
        model_name=[
            "resnet26",
        ],
        model_adaptation_method=[
            "enetta"
        ],
        model_selection_method=[
            "last_iterate",
        ],
        offline_pre_adapt=[
            "false",
        ],
        data_wise=["batch_wise"],
        batch_size=[64],
        episodic=[
            "false",
            # "true",
        ],
        inter_domain=["HomogeneousNoMixture"],
        non_iid_ness=[0.1],
        non_iid_pattern=["class_wise_over_domain"],
        python_path=["/home/dnjswjd5457/anaconda3/envs/ttab3.8/bin/python"],
        data_path=["/drive2/CWJ/data/DG"],
        ckpt_path=[
            "./pretrain/ckpt/resnet26_bn_ssh_cifar10.pth",
        ],

        lr = [
            1e-3,

        ],
        n_train_steps=[
            1,
        ],

        intra_domain_shuffle=["true"],
        record_preadapted_perf=["true"],
        device=[
            "cuda:0",
            "cuda:1",
            "cuda:2",
            "cuda:3",
            "cuda:4",
            "cuda:5",
            "cuda:6",
            "cuda:7",
        ],
        gradient_checkpoint=["false"],

        # for ENETTA
        adapt_layers=[
            # "feat_ext",
            "norm",
        ],

        loss_name=[
            "em_energy_sp_wlcs",
        ],

        lamb2=[
            1.0,
        ],

        lamb3=[
            # # for full adapt
            # 0.1,
            # 1.0,
            # 2.0,
            # 5.0,

            # for norm adapt
            1.0,
            5.0,
            10.0,
            15.0,
            20.0,
            25.0,
            30.0,
        ],


        lcs_thr=[
            # 0.9,
            0.95,
            0.99,
        ],

        ss_ratio=[
            0.5,
        ]
    )
