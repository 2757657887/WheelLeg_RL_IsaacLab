# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from isaaclab.utils import configclass


@configclass
class wheellegrobotPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 25
    max_iterations = 20000
    save_interval = 100
    experiment_name = "wheellegrobot_rough_direct"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=4,
        actor_hidden_dims=[512, 256, 128, 64], 
        critic_hidden_dims=[256, 128, 64],
        activation="elu" ,
  
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=5,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )
