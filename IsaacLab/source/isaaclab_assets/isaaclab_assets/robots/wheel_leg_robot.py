# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""轮腿机器人配置"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorBaseCfg, IdealPDActuator
from isaaclab.assets import ArticulationCfg

##
# 配置
##

_ROBOT_USD_PATH = Path(__file__).resolve().parents[4] / "Robot_USD" / "Robot_Model.usd"

# 定义轮腿机器人的配置
WHEELLEGROBOT_CFG = ArticulationCfg(
    # 配置机器人在场景中的生成方式
    spawn=sim_utils.UsdFileCfg(
        usd_path=_ROBOT_USD_PATH.as_posix(),  # 模型文件路径
        activate_contact_sensors=True,  # 激活接触传感器
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,  # 禁用重力，设置为 False 保持重力作用
            retain_accelerations=False,  # 是否保持加速度状态，False 表示不保留
            linear_damping=0.0,  # 线性阻尼系数，0 表示不施加阻尼
            angular_damping=0.0,  # 角阻尼系数，0 表示不施加角阻尼
            max_linear_velocity=100.0,  # 最大线速度
            max_angular_velocity=100.0,  # 最大角速度
            max_depenetration_velocity=1.0,  # 最大非穿透速度
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,  # 禁用自碰撞检测
            solver_position_iteration_count=32,  # 解算位置迭代次数
            solver_velocity_iteration_count=4,  # 解算速度迭代次数
        ),
    ),
    
    # 初始状态配置，包括机器人位置和关节位置、速度
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.25),  # 初始位置，设定为 (0, 0, 0.245)
        joint_pos={  # 初始关节位置
            "L_F_Joint1": -0.0,  # 左前腿关节1位置
            "L_H_Joint1": 0.0,  # 左后腿关节1位置
            "R_F_Joint1": -0.0,   # 右前腿关节1位置
            "R_H_Joint1": 0.0,   # 右后腿关节1位置
        },
        joint_vel={  # 初始关节速度
            "L_F_Joint1": 0.0,  # 左前腿关节1速度
            "L_H_Joint1": 0.0,  # 左后腿关节1速度
            "R_F_Joint1": 0.0,   # 右前腿关节1速度
            "R_H_Joint1": 0.0,   # 右后腿关节1速度
        },
    ),
    
    # 定义机器人的致动器
    actuators={
        # 机器人腿部的致动器
        "legs": ActuatorBaseCfg(
            class_type=IdealPDActuator,  # 使用 DCMotor 来进行精确的力矩控制
            joint_names_expr=[  # 机器人腿部的关节名称
                "L_F_Joint1",  # 左前腿关节1
                "L_H_Joint1",  # 左后腿关节1
                "R_F_Joint1",  # 右前腿关节1
                "R_H_Joint1",  # 右后腿关节1
            ],
            effort_limit=9.0,  # 力矩限制
            velocity_limit=30.0,  # 速度限制
            stiffness={  # 每个关节的刚度设置
                "L_F_Joint1": 0,  # 左前腿关节1的刚度
                "L_H_Joint1": 0,  # 左后腿关节1的刚度
                "R_F_Joint1": 0,  # 右前腿关节1的刚度
                "R_H_Joint1": 0,  # 右后腿关节1的刚度
            },
            damping={  # 每个关节的阻尼设置
                "L_F_Joint1": 0,  # 左前腿关节1的阻尼
                "L_H_Joint1": 0,  # 左后腿关节1的阻尼
                "R_F_Joint1": 0,  # 右前腿关节1的阻尼
                "R_H_Joint1": 0,  # 右后腿关节1的阻尼
            },
            armature={  # 设置关节的惯性（转动惯量）
                ".*Joint.*": 0.05,  # 所有关节的惯性设置为 0.01
            },
        ),
        
        # 机器人轮子的致动器
        "wheel": ActuatorBaseCfg(
            class_type=IdealPDActuator,  # 使用 DCMotor 来进行精确的力矩控制
            effort_limit=9,  # 力矩限制
            velocity_limit=45.0,  # 速度限制
            joint_names_expr=["L_lun_Joint", "R_lun_Joint"],  # 机器人轮子的关节名称
            stiffness=0,  # 轮子的刚度设置
            damping=0,  # 轮子的阻尼设置
            armature=0.05,  # 设置轮子关节的惯性为 0.01
        ),
    },
    
    collision_group=0,  # 设置碰撞组
)
