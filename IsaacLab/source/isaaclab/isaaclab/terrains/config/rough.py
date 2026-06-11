# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from ..terrain_generator_cfg import TerrainGeneratorCfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        "flat_plane": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.9,               # 占50%生成概率
            size=(8.0, 8.0),              # 地形尺寸（x,y方向各8米）
            flat_patch_sampling=None      # 不进行子区域平坦采样
        ),
        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.2,
        #     grid_width=0.55,          # 确保计算余量
        #     grid_height_range=(0.02, 0.1),
        #     platform_width=3.5         # 增加平台容错空间
        # ),

        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.002, 0.03), noise_step=0.01, border_width=2
        ),
        # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        # ),
        # "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        # ),
    },
)
"""Rough terrains configuration."""
