# WheelLeg Robot IsaacLab Minimal Runtime

这是一个为 `WheelLegRobot + RSL-RL` 保留的最小运行仓库。

当前仓库保留了两条可用入口：

```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-wheellegrobot-Direct-v0 --headless
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-wheellegrobot-Direct-v0 --num_envs 1
```

如果你只是想把仓库拉下来、装好环境并开始训练或播放，这份 README 就够用。

## 1. 当前仓库内容

这个仓库现在只服务一个任务：

- 任务 ID：`Isaac-wheellegrobot-Direct-v0`
- 强化学习框架：`rsl_rl`
- 训练入口：`scripts/reinforcement_learning/rsl_rl/train.py`
- 播放入口：`scripts/reinforcement_learning/rsl_rl/play.py`
- 机器人环境：`source/isaaclab_tasks/isaaclab_tasks/direct/WheelLegRobot/wheellegrobot_env.py`
- 机器人资产：`source/isaaclab_assets/isaaclab_assets/robots/wheel_leg_robot.py`
- 机器人 USD：`Robot_USD/Robot_Model.usd`


## 2. 运行前提

推荐环境：

- Windows 10 / 11
- NVIDIA GPU
- Isaac Sim `4.5.0`
- Conda
- Python `3.10`

说明：

- 下面命令默认在 Windows 下执行。
- 如果你用 Linux，把 `isaaclab.bat` 换成 `./isaaclab.sh`。
- 仓库现在使用相对路径定位 `Robot_USD/Robot_Model.usd`，所以别人把仓库放到别的目录也可以运行，不需要手改代码路径。

## 3. 获取仓库

```bash
git clone <your-repo-url>
cd IsaacLab
```

建议把仓库放在纯英文路径下，避免空格和中文路径带来的额外兼容问题。

## 4. 安装 Isaac Sim

先按 NVIDIA 官方方式安装 Isaac Sim 4.5.0。

只要满足下面任意一种，仓库就能启动：

- 当前机器已经有可用的 Isaac Sim 安装
- 或者当前环境里可以找到 `isaacsim` 相关 Python 包
- 或者仓库根目录下有可用的 `_isaac_sim` 关联目录

如果 `isaaclab.bat` 能正常拉起仿真，就说明这一步已经满足。

## 5. 创建并激活 Conda 环境

在仓库根目录执行：

```bash
isaaclab.bat -c env_isaaclab
conda activate env_isaaclab
```

这一步会创建推荐环境名 `env_isaaclab`，并把 Isaac Lab 的启动别名配置进去。

## 6. 安装仓库依赖

在已经激活 `env_isaaclab` 的前提下执行：

```bash
isaaclab.bat -i rsl_rl
```

这一步会完成几件事：

- 把当前仓库里的 `source/isaaclab` 安装到当前环境
- 把当前仓库里的 `source/isaaclab_assets` 安装到当前环境
- 把当前仓库里的 `source/isaaclab_tasks` 安装到当前环境
- 把当前仓库里的 `source/isaaclab_rl` 安装到当前环境
- 安装 `rsl_rl` 所需依赖

## 7. 先做一次最小校验

建议先检查任务是否注册成功：

```bash
isaaclab.bat -p -c "import gymnasium as gym; import isaaclab_tasks; print(gym.spec('Isaac-wheellegrobot-Direct-v0').id)"
```

如果输出：

```bash
Isaac-wheellegrobot-Direct-v0
```

说明当前环境已经正确加载这个仓库，而不是加载到了别的项目或旧版本包。

## 8. 开始训练

正式训练命令：

```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-wheellegrobot-Direct-v0 --headless
```

如果你只是想快速验证能不能跑通，先用一个极小配置测试：

```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-wheellegrobot-Direct-v0 --headless --num_envs 1 --max_iterations 1
```

这个最小命令已经实际验证通过，能够完成：

- 环境解析
- 场景创建
- 仿真启动
- 进入 `rsl_rl` 训练循环

## 9. 如何“播放”或可视化运行

当前最小仓库已经恢复了单独的 `play.py`。

### 方式 A：直接播放最近一次训练出来的模型

```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-wheellegrobot-Direct-v0 --num_envs 1
```

这条命令会默认到下面目录里找最新一次 run 和最新 checkpoint：

```bash
logs/rsl_rl/wheellegrobot_rough_direct/
```

### 方式 B：指定某个 checkpoint 播放

```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-wheellegrobot-Direct-v0 --num_envs 1 --load_run <run_folder> --checkpoint <checkpoint_file>
```

例如：

```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-wheellegrobot-Direct-v0 --num_envs 1 --load_run 2026-06-11_00-29-15 --checkpoint model_0.pt
```

### 方式 C：无界面播放

```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-wheellegrobot-Direct-v0 --headless --num_steps 1000
```

### 方式 D：播放并录视频

```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-wheellegrobot-Direct-v0 --headless --video --video_length 1000 --num_steps 1000
```

视频会保存在所加载 checkpoint 对应 run 目录下的：

```bash
videos/play/
```

### 方式 E：直接开窗口观察训练过程

如果你不是想播已有模型，而是想边训练边看，可以直接去掉 `--headless`：

```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-wheellegrobot-Direct-v0 --num_envs 1
```

补充说明：

- `play.py` 已经做过最小实测，能够成功加载 `model_0.pt` 并推进环境步进。
- `--num_steps` 控制这次播放最多跑多少步，默认 `1000`。
- 如果你没有显式指定 `--load_run` 和 `--checkpoint`，脚本会按最新 run / 最新 checkpoint 的规则自动选择。

## 10. 日志和模型保存位置

训练日志默认保存在：

```bash
logs/rsl_rl/wheellegrobot_rough_direct/
```

每次运行会生成一个时间戳目录，里面通常会有：

- 参数快照
- 训练日志
- checkpoint
- 如果开启了 `--video`，还会有训练或播放视频文件

## 11. 关键文件结构

```text
IsaacLab/
├─ isaaclab.bat
├─ Robot_USD/
│  ├─ Robot_Model.usd
│  └─ config.yaml
├─ scripts/
│  └─ reinforcement_learning/
│     └─ rsl_rl/
│        ├─ play.py
│        └─ train.py
└─ source/
   ├─ isaaclab_assets/
   │  └─ isaaclab_assets/
   │     └─ robots/
   │        └─ wheel_leg_robot.py
   ├─ isaaclab_rl/
   │  └─ isaaclab_rl/
   │     └─ rsl_rl/
   └─ isaaclab_tasks/
      └─ isaaclab_tasks/
         └─ direct/
            └─ WheelLegRobot/
               ├─ __init__.py
               ├─ wheellegrobot_env.py
               └─ agents/
                  └─ rsl_rl_ppo_cfg.py
```

## 12. 关于 `Robot_USD` 和 `URDF`

当前运行时真正依赖的是：

```bash
Robot_USD/Robot_Model.usd
```

也就是说：

- 正常训练需要 `Robot_USD/Robot_Model.usd`
- 正常可视化运行也需要 `Robot_USD/Robot_Model.usd`
- 当前最小仓库**不依赖** `URDF` 目录来启动训练

`Robot_USD/config.yaml` 主要是历史转换记录，不参与当前训练主链。

如果你后面想重新从 URDF 重新生成 USD，那才需要原始 `URDF` 资产；如果你只是使用当前仓库训练和观察结果，`URDF` 不是运行必需项。

## 13. 常见问题

### 1）出现 `Environment 'Isaac-wheellegrobot-Direct' doesn't exist`

说明当前 Python 环境没有正确加载这个仓库里的 `isaaclab_tasks`。

处理顺序：

```bash
conda activate env_isaaclab
isaaclab.bat -i rsl_rl
isaaclab.bat -p -c "import gymnasium as gym; import isaaclab_tasks; print(gym.spec('Isaac-wheellegrobot-Direct-v0').id)"
```

### 2）出现 `ModuleNotFoundError: No module named 'omni.client'`

说明你是直接用普通 `python` 在跑，而不是通过 `isaaclab.bat` 进入 Isaac Sim 运行时。

处理方式：

- 不要直接用裸 `python scripts/...`
- 要用 `isaaclab.bat -p ...`
- 并且先激活 `env_isaaclab`

### 3）训练能启动，但看到 `6 != 10` 的 actuator warning

当前机器人资产里只给 6 个关节配置了驱动，训练链本身仍可正常启动。

这是一个非阻塞告警，不会阻止当前训练主链运行。

## 14. 当前推荐使用流程

第一次使用：

```bash
isaaclab.bat -c env_isaaclab
conda activate env_isaaclab
isaaclab.bat -i rsl_rl
isaaclab.bat -p -c "import gymnasium as gym; import isaaclab_tasks; print(gym.spec('Isaac-wheellegrobot-Direct-v0').id)"
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-wheellegrobot-Direct-v0 --headless --num_envs 1 --max_iterations 1
```

确认没问题后，再跑正式训练：

```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-wheellegrobot-Direct-v0 --headless
```