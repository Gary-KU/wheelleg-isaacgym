# 公司电脑安装与学习指南

本仓库是独立的 Isaac Gym Preview 4 轮腿工程。先下载代码，再按需要安装仿真环境；只阅读、修改代码不需要安装 Isaac Gym，也不要求 NVIDIA 显卡。

## 下载与同步

私人仓库需要先在公司电脑登录同一个 GitHub 账号，或给另一个账号授予仓库访问权限。在 GitHub 仓库页点 **Code → HTTPS** 复制地址，然后执行：

```bash
git clone https://github.com/Gary-KU/wheelleg-isaacgym.git wheelleg-isaacgym
cd wheelleg-isaacgym
code .
```

`code .` 用 VS Code 打开当前目录。也可以用 VS Code 的“文件 → 打开文件夹”。仓库内所有日常命令都从这个克隆目录执行，不需要原电脑的 `/home/gary/` 路径。

以后开始工作前，先同步：

```bash
git pull --ff-only
```

修改完成后检查并提交：

```bash
git status
git diff
git add 你实际修改的文件
git commit -m "说明这次修改"
git push
```

首次提交需要按 Git 提示设置自己的姓名和提交邮箱。两台电脑交替工作时，先提交并推送，再换电脑拉取，避免在两边同时改同一份文件。

## 先看哪些代码

| 阅读顺序 | 文件 | 关注点 |
| --- | --- | --- |
| 1 | `stand.sh`、`stairs.sh`、`train_stairs.sh` | 常用命令如何组合参数 |
| 2 | `legged_gym/envs/wheelleg/constants.py`、`cli.py`、`runtime_config.py` | 默认配置、命令行参数与权重配置的覆盖关系 |
| 3 | `play.py`、`keyboard.py` | 按键怎样变成速度和腿长目标 |
| 4 | `legged_gym/envs/wheelleg/wheelleg_robot.py` | 创建仿真、步进、观测、奖励、终止和复位 |
| 5 | `constraints.py`、`leg_control.py`、`wheel_control.py` | 近似闭环外力、支撑前馈、PD、轮子平衡与航向反馈 |
| 6 | `train.py`、`legged_gym/learning/ppo_runner.py`、`third_party/rsl_rl/` | PPO 采样和更新 |

更详细的调用流程见[代码导读](CODE_GUIDE.md)。`./run.sh play.py --help` 等帮助命令也需要先安装仿真环境，因为入口会导入 Isaac Gym。

## 要运行仿真时安装什么

本机验证环境是 Linux、Python 3.8、PyTorch 1.13.1+cu117、Isaac Gym Preview 4。建议在带 NVIDIA 显卡和可用驱动的 Linux 电脑运行；Windows/macOS 电脑可以先阅读代码。CPU 物理模式也依赖 Isaac Gym SDK，窗口显示还需要图形环境。

1. 从 [NVIDIA Isaac Gym 官方页面](https://developer.nvidia.com/isaac-gym)单独取得 Preview 4 SDK，并遵守其许可。仓库不包含 NVIDIA 安装包。不要用名称相似的 PyPI 包代替 SDK。
2. 创建独立的 Python 3.8 环境，与 Isaac Lab 的环境分开。
3. 安装 PyTorch、SDK 和本工程依赖。下面 SDK 路径需要改为你实际解压的路径。

```bash
conda create -n isaacgym python=3.8 -y
conda activate isaacgym
python -m pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install -e /你的路径/isaacgym/python
python -m pip install -r requirements.txt
```

`requirements.txt` 是工程依赖范围，并非完整环境锁定。本机实际使用 NumPy 1.19.5、TensorBoard 2.14.0；前者低于文件声明的 NumPy 下限，因此不能把上述安装步骤视为在新电脑已验证的完整复现。遇到依赖冲突时，先记录 Python、torch、numpy 和 SDK 版本，再参考 SDK 自带安装说明处理。

先运行已有测试，再看演示：

```bash
./run.sh -m unittest discover -s tests -q
./stand.sh
```

`run.sh` 会设置当前 Python 环境的动态库路径，并保证入口先导入 Isaac Gym 再导入 torch。若未激活环境，可临时指定：`PYTHON_BIN=/你的环境/bin/python ./stand.sh`。

## 随仓库提供的权重

| 用途 | 文件 |
| --- | --- |
| 平地站立和键盘演示 | `logs/standing_support_20261022_093823/input/standing_hold.pt` |
| 已复测的 1 cm 阶梯演示 | `logs/stair_training_20261022_103417/evaluation/verified_level0.pt` |
| 继续阶梯训练的起点 | `logs/stair_training_20261022_103417/input/stage4_start.pt` |

```bash
./stand.sh          # 方向键行驶/转向，Q/E 调腿长，空格跳跃，X 停止，R 复位
./stairs.sh         # 自动运行 1 cm 阶梯演示
./train_stairs.sh   # 主动开始新一轮 GPU 训练；只想看效果不要执行这一条
```

平地演示目标速度 ±0.6 m/s，转向 ±2 rad/s，腿长目标范围 8～14 cm。目标值不等于所有工况都能准确跟踪。站立、驾驶与跳跃统一在 `stand.sh`，当前控制与有限复测见 [站立与跳跃](JUMP.md)。1 cm 阶梯只在已有有限复测中通过；2 cm 阶梯的 stage4 新增 400 次更新已结束，最终复测仍未通过。仓库整理没有启动新训练。

## 哪些文件没有上传

`.gitignore` 只放行三份精选权重和少量评估摘要；完整历史训练曲线、原始逐帧轨迹、其他权重仍在原电脑。后续产生的新日志默认也不会自动上传。确需保存新的精选结果时，要明确添加对应的忽略规则例外，再检查体积和内容。

历史文档与 JSON 中的绝对路径是原实验记录，其他历史命令引用的权重也可能未随仓库提供。日常先使用本页的三个脚本。资产和第三方代码来源见 [NOTICE](../NOTICE.md)。

## 新的 legged_gym 入口

同步后使用 `./run.sh legged_gym/scripts/play.py --task wheelleg_stand` 查看精选站立权重；训练、目录和配置继承见 [框架指南](LEGGED_GYM.md)。旧脚本继续可用。运行新入口无需另装参考仓库；`pip install -e . --no-deps` 是可选的源码可编辑安装，本仓库不提供包含 SDK 的独立安装包。
