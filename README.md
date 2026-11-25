# WheelLeg · Isaac Gym Preview 4

五连杆轮腿机器人的 Isaac Gym 项目，包含独立运行所需的 URDF、网格、闭环近似与 PPO 训练。运行需要 NVIDIA Isaac Gym Preview 4 与 Python 3.8；安装和换电脑使用见[环境指南](docs/COMPANY_SETUP.md)。

所有运行命令统一从 **`./manage.sh`** 启动，脚本集中在 [scripts/](scripts/README.md)。根目录旧的 `.sh` 和训练/播放 `.py` 入口已移除；运行 `./manage.sh --help` 查看命令。

## 当前模型：2026-11-25 CAD URDF

运行资产已全部替换为 `wheelleg-cad-urdf.zip`：上连杆 50 mm、下连杆 105 mm、轮半径 32.5 mm。腿长命令范围为 8–13 cm，运动学、传感器编码器状态计算、初始姿态和控制目标同步使用这些尺寸。质量、质心与惯量仍需实机标定，详见[当前资产说明](assets/wheelleg/README.md)。

**旧精选权重已从工程移除，不能用于新模型。当前尚无新模型的站稳策略。** 新检查点保存模型指纹，播放和续训会拒绝不匹配的旧权重。历史文档中的站立、跳跃、阶梯效果均属于替换前模型。

在本目录执行模型检查和观察：

```bash
conda activate isaacgym
./manage.sh run tools/check_model.py --report validation/cad_model_check.json
./manage.sh preview --length 0.10
```

预览默认固定机身观察机构，不加载策略。`--free-base` 取消固定；这时没有已训练的平衡策略。可用 `--headless --sim-device cuda:0 --num-envs 16 --steps 300` 做有限 GPU 模型检查。

之后如需重新训练，可使用原注册任务框架，例如仅使用实机可获取状态的站立/速度课程：

```bash
./manage.sh train --task wheelleg_sensor_tracking --num_envs 128 --headless
```

训练默认 `cuda:0`；`--max_iterations 3` 用于有限接口检查。训练完成后，使用 `./manage.sh play --task wheelleg_sensor_tracking --checkpoint <新权重路径>` 播放。`./manage.sh stand --checkpoint <新模型198维站立权重>` 保留键盘驾驶和跳跃控制入口；不能把132维传感器策略传给198维演示控制器。

`wheelleg_flat`、`wheelleg_stand`、`wheelleg_stairs`、`wheelleg_sensor` 及 `wheelleg_sensor_tracking` 均使用同一份新 CAD 资产。任务接口见[任务入口](docs/LEGGED_GYM.md)，传感器定义见[STM32传感器训练](docs/STM32_SENSORS.md)。

当前输出力矩限幅：**每轮 ±0.1 N·m；每髋峰值 ±0.68 N·m、连续 ±0.32 N·m**。髋关节按每回合累计超连续值 2 秒后降额，详见[资产与电机参数](assets/wheelleg/README.md#电机输出力矩)。参考质量尚需实测校准。

## 代码从哪里读

Python 源码统一放在 `legged_gym/`：`envs/wheelleg/` 负责当前机器人，`learning/` 负责共用训练，`utils/` 负责注册与配置。建议按以下顺序阅读：

1. [代码导读](docs/CODE_GUIDE.md)：先看完整调用顺序，再读实现。
2. [wheelleg_config.py](legged_gym/envs/wheelleg/wheelleg_config.py) 与 [model/geometry.py](legged_gym/envs/wheelleg/model/geometry.py)：配置和当前模型尺寸。
3. [wheelleg_robot.py](legged_gym/envs/wheelleg/wheelleg_robot.py) 的 `step()`：按控制、物理、观测/奖励、重置四段理解运行。
4. 按需进入 `control/`、传感器任务和 `learning/`，不必从历史工具开始。

完整阅读路线与动作/观测单位见[代码导读](docs/CODE_GUIDE.md)，目录职责见[项目结构](docs/PROJECT_LAYOUT.md)，添加任务、奖励或机器人见[扩展指南](docs/EXTENDING.md)。现有启动命令保留；Python 导入改用统一的 `legged_gym` 路径。

## 当前能力边界

新 CAD 模型已经完成导入和基础数值检查，尚无通过验收的站立、跳跃或阶梯策略。闭环仍采用弹簧阻尼近似；参考质量和惯量需实机标定。旧模型权重、转换工具和重复训练流程已移除，历史内容可以通过 Git 追溯。

## 检查与资料

```bash
# 固定机身、零动作检查机构
./manage.sh preview --headless --steps 500
# 静态资产检查，不启动策略训练
./manage.sh check
```

工具及其适用范围见[工具索引](tools/README.md)。旧精选权重及 tests/ 已按要求移除，历史内容可从 Git 历史追溯；其余本机 `logs/`、`validation/` 产物被忽略，历史文档中的部分路径可能不随克隆提供。历史状态文件中的 PID、`completed` 和训练轮数都应结合对应权重与评估报告理解。
