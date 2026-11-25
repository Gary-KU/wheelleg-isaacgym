# WheelLeg · Isaac Gym Preview 4

五连杆轮腿机器人的 Isaac Gym 项目，包含独立运行所需的 URDF、网格、闭环近似、PPO 和历史权重。运行需要 NVIDIA Isaac Gym Preview 4 与 Python 3.8；安装和换电脑使用见[环境指南](docs/COMPANY_SETUP.md)。

## 当前模型：2026-11-25 CAD URDF

运行资产已全部替换为 `wheelleg-cad-urdf.zip`：上连杆 50 mm、下连杆 105 mm、轮半径 32.5 mm。腿长命令范围为 8–13 cm，运动学、传感器编码器状态计算、初始姿态和控制目标同步使用这些尺寸。质量、质心与惯量仍需实机标定，详见[当前资产说明](assets/wheelleg/README.md)。

**旧精选权重只作历史记录，不能用于新模型。当前尚无新模型的站稳策略。** 新检查点保存模型指纹，播放和续训会拒绝不匹配的旧权重。历史文档中的站立、跳跃、阶梯效果均属于替换前模型。

在本目录执行模型检查和观察：

```bash
conda activate isaacgym
./run.sh tools/check_model.py --report validation/cad_model_check.json
./preview_model.sh --length 0.10
```

预览默认固定机身观察机构，不加载策略。`--free-base` 取消固定；这时没有已训练的平衡策略。可用 `--headless --sim-device cuda:0 --num-envs 16 --steps 300` 做有限 GPU 模型检查。

之后如需重新训练，可使用原注册任务框架，例如仅使用实机可获取状态的站立/速度课程：

```bash
./run.sh legged_gym/scripts/train.py --task wheelleg_sensor_tracking --num_envs 128 --headless
```

训练默认 `cuda:0`；`--max_iterations 3` 用于有限接口检查。训练完成后，使用 `legged_gym/scripts/play.py --task wheelleg_sensor_tracking --checkpoint <新权重路径>` 播放。`./stand.sh --checkpoint <新模型198维站立权重>` 保留键盘驾驶和跳跃控制入口；不能把132维传感器策略传给198维演示控制器。

`wheelleg_flat`、`wheelleg_stand`、`wheelleg_stairs`、`wheelleg_sensor` 及 `wheelleg_sensor_tracking` 均使用同一份新 CAD 资产。任务接口见[任务入口](docs/LEGGED_GYM.md)，传感器定义见[STM32传感器训练](docs/STM32_SENSORS.md)。

## 代码从哪里读

Python 源码统一放在 `legged_gym/`：`envs/wheelleg/` 负责当前机器人，`learning/` 负责共用训练，`utils/` 负责注册与配置。建议按以下顺序阅读：

1. [代码导读](docs/CODE_GUIDE.md)：先看完整调用顺序，再读实现。
2. [wheelleg_config.py](legged_gym/envs/wheelleg/wheelleg_config.py) 与 [model/geometry.py](legged_gym/envs/wheelleg/model/geometry.py)：配置和当前模型尺寸。
3. [wheelleg_robot.py](legged_gym/envs/wheelleg/wheelleg_robot.py) 的 `step()`：按控制、物理、观测/奖励、重置四段理解运行。
4. 按需进入 `control/`、传感器任务和 `learning/`，不必从历史工具开始。

完整阅读路线与动作/观测单位见[代码导读](docs/CODE_GUIDE.md)，目录职责见[项目结构](docs/PROJECT_LAYOUT.md)，添加任务、奖励或机器人见[扩展指南](docs/EXTENDING.md)。现有启动命令保留；Python 导入改用统一的 `legged_gym` 路径。

## 替换前模型的历史结果

- `stand.sh` 统一安静站立、驾驶与行驶跳跃；有限平地复测已降低静止轮速，长期慢漂仍需验证，见[运行与复测](docs/USAGE.md)。
- 精选阶梯策略通过了 1 cm 课程的有限复测；2 cm 及更高阶梯尚未通过验收，见[阶梯训练与报告](docs/STAIRS.md)。
- 旧抬轮实验采用轮子策略与显式腿部控制，尚未向前迈步；旧球面胎面代理已从当前模型禁用，见[抬轮说明](docs/WHEEL_LIFT.md)。
- 五连杆采用弹簧阻尼外力近似闭合，具有有限刚度与闭合误差，见[资产说明](assets/wheelleg/README.md)和[SCUT 闭链核对](docs/SCUT_CLOSED_CHAIN.md)。

以下结构整理记录属于替换前模型，当时沿用已有权重，测试与复测结果见[整理验收记录](validation/readability_cleanup_work.json)；具体实验参数和测量留在[训练优化](docs/TRAINING_OPTIMIZATION.md)、[实验记录](docs/EXPERIMENTS.md)与[整理前 README](docs/archive/README_20261027_before_readability_cleanup.md)。

## 检查与资料

```bash
# 固定机身、零动作检查机构
./run.sh play.py --allow-approximate-loop --zero-actions --fix-base --num-steps 500
# 回归测试，不启动策略训练
./run.sh -m unittest discover -s tests -q
```

工具及其适用范围见[工具索引](tools/README.md)。Git 保留三份精选权重和部分评估摘要；其他本机 `logs/`、`validation/` 产物被忽略，历史文档中的部分路径可能不随克隆提供。历史状态文件中的 PID、`completed` 和训练轮数都应结合对应权重与评估报告理解。
