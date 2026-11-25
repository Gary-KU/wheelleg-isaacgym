# 左右交替抬轮

当前 CAD 模型尚无训练好的站立策略。抬轮实验入口已保留，需要先训练匹配模型的 198 维站立策略；132 维传感器策略不能用于此入口。旧模型的抬轮效果不代表新模型能力。

```bash
conda activate isaacgym
./manage.sh lift --checkpoint /绝对路径/新站立权重.pt
./manage.sh lift --help
```

点击仿真窗口后按 N 请求单次抬轮，按空格开启或暂停连续交替。默认等待键盘；自动实验加 `--auto`。动作中暂停会先完成落轮与恢复，不会冻结机器人。方向键驾驶和 Q/E 调腿长由 `./manage.sh stand` 提供。

RL 策略负责基础平衡，抬轮控制器负责等待站稳、换重心、收腿、落轮和恢复。每步检查支撑与姿态，通过门槛才进入下一阶段；这仍是原地抬轮实验，尚未实现向前迈步。

| 文件 | 职责 |
| --- | --- |
| [manage.sh](../manage.sh) | 统一启动入口，当前使用 CAD 网格碰撞 |
| [scripts/lift.py](../scripts/lift.py) | 参数、权重加载、仿真循环与报告 |
| [control/wheel_lift.py](../legged_gym/envs/wheelleg/control/wheel_lift.py) | 交替抬轮状态机、支撑与恢复条件 |

`--headless --auto --report validation/my_lift.json` 可运行自动检查并保存结果。应结合姿态、接触、闭环误差和失败原因判断效果，不能只看抬轮次数。历史验收数据与旧实现可通过 Git 追溯。
