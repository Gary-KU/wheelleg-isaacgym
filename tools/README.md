# 当前工具

日常训练、播放和评估统一使用 `manage.sh`。这里只保留实际使用的资产检查和专项诊断：

| 工具 | 用途 |
| --- | --- |
| [check_model.py](diagnostics/check_model.py) | URDF、网格、质量惯量、关节力矩限制、初始闭环姿态检查 |
| [benchmark_closure.py](diagnostics/benchmark_closure.py) | 固定机身下的闭环载荷/髋速度与步长收敛诊断，不代表平衡能力 |
| [bootstrap_motion.py](policy/bootstrap_motion.py) | 从已验收站立网络初始化速度指令与转向阻尼，再用 PPO 微调、重新验收 |
| [configure_driving.py](policy/configure_driving.py) | 保留原网络，复制并配置播放速度、Q/E 腿长范围及静止修正；生成后仍需验收 |
| [evaluate_sensor_drive.py](evaluation/evaluate_sensor_drive.py) | 四环境模拟前后/左右转向、松键停车及反向；`--leg-length` 按键盘速率升高后测试 |
| [bootstrap_standing.py](policy/bootstrap_standing.py) | 将传感器反馈写为网络初值，供 PPO 微调；不是已训练策略 |
| [evaluate_sensor_balance.py](evaluation/evaluate_sensor_balance.py) | 132 维策略的确定性连续站立、漂移、腿长及逐关节限幅评估 |
| [evaluate_support.py](evaluation/evaluate_support.py) | 198 维站立策略的支撑、力矩、接触与漂移测量 |
| [evaluate_drive_stability.py](evaluation/evaluate_drive_stability.py) | 198 维策略的前后、启停和转向序列测量 |
| [evaluate_stairs.py](evaluation/evaluate_stairs.py) | 252 维阶梯策略的完整通过与接触测量 |

```bash
./manage.sh check
./manage.sh evaluate --task wheelleg_sensor_tracking --checkpoint <新权重> --report logs/evaluate.json
./manage.sh run tools/evaluation/evaluate_support.py --help
```

132 维传感器策略可以用统一 `evaluate` 入口做播放检查；完整站立诊断使用 `evaluate_sensor_balance.py`，例如：

```bash
./manage.sh run tools/evaluation/evaluate_sensor_balance.py --task wheelleg_sensor \
  --checkpoint <权重路径> --headless --sim_device cpu --num_steps 3000 \
  --report logs/my_stand_report.json
```

该工具关闭 16 秒超时；默认 20 ms 控制周期下 3000 步等于每个机器人 60 秒，若检查点使用 5 ms 则需 12000 步；物理失败仍自动重置并计数。只有没有物理失败，才能把总时长当作连续站立；传感器噪声仍按训练配置保留。报告的安静站立门槛为：至少 60 秒无倒地/超时，前后与横向速度 RMS 不超过 0.02 m/s，偏航角速度 RMS 不超过 0.06 rad/s，前后/横向相对位移峰值不超过 0.1 m，航向变化峰值不超过 0.1 rad，腿长 RMS 误差不超过 5 mm，俯仰/横滚峰值不超过 0.2 rad；通过单次检查不代表所有初态或实机都通过。专项评估工具有自己的观测尺寸约束，不能混用不同任务的权重。

旧权重转换、旧 USD 导出、旧训练吞吐脚本和历史一键优化流程已移除，可从 Git 历史追溯。当前 CAD 模型从注册任务重新训练；改变观测布局时重新训练对应任务。
