# 当前工具

日常训练、播放和评估统一使用 `manage.sh`。这里只保留实际使用的资产检查和专项诊断：

| 工具 | 用途 |
| --- | --- |
| [check_model.py](check_model.py) | URDF、网格、质量惯量、关节力矩限制、初始闭环姿态检查 |
| [benchmark_closure.py](benchmark_closure.py) | 固定机身下的闭环载荷响应诊断，不代表平衡能力 |
| [evaluate_support.py](evaluate_support.py) | 198 维站立策略的支撑、力矩、接触与漂移测量 |
| [evaluate_drive_stability.py](evaluate_drive_stability.py) | 198 维策略的前后、启停和转向序列测量 |
| [evaluate_stairs.py](evaluate_stairs.py) | 252 维阶梯策略的完整通过与接触测量 |

```bash
./manage.sh check
./manage.sh evaluate --task wheelleg_sensor_tracking --checkpoint <新权重> --report validation/evaluate.json
./manage.sh run tools/evaluate_support.py --help
```

132 维传感器策略使用统一 `evaluate` 入口。专项评估工具有自己的观测尺寸约束，不能混用不同任务的权重。

旧权重转换、旧 USD 导出、旧训练吞吐脚本和历史一键优化流程已移除，可从 Git 历史追溯。当前 CAD 模型从注册任务重新训练；改变观测布局时重新训练对应任务。
