# 脚本管理

统一从根目录 `./manage.sh` 启动，先激活 `isaacgym` 环境。

| 命令 | 实现 | 用途 |
| --- | --- | --- |
| `preview` | `preview_model.py` | 无策略模型预览 |
| `check` | `../tools/diagnostics/check_model.py` | 静态模型检查 |
| `tasks` | `list_tasks.py` | 列出任务 |
| `train` / `train-stairs` | `train.py` | 注册任务训练 |
| `play` / `stairs` / `evaluate` | `play.py` | 注册任务播放或无窗口评估 |
| `stand` | `stand.py` | 198 维站立策略的驾驶/跳跃演示 |
| `lift` | `lift.py` | 198 维站立策略的交替抬轮实验 |
| `smoke` | `smoke.py` | 有限物理步检查 |
| `run` | `run.sh` | 运行指定 Python 工具，统一设置动态库与工作目录 |

```bash
./manage.sh --help
./manage.sh preview --steps 15000
./manage.sh train --task wheelleg_sensor_tracking --num_envs 128 --headless
./manage.sh play --task wheelleg_sensor_tracking --checkpoint <新权重>
./manage.sh evaluate --task wheelleg_sensor_tracking --checkpoint <新权重> --report validation/result.json
```

新增脚本放本目录，在 `manage.sh` 的 `case` 中增加一条命令，用 `"$@"` 传递参数。训练与播放只维护一套注册任务实现，旧的 `legacy/` 目录已删除。安装文件 `setup.py` 保留在根目录。

当前没有训练好的新 CAD 策略。预览默认固定机身，不能证明能站稳。`tests/` 和旧精选权重按用户要求保持删除。
