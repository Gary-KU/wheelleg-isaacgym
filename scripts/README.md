# 脚本统一管理

所有命令从项目根目录的 `./manage.sh` 启动。运行前激活 `isaacgym` 环境；`./manage.sh --help` 可列出命令。

```text
manage.sh                唯一对外入口：命令清单、默认参数、参数转发
scripts/
  run.sh                 内部 Python 启动器：环境动态库路径和项目工作目录
  preview_model.py       无策略检查 CAD 模型
  list_tasks.py          列出任务
  train.py               注册任务训练
  play.py                注册任务播放
  stand.py               站立/驾驶/跳跃演示
  legacy/                旧参数接口，兼容评估工具和历史实验
```

| 目的 | 命令 |
| --- | --- |
| 看模型约 5 分钟 | `./manage.sh preview --steps 15000` |
| 静态模型检查 | `./manage.sh check` |
| 列出任务 | `./manage.sh tasks` |
| 开始训练 | `./manage.sh train --task wheelleg_sensor_tracking --num_envs 128 --headless` |
| 播放新策略 | `./manage.sh play --task wheelleg_sensor_tracking --checkpoint <路径>` |
| 查看某个命令的参数 | `./manage.sh train --help` |
| 执行专用工具 | `./manage.sh run tools/check_model.py` |

新增脚本时放到本目录，在 `manage.sh` 的 `case` 中增加一条命令；通用检查工具继续放 `tools/`。Python 实现通过项目根目录定位资产、日志和报告，不能依赖脚本原来在根目录的位置。用户参数用 `"$@"` 原样传递。

根目录旧的 `run.sh`、`stand.sh`、`lift.sh`、`stairs.sh`、`train_stairs.sh`、`preview_model.sh` 以及训练/播放 Python 入口已移除。`setup.py` 仍用于可编辑安装。

模型预览默认固定机身，不能证明策略已站稳。新 CAD 模型尚需重新训练，旧权重会被模型指纹校验拒绝。

按用户清理要求，工程中的 `tests/` 和 Git 跟踪的旧精选权重已移除；历史内容可从 Git 历史追溯。
