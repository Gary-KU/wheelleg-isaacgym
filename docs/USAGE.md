# 运行命令

工程根目录只保留一个管理入口 `manage.sh`，实现统一放在 `scripts/`。先激活环境：

```bash
conda activate isaacgym
./manage.sh --help
```

## 模型检查

```bash
./manage.sh check
./manage.sh preview --length 0.10 --steps 15000
```

预览默认固定机身、运行约 10 秒；上面的 15000 步约为 5 分钟。`--free-base` 取消固定，`--headless` 关闭窗口。当前 CAD 腿长范围为 8–13 cm。

## 训练和播放

```bash
./manage.sh tasks
./manage.sh train --task wheelleg_sensor_tracking --num_envs 128 --headless
./manage.sh play --task wheelleg_sensor_tracking --checkpoint <新权重路径>
```

新 CAD 模型没有可以直接复用的旧精选权重，需要重新训练。训练参数见 `./manage.sh train --help`，播放参数见 `./manage.sh play --help`。

198 维站立策略的专用键盘控制入口为：

```bash
./manage.sh stand --checkpoint <匹配新CAD模型的198维权重>
```

方向键驾驶，Q/E 调腿长，空格跳跃，X 停止，R 重置。132 维传感器策略使用通用 `play`，不能传给这个专用控制器。

## 其他入口

```bash
./manage.sh train-stairs --max_iterations 3
./manage.sh stairs --checkpoint <匹配权重>
./manage.sh lift --checkpoint <匹配的198维权重>
./manage.sh run tools/check_model.py --report validation/model_check.json
```

`train-stairs` 会启动训练，其他命令的参数可用 `--help` 查看。`evaluate` 与 `play` 使用同一注册任务实现，默认无窗口并输出运动报告。

未激活环境时，可以指定 Python：

```bash
PYTHON_BIN=/你的环境/bin/python ./manage.sh preview
```

所有入口都会切换到工程根目录，因此相对的资产、日志与报告路径按工程目录解释。脚本职责和新增方法见 [scripts/README.md](../scripts/README.md)，代码阅读顺序见[代码导读](CODE_GUIDE.md)。
