# 阶梯任务

当前 CAD 模型需要重新训练，旧模型的通过记录不能用于证明新模型的能力。

```bash
./manage.sh train-stairs --num_envs 128 --headless
./manage.sh stairs --checkpoint <新模型252维阶梯权重>
./manage.sh evaluate --task wheelleg_stairs --checkpoint <新权重> --report validation/stairs.json
```

地形生成在 `envs/wheelleg/terrain.py`；任务配置在 `config/base.py::WheelLeggedStairsCfg`；奖励在 `rewards/stairs.py`；实际接触、失败和成功判断在 `WheelLegRobot._stair_diagnostics()`。

专项完整课程测量可查看 `./manage.sh run tools/evaluation/evaluate_stairs.py --help`。报告中的成功、物理失败和超时分别统计；普通程序退出不代表通过课程。

旧 198→252 维权重转换已删除。新模型、新观测布局使用对应注册任务重新训练。
