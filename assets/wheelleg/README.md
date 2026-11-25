# 当前 CAD 轮腿模型

当前活动模型来自用户提供的 `wheelleg-cad-urdf.zip`，导入日期为 2026-11-25。`model_info.json` 记录源压缩包 SHA256 和导入文件清单。原 IsaacLab USD 不再是本工程运行时的模型来源。

模型包含 11 个刚体、10 个树关节和两个额外闭环连接。视觉网格与碰撞网格分别存放；机身采用五块碰撞网格。运行时仍使用 Isaac Gym 的弹簧阻尼闭环近似。

| 参数 | 当前值 |
| --- | --- |
| 上连杆 | 50 mm |
| 下连杆 | 105 mm |
| 髋间距 | 约 60.0195 mm |
| 轮半径 / 宽度 | 32.5 / 28 mm |
| 轮距 | 约 218.745 mm |
| 有效腿长命令范围 | 80–130 mm |
| URDF 总质量 | 约 5.84 kg |

**质量来自参考模型，质心及惯量是 CAD 预估值，尚未实测标定。** 当前模型不代表这些动力学参数已经与实机一致。STM32 工程中的旧轮半径也需在以后部署时重新核对。

`loop_constraints.json` 是闭环锚点、主动/被动关节顺序及 CAD 尺寸的共同来源；`legged_gym/envs/wheelleg/model/geometry.py` 统一提供运动学、传感器状态计算及控制器使用的尺寸。为配合新连杆惯量，已移除旧模型的 0.05 kg·m² 人工关节惯量，并将默认髋关节阻尼从 4.0 调为 0.2 N·m·s/rad；这只是导入后的数值适配，仍需训练和平衡验收。右轮在 URDF 树上连接 `R_H_Link2`，额外闭环连接 `R_F_Link2`；不能套用旧模型的右侧父子关系。

旧资产完整备份于本机 `logs/model_backup_20261125/old_wheelleg/`，旧精选权重已移除，可在 Git 历史中追溯。新保存的检查点包含模型指纹，加载时必须匹配；旧权重不能作为新模型已经训练完成的结果。

在 IsaacGym 目录执行：

```bash
./manage.sh run tools/check_model.py --report validation/cad_model_check.json
./manage.sh preview --length 0.10
./manage.sh preview --headless --sim-device cuda:0 --num-envs 16 --steps 300
```

预览默认固定机身，用于检查机构；`--free-base` 取消固定。预览不加载策略，不代表机器人已经能站稳。`--steps` 按 50 Hz 控制步计数。

历史 USD 导出工具默认写到 `logs/usd_export/`，避免覆盖当前 CAD 资产。要更新本模型，应同时更新 URDF、网格、闭环元数据及文件指纹清单，再重新检查并训练。
