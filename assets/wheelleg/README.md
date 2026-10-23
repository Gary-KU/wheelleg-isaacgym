# 从原 USD 导出的轮腿资产

`urdf/Robot_Model.urdf` 来自本仓库 `IsaacLab/Robot_USD/Robot_Model.usd`，保留原模型的 11 个刚体、10 个树关节、网格、质量、质心及完整惯量张量。总质量约 5.84 kg，单位为米、千克、秒及弧度。未复用相邻项目中的简化箱体和杆件。

URDF 采用机身 X 向前、Y 向左、Z 向上的坐标系。初始 10 个关节位置均为零，机身位置默认 `[0, 0, 0.25]`，旋转为单位四元数。轮心在机身下方约 0.182 m。

## 闭环与坐标系

URDF 只保存树结构。`loop_constraints.json` 保存两个额外旋转关节：

- `L_lun_Joint2`：`L_H_Link2` 与 `L_lun`。
- `R_lun_Joint2`：`R_H_Link2` 与 `R_lun`。

`constraints` 中的 `local_pos0/1`、`local_axis0/1` 以及 `local_rot0/1_wxyz` 都在导出后对应 URDF 刚体的局部坐标系中。JSON 同时提供 `default_root_pose`、`default_joint_positions`、`active_joints` 和 `passive_joints`。初始姿态的闭环锚点几何误差小于 0.1 微米；完整数值记录见 `export_report.json`。

URDF 要求子连杆的原点位于其传入关节，因此部分连杆坐标系相对原 USD 发生旋转。网格、质心、惯量和闭环锚点已同步转换，不能再把原 USD 的局部锚点直接用于该 URDF。`export_report.json` 的 `usd_body_from_urdf_body` 给出此变换。

原 USD 的位置驱动未导出。运行环境应只控制四个髋关节与两个轮关节，其余四个膝关节保持被动。URDF 的力矩、速度上限与原 Python 资产配置一致：9 Nm，腿关节 30 rad/s，轮关节 45 rad/s。闭环本身需要运行代码读取 JSON 并施加约束。

`meshes/*.obj` 为原始三角网格，视觉及碰撞相同的网格共用文件。运行时 PhysX 仍会按 Isaac Gym 的碰撞设置进行凸包或凸分解处理。导入应设置 `override_com=False`、`override_inertia=False`，避免覆盖原质量属性。

## 重新导出

一般运行无需 USD 依赖，导出的资产已经包含在仓库中。仅在原 USD 修改后，使用包含 `usd-core==26.8` 的 Python 3.10 或更新解释器运行：

```bash
python IsaacGym/tools/export_usd_asset.py
```

导出工具仅依赖 Python 标准库和 `pxr`，不需要 Isaac Sim，也不修改源 USD。可在独立的临时虚拟环境中安装导出依赖，避免改动 Isaac Gym 的 Python 3.8 环境。

工具会重新读取写出的 URDF，做树结构前向运动学，校验两处闭环锚点、转轴和原始刚体姿态，并生成可检查的 `export_report.json`。几何不闭合时会失败退出。
