# SCUT 闭链实现核对

参考仓库：[scutrobotlab/wheeled-legged_RL](https://github.com/scutrobotlab/wheeled-legged_RL)，本次读取版本 `1a26356d6f591734d4f6f666f5cead1a6925da67`。

它使用 Isaac Sim / Isaac Lab。闭链关键定义位于二进制 USD 物理层的 `loop_joints`，不是训练奖励函数：

- [wheelbipeV14_2_physics.usd](https://github.com/scutrobotlab/wheeled-legged_RL/blob/1a26356d6f591734d4f6f666f5cead1a6925da67/source/agent_world/agent_world/assets/usd_files/wheelbipeV14_2_1/configuration/wheelbipeV14_2_physics.usd)
- [wheelbipe_V14_2.py](https://github.com/scutrobotlab/wheeled-legged_RL/blob/1a26356d6f591734d4f6f666f5cead1a6925da67/source/agent_world/agent_world/assets/wheelbipe_V14_2.py)：资产加载、驱动、重力和求解迭代配置。

本次用临时目录中的 USD 读取器实际解开了物理层，找到左右各 3 个 `PhysicsSphericalJoint`：`f3r1`、`f4r2`、`s2r2`。每个关节都有两端刚体 `body0/body1`、各自的局部锚点和 `excludeFromArticulation=true`。也就是先保留树形 articulation，再用树外的物理约束把分支接起来。球铰约束两端位置重合，保留相对旋转；它不是单自由度转动铰链。

USD 层中保存了 articulation 的位置/速度迭代次数 32/1；资产 Python 配置又指定 12/6。实际运行应检查最终生效配置，不能只看 USD 或只复制某个数字。

## 对本项目的应用

沿用它的结构思路：树形主体和树外闭合连接分开描述，闭合锚点属于各自连杆局部坐标系；控制器通过主动关节施加力矩，被动关节和闭环约束负责机构运动。

但当前机器人是另一套五连杆模型，原闭合点是两处铰链，SCUT 机器人有更多分支、导向连杆和弹簧。不能直接照搬它的六个球铰、锚点坐标或执行器参数。

当前 Isaac Gym Preview 4 工程通过 URDF 导入树形模型；所用公开 Python 接口和资产加载路径没有直接复用上述 Isaac Sim USD 场景关节的接入方式。本次框架迁移继续使用：

- [loop_constraints.json](../assets/wheelleg/loop_constraints.json)：本机器人的两处闭合连接、局部锚点和铰链轴。
- [constraints.py](../wheelleg_gym/constraints.py)：局部点变换、点速度、成对弹簧阻尼力及轴对齐力矩。
- [legged_robot.py](../legged_gym/envs/base/legged_robot.py)：每个 0.5 ms 物理步重新计算闭环外力；重力 -9.81 m/s²，默认求解迭代 32/4。
- 新配置 `cfg.closure`：近似模型确认、刚度和阻尼。力限幅仍在原共用配置中。

因此，这是**参考其树外闭合的组织方式，继续使用 Gym 中已有的柔性近似**。这次没有把 Gym 的弹簧阻尼连接升级为原生刚性关节，没有导入 SCUT 资产，也没有用奖励或直接改位置来伪装闭合。若以后改用 Isaac Sim / Isaac Lab，可按本机器人重新建立树外关节并单独验收闭合精度。

本次提取的关节类型和锚点数据作为只读参考，放在 [迁移验收记录](../validation/legged_framework_work.json) 中，运行工程不依赖 USD 读取器或 SCUT 仓库。上游 MIT 许可副本见 [SCUT_REFERENCE_LICENSE](../third_party/SCUT_REFERENCE_LICENSE)。
