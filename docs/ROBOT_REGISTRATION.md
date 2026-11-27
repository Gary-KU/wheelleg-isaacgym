# 接入机器人

通过 `LEGGED_TASK_MODULES` 加载自定义模块，无需修改核心注册表。

```bash
LEGGED_TASK_MODULES=examples.custom_robot ./manage.sh tasks
LEGGED_TASK_MODULES=examples.custom_robot ./manage.sh train --task my_robot
```

示例只演示注册，仍使用现有 WheelLeg。新机器人需要自己的环境，不能只换 URDF。

## 环境契约

环境提供 step/reset、观测以及训练器需要的属性。参考 tasks/base.py。控制器、几何和奖励属于机器人模块；PPO 不包含机器人知识。

配置分成环境/控制、观测、奖励。注册时传入配置实例，注册表深拷贝保证实验隔离。

## 阅读顺序

- `legged_gym/envs/__init__.py`：统一入口。
- `wheelleg/register.py`：任务名称与配置对应。
- `wheelleg/config/`：参数。
- `wheelleg/tasks/`：回合、观测、奖励。
- `wheelleg/control/`：指令斜坡、PD 与外环。
- `wheelleg/model/`：几何和闭环。

## 部署

网络不包含传感器外环和力矩限幅。导出配置必须随网络一起交付。

## 独立接口示例

`examples.point_robot` 是四维观测、二维动作的张量质点示例；用于检验通用接口，不是 miroduck 仿真。

```bash
LEGGED_TASK_MODULES=examples.point_robot ./manage.sh train --task point_robot --sim_device cpu --headless
LEGGED_TASK_MODULES=examples.point_robot ./manage.sh play --task point_robot --checkpoint <model_10.pt> --headless
```

它不依赖五连杆控制器。接入真机器人时将 env.py 替换为物理环境，并补充自己的验收与可视化。
