# 运行与参数

全部命令从 `IsaacGym/` 目录执行。日常站立、驾驶和跳跃统一使用 `stand.sh`。

## 先看当前起立与站立效果

```bash
conda activate isaacgym
cd /home/gary/wheelleg_work/WheelLeg_RL_IsaacLab-main/IsaacGym
./stand.sh
```

默认从 10 cm 有效腿长站姿开始，使用安静站立设置。点击仿真窗口即可按空格请求起跳，不需要等待稳定。

| 按键 | 动作 |
| --- | --- |
| 空格 | 直接起跳，加速/转向也可；着地后可再跳，腾空中不重复触发 |
| X | 清除驾驶按键，减速停车 |
| ↑ / ↓ | 前进 / 后退，目标 ±0.6 m/s |
| ← / → | 转向，目标 ±2 rad/s |
| Q / E | 缩腿 / 伸腿，目标范围 8–14 cm |
| R | 重置机器人 |

松开方向键后先刹车，再保持停车位置。预蹲和腾空保持起跳时已下发的平滑速度目标，着地后响应最新驾驶/停车指令；Q/E 始终记录腿长选择，着地后立即应用；双轮持续接地约 0.10 秒即可提示动作完成，稳定性验收单独进行。详细控制方法、测量和代码位置见 [站立与跳跃](JUMP.md)。

```bash
./stand.sh --length 0.12 --seconds 600
./stand.sh --headless --seconds 60 --report validation/my_stand.json
./stand.sh --headless --auto-jumps 3 --seconds 30 --report validation/my_jump.json
./stand.sh --help
```

新入口不再使用旧 `stand.sh --command ...`、`--keyboard-leg-range ...` 等参数。通用 `play.py` 仍提供原参数集，按键和配置以该入口说明为准。

## 阶梯演示与训练

```bash
./stairs.sh
./train_stairs.sh
```

`stairs.sh` 查看已复测的 1 cm 阶梯；`train_stairs.sh` 会开始新的 GPU 续训。2 cm 及以上仍未通过验收，详见 [阶梯训练与复测](STAIRS.md)。

## 原地交替抬轮

```bash
./lift.sh
```

N 请求下一侧，空格开启连续交替或落稳后暂停。它的空格含义与 `stand.sh` 不同，详见 [抬轮说明](WHEEL_LIFT.md)。

## 机构、训练与评估

```bash
# 固定机身、零动作检查机构
./run.sh play.py --allow-approximate-loop --zero-actions --fix-base --num-steps 500
# 注册任务列表
./run.sh legged_gym/scripts/list_tasks.py
# 新的平地训练
./run.sh legged_gym/scripts/train.py --task wheelleg_flat --num_envs 128 --headless
# 回归测试
./run.sh -m unittest discover -s tests -q
```

注册任务训练/播放、续训和权重恢复见 [任务入口](LEGGED_GYM.md)。通用工具与参数见 [工具索引](../tools/README.md)，源码学习路线见 [代码导读](CODE_GUIDE.md)。

## 原地慢漂排查（尚未根治）

当前 `stand.sh` 在原有轮子策略上加入接触预设和停车反馈，有限平地测试的静止轮速明显降低。仍使用近似闭环模型，没有锁住轮子或机身；长期侧向漂移和其他地形需要另外复测。[静止比较](../validation/jump_work.json)与[行驶跳跃](../validation/moving_jump_work.json)保存了各版复测结果。

## 平地驾驶复测

当前已复测前进、停车、单跳、后退和转向的按键序列，见 [站立与跳跃](JUMP.md)。合并前的 5 cm 低蹲起步、8 cm 目标、6–20 cm 调腿和慢漂数据保存在 [历史运行说明](archive/USAGE_20261027_before_unified_stand.md)；这些数据不能直接当作当前入口的结果。
