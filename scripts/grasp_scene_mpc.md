# AP001 MPC 抓取场景说明

## 运行方式

```bash
/mnt/DOCUMENT/ap001_mujoco/.venv/bin/python /mnt/DOCUMENT/ap001_mujoco/scripts/grasp_scene_mpc.py
```

纯手动模式：

```bash
/mnt/DOCUMENT/ap001_mujoco/.venv/bin/python /mnt/DOCUMENT/ap001_mujoco/scripts/grasp_scene_mpc.py --manual
```

脚本会自动生成独立的 MPC 场景文件：

```text
assets/grasp_scene/ap001_left_grasp_mpc_scene.generated.xml
```

## 控制目标

食指、中指、无名指通过位置执行器闭环控制指尖触觉力，目标力由下面参数设定：

```python
TARGET_GRASP_FORCE = 0.2
```

抓稳后，手的基座沿 `hand_z` 做 30 cm 上下循环运动。轨迹峰值加速度由下面参数设定：

```python
LIFT_HEIGHT = 0.30
END_EFFECTOR_MAX_ACCEL = 1.0
```

## 第一版 MPC 算法

每根受控手指使用一个独立的一维离散 MPC：

```text
输入: 手指 position actuator 的 ctrl
输出: 对应指尖 touch sensor 的力
目标: 让指尖力稳定追踪 TARGET_GRASP_FORCE
```

预测模型：

```text
f_next = f_now + force_gain * delta_u + accel_gain * hand_z_accel * dt
```

其中：

```text
f_now: 当前触觉力
delta_u: 候选手指位置增量
force_gain: 在线估计的力-位置增益
hand_z_accel: 已知末端上下循环轨迹的未来加速度
```

第一版里 `accel_gain` 默认是 `0.0`，也就是先保守地主要依赖力-位置模型；后续可以根据实验数据调大。

## 代价函数

MPC 会在一组离散候选动作中选择代价最低的动作：

```text
cost =
  MPC_FORCE_WEIGHT  * 力误差^2
+ MPC_CTRL_WEIGHT   * 控制增量^2
+ MPC_SMOOTH_WEIGHT * 控制增量变化^2
```

相关参数：

```python
MPC_HORIZON = 18
MPC_DT = 0.03
MPC_DELTA_CANDIDATES = (-0.0015, -0.00075, 0.0, 0.00075, 0.0015)
MPC_FORCE_WEIGHT = 1.0
MPC_CTRL_WEIGHT = 0.12
MPC_SMOOTH_WEIGHT = 0.55
MPC_FORCE_GAIN_INIT = 8.0
MPC_ACCEL_GAIN_INIT = 0.0
MPC_GAIN_UPDATE_RATE = 0.02
FORCE_FILTER_ALPHA = 0.88
```

## 调参建议

如果力跟踪太慢：

```python
MPC_DELTA_CANDIDATES = (-0.005, -0.0025, 0.0, 0.0025, 0.005)
MPC_FORCE_WEIGHT = 2.0
```

如果力抖动：

```python
MPC_SMOOTH_WEIGHT = 0.8
MPC_CTRL_WEIGHT = 0.2
FORCE_FILTER_ALPHA = 0.93
```

如果抓取过程中上抬导致力明显波动，可以尝试：

```python
MPC_ACCEL_GAIN_INIT = 0.02
```

## 与原脚本的区别

`grasp_scene.py` 使用的是简单比例力反馈：

```text
force_error -> 修改 position ctrl
```

`grasp_scene_mpc.py` 使用的是预测控制：

```text
预测未来多步触觉力 -> 选择最平滑且误差最小的下一步 ctrl
```
