#!/usr/bin/env python3
"""Launch an AP001 grasp scene with lightweight MPC force tracking.

python /mnt/DOCUMENT/ap001_mujoco/scripts/grasp_scene_mpc.py

"""

from __future__ import annotations

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt


SOURCE_HAND_XML = REPO_ROOT / "assets" / "AP001" / "model" / "rohand_gen2_left.xml"
SCENE_DIR = REPO_ROOT / "assets" / "grasp_scene"
DEFAULT_SCENE_PATH = SCENE_DIR / "ap001_left_grasp_mpc_scene.generated.xml"
FINGER_ACTUATORS = (
    "index_finger",
    "middle_finger",
    "ring_finger",
    "little_finger",
    "thumb_root",
    "thumb",
)
HAND_POSE_ACTUATORS = (
    "hand_x",
    "hand_y",
    "hand_z",
    "hand_roll",
    "hand_pitch",
    "hand_yaw",
)
INITIAL_FINGER_CTRL = {
    "index_finger": 0.0,
    "middle_finger": 0.0,
    "ring_finger": 0.0,
    "little_finger": 0.0,
    "thumb_root": 1.57,
    "thumb": 0.0,
}
FINGER_ACTUATOR_KP = {
    "index_finger": "35",
    "middle_finger": "35",
    "ring_finger": "35",
    "little_finger": "35",
    "thumb_root": "30",
    "thumb": "30",
}
SIM_STEPS_PER_RENDER = 5
PLOT_UPDATE_INTERVAL = 0.12
FORCE_CONTROL_FINGERS = ("index", "middle", "ring")
FORCE_SENSOR_BY_FINGER = {finger: f"{finger}_touch" for finger in FORCE_CONTROL_FINGERS}
ACTUATOR_BY_FORCE_FINGER = {
    "index": "index_finger",
    "middle": "middle_finger",
    "ring": "ring_finger",
}
# 手指额目标力调整
# 三指抓取的目标接触力，单位 N；力控阶段会尽量让食指/中指/无名指维持在该力附近。
TARGET_GRASP_FORCE = 1.0
# 允许的力误差范围，单位 N；闭合阶段达到 TARGET_GRASP_FORCE - FORCE_TOLERANCE 后进入抓稳/抬升逻辑。
FORCE_TOLERANCE = 0.1
# 手指闭合速度，单位约为 rad/s；数值越大，手指越快闭合，但也更容易冲过目标力。
FINGER_CLOSE_RATE = 0.10
# 抓稳后基座上下循环运动的高度差，单位 m；0.30 表示上下行程 30 cm。
LIFT_HEIGHT = 0.30
# 末端执行器上下运动的峰值加速度，单位 m/s^2；用于生成平滑循环抬升轨迹。
END_EFFECTOR_MAX_ACCEL = 1.0
# MPC 预测步数；MPC_HORIZON * MPC_DT 是预测窗口长度。
MPC_HORIZON = 18
# MPC 控制周期，单位 s；不是 MuJoCo timestep，而是每隔多久重算一次控制量。
MPC_DT = 0.03
# 每个 MPC 周期允许的手指位置增量候选，单位 rad。
MPC_DELTA_CANDIDATES = (-0.0015, -0.00075, 0.0, 0.00075, 0.0015)
# 预测力误差权重；越大越重视力跟踪。
MPC_FORCE_WEIGHT = 1.0
# 控制增量权重；越大越不愿意大幅改变手指位置。
MPC_CTRL_WEIGHT = 0.12
# 控制平滑权重；越大越抑制相邻控制动作突变。
MPC_SMOOTH_WEIGHT = 0.55
# 初始局部力-位置增益，单位约 N/rad；运行中会用传感器数据在线估计。
MPC_FORCE_GAIN_INIT = 8.0
# 末端加速度对接触力的简化扰动增益；第一版默认保守设 0。
MPC_ACCEL_GAIN_INIT = 0.0
# 在线估计力-位置增益的更新率。
MPC_GAIN_UPDATE_RATE = 0.02
# 触觉力低通滤波系数；越接近 1 越平滑，响应也越慢。
FORCE_FILTER_ALPHA = 0.88
FINGER_JOINTS = {
    "if_proximal_link",
    "mf_proximal_link",
    "rf_proximal_link",
    "lf_proximal_link",
    "th_root_link",
    "th_slider_connecting_link",
}
CONTACT_ATTRS = {
    "contype": "1",
    "conaffinity": "1",
    "condim": "4",
    "friction": "2.5 0.08 0.008",
    "solimp": "0.995 0.999 0.0005",
    "solref": "0.004 1",
    "margin": "0.0002",
    "gap": "0",
}
# 触觉传感器位点配置
# 键: 手指名称, 值: (所属连杆名称, 传感器位置偏移, 传感器尺寸)
TACTILE_SITES = {
    "index": ("if_distal_link", "0.007 0.03482 -0.02152", "0.016"),
    "middle": ("mf_distal_link", "0.007 0.03726 -0.02346", "0.0125"),
    "ring": ("rf_distal_link", "0.007 0.03349 -0.0235", "0.015"),
    "little": ("lf_distal_link", "0.007 0.02295 -0.01689", "0.0125"),
    "thumb": ("th_distal_link", "0.007 0.01042 -0.00172", "0.0145"),
}
# TACTILE_SITES = {
#     "index": ("if_distal_link", "0.007 0.026 -0.003", "0.012"),
#     "middle": ("mf_distal_link", "0.007 0.030 -0.003", "0.012"),
#     "ring": ("rf_distal_link", "0.007 0.026 -0.003", "0.012"),
#     "little": ("lf_distal_link", "0.007 0.022 -0.003", "0.011"),
#     "thumb": ("th_distal_link", "0.007 0.000 -0.020", "0.012"),
# }
TACTILE_SENSOR_NAMES = tuple(f"{name}_touch" for name in TACTILE_SITES)
TACTILE_SITE_NAMES = tuple(f"{name}_tip_touch_site" for name in TACTILE_SITES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open an AP001 cube grasp scene with MPC force tracking."
    )
    parser.add_argument(
        "--scene",
        type=Path,
        default=DEFAULT_SCENE_PATH,
        help=f"Generated scene XML path. Default: {DEFAULT_SCENE_PATH}",
    )
    parser.add_argument(
        "--source-hand",
        type=Path,
        default=SOURCE_HAND_XML,
        help=f"Source AP001 left-hand XML. Default: {SOURCE_HAND_XML}",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Disable the automatic MPC grasp and lift controller.",
    )
    return parser.parse_args()


def indent(element: ET.Element, level: int = 0) -> None:
    space = "\n" + level * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = space + "  "
        for child in element:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = space
    if level and (not element.tail or not element.tail.strip()):
        element.tail = space


def retarget_mesh_paths(hand_root: ET.Element, source_hand_xml: Path, scene_path: Path) -> None:
    source_model_dir = source_hand_xml.parent
    scene_dir = scene_path.parent
    for mesh in hand_root.findall("./asset/mesh"):
        mesh.attrib.pop("content_type", None)
        mesh_file = mesh.get("file")
        if mesh_file:
            mesh_path = (source_model_dir / mesh_file).resolve()
            mesh.set("file", os.path.relpath(mesh_path, scene_dir))


def stabilize_hand_tree(hand_root: ET.Element) -> None:
    worldbody = hand_root.find("worldbody")
    if worldbody is None:
        return

    for body in worldbody.findall(".//body"):
        body.set("gravcomp", "1")

    for geom in worldbody.findall(".//geom"):
        geom.attrib.update(CONTACT_ATTRS)

    for joint in worldbody.findall(".//joint"):
        if joint.get("name") in FINGER_JOINTS:
            joint.set("damping", "0.22")
            joint.set("armature", "0.0015")
            continue

        if joint.get("type") == "slide":
            joint.set("damping", max_float_text(joint.get("damping"), 0.3))
            joint.set("armature", max_float_text(joint.get("armature"), 0.002))
        else:
            joint.set("damping", max_float_text(joint.get("damping"), 0.2))
            joint.set("armature", max_float_text(joint.get("armature"), 0.001))


def add_tactile_sites(worldbody: ET.Element) -> None:
    for finger_name, (body_name, pos, size) in TACTILE_SITES.items():
        body = find_body(worldbody, body_name)
        if body is None:
            raise ValueError(f"Fingertip body not found for tactile site: {body_name}")

        ET.SubElement(
            body,
            "site",
            {
                "name": f"{finger_name}_tip_touch_site",
                "type": "sphere",
                "pos": pos,
                "size": size,
                "rgba": "1 0.15 0.05 0.45",
            },
        )


def find_body(root: ET.Element, name: str) -> ET.Element | None:
    for body in root.findall(".//body"):
        if body.get("name") == name:
            return body
    return None


def add_tactile_sensors(scene: ET.Element) -> None:
    sensor = ET.SubElement(scene, "sensor")
    for finger_name in TACTILE_SITES:
        ET.SubElement(
            sensor,
            "touch",
            {
                "name": f"{finger_name}_touch",
                "site": f"{finger_name}_tip_touch_site",
            },
        )


def max_float_text(current: str | None, minimum: float) -> str:
    try:
        value = float(current) if current is not None else minimum
    except ValueError:
        value = minimum
    return f"{max(value, minimum):g}"


def add_hand_pose_actuators(actuator: ET.Element) -> None:
    pose_actuators = (
        ("hand_x", "hand_x", "5000", "-0.50 0.50"),
        ("hand_y", "hand_y", "5000", "-0.40 0.60"),
        ("hand_z", "hand_z", "5000", "-0.20 0.80"),
        ("hand_roll", "hand_roll", "1200", "-0.8 0.8"),
        ("hand_pitch", "hand_pitch", "1200", "-0.8 0.8"),
        ("hand_yaw", "hand_yaw", "1200", "-1.2 1.2"),
    )
    for name, joint, kp, ctrlrange in pose_actuators:
        ET.SubElement(
            actuator,
            "position",
            {
                "name": name,
                "joint": joint,
                "kp": kp,
                "ctrlrange": ctrlrange,
                "ctrllimited": "true",
            },
        )


def build_manual_scene(source_hand_xml: Path, scene_path: Path) -> None:
    hand_tree = ET.parse(source_hand_xml)
    hand_root = hand_tree.getroot()
    retarget_mesh_paths(hand_root, source_hand_xml, scene_path)
    stabilize_hand_tree(hand_root)

    compiler = hand_root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler", {"angle": "radian", "autolimits": "true"})
    compiler.set("autolimits", "true")

    scene = ET.Element("mujoco", {"model": "ap001_left_mpc_grasp_scene"})
    scene.append(compiler)
    ET.SubElement(
        scene,
        "option",
        {
            "timestep": "0.002",
            "gravity": "0 0 -9.81",
            "integrator": "implicitfast",
            "iterations": "100",
            "ls_iterations": "50",
        },
    )
    ET.SubElement(scene, "statistic", {"center": "0 0.1 0.03", "extent": "0.35"})

    visual = ET.SubElement(scene, "visual")
    ET.SubElement(visual, "quality", {"shadowsize": "2048"})
    ET.SubElement(
        visual,
        "headlight",
        {
            "diffuse": "0.7 0.7 0.7",
            "ambient": "0.25 0.25 0.25",
            "specular": "0.2 0.2 0.2",
        },
    )
    ET.SubElement(visual, "map", {"znear": "0.01"})

    asset = ET.SubElement(scene, "asset")
    source_asset = hand_root.find("asset")
    if source_asset is not None:
        for child in list(source_asset):
            asset.append(child)
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "table_grid",
            "type": "2d",
            "builtin": "checker",
            "width": "256",
            "height": "256",
            "rgb1": "0.78 0.80 0.76",
            "rgb2": "0.58 0.62 0.58",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "table_mat",
            "texture": "table_grid",
            "texrepeat": "5 5",
            "rgba": "0.72 0.74 0.70 1",
        },
    )
    ET.SubElement(asset, "material", {"name": "cube_mat", "rgba": "0.15 0.50 0.88 1"})

    worldbody = ET.SubElement(scene, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        {"name": "key_light", "pos": "-0.35 -0.25 0.6", "dir": "0.6 0.6 -1", "directional": "true"},
    )
    ET.SubElement(
        worldbody,
        "camera",
        {"name": "grasp_view", "pos": "0.22 -0.24 0.22", "xyaxes": "0.73 0.68 0 -0.32 0.34 0.88"},
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "table_top",
            "type": "box",
            "pos": "0 0.1 -0.045",
            "size": "0.22 0.22 0.015",
            "material": "table_mat",
            "contype": "1",
            "conaffinity": "1",
            "condim": "4",
            "friction": "1.5 0.05 0.005",
            "solimp": "0.995 0.999 0.0005",
            "solref": "0.004 1",
            "margin": "0.0002",
            "gap": "0",
        },
    )
    # 创建抓取立方体主体，位置在 (0.001, 0.145, -0.005) 米处
    cube = ET.SubElement(worldbody, "body", {"name": "grasp_cube", "pos": "0.01 0.145 -0.005"})
    ET.SubElement(cube, "freejoint", {"name": "cube_freejoint"})
    ET.SubElement(
        cube,
        "geom",
        {
            "name": "grasp_cube_geom",
            "type": "box",
            "size": "0.025 0.025 0.025",
            "material": "cube_mat",
            "mass": "0.08",
            "contype": "1",
            "conaffinity": "1",
            "condim": "4",
            "friction": "0.8 0.02 0.002",
            "solimp": "0.995 0.999 0.0005",
            "solref": "0.003 1",
            "margin": "0.0002",
            "gap": "0",
        },
    )

    hand_mount = ET.SubElement(worldbody, "body", {"name": "hand_mount", "pos": "0 0 0"})
    ET.SubElement(hand_mount, "joint", {"name": "hand_x", "type": "slide", "axis": "1 0 0", "damping": "1400", "armature": "1.0"})
    ET.SubElement(hand_mount, "joint", {"name": "hand_y", "type": "slide", "axis": "0 1 0", "damping": "1400", "armature": "1.0"})
    ET.SubElement(hand_mount, "joint", {"name": "hand_z", "type": "slide", "axis": "0 0 1", "damping": "1400", "armature": "1.0"})
    ET.SubElement(hand_mount, "joint", {"name": "hand_roll", "type": "hinge", "axis": "1 0 0", "damping": "350", "armature": "0.35"})
    ET.SubElement(hand_mount, "joint", {"name": "hand_pitch", "type": "hinge", "axis": "0 1 0", "damping": "350", "armature": "0.35"})
    ET.SubElement(hand_mount, "joint", {"name": "hand_yaw", "type": "hinge", "axis": "0 0 1", "damping": "350", "armature": "0.35"})
    source_worldbody = hand_root.find("worldbody")
    if source_worldbody is not None:
        add_tactile_sites(source_worldbody)
        for child in list(source_worldbody):
            hand_mount.append(child)

    for tag in ("contact", "equality"):
        section = hand_root.find(tag)
        if section is not None:
            scene.append(section)

    actuator = ET.SubElement(scene, "actuator")
    add_hand_pose_actuators(actuator)
    source_actuator = hand_root.find("actuator")
    if source_actuator is not None:
        for child in list(source_actuator):
            child.set("ctrllimited", "true")
            if child.get("name") in FINGER_ACTUATOR_KP:
                child.set("kp", FINGER_ACTUATOR_KP[child.get("name")])
            actuator.append(child)

    add_tactile_sensors(scene)
    indent(scene)
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(scene).write(scene_path, encoding="utf-8", xml_declaration=True)


def set_initial_controls(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    # 初始手的位姿设置（根据可视化调整）
    initial_ctrl = {
        "hand_x": 0.0,    # X轴位置 (左右方向)
        "hand_y": 0.03,    # Y轴位置 (前后方向)
        "hand_z": 0.1,    # Z轴位置 (上下方向)
        "hand_roll": -0.512, # 滚转角 (绕X轴旋转)
        "hand_pitch": 0.0,   # 俯仰角 (绕Y轴旋转)
        "hand_yaw": 0.0,     # 偏航角 (绕Z轴旋转)
    }
    for name, value in initial_ctrl.items():
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id >= 0:
            data.ctrl[actuator_id] = value
            set_actuated_joint_qpos(model, data, actuator_id, value)

    for name in FINGER_ACTUATORS:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id >= 0:
            low, high = model.actuator_ctrlrange[actuator_id]
            value = INITIAL_FINGER_CTRL[name]
            value = min(high, max(low, value))
            data.ctrl[actuator_id] = value
            set_actuated_joint_qpos(model, data, actuator_id, value)

    mujoco.mj_forward(model, data)


def make_zero_hand_reference_data(model: mujoco.MjModel) -> mujoco.MjData:
    reference_data = mujoco.MjData(model)
    set_initial_controls(model, reference_data)
    for name in HAND_POSE_ACTUATORS:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id >= 0:
            reference_data.ctrl[actuator_id] = 0.0
            set_actuated_joint_qpos(model, reference_data, actuator_id, 0.0)

    mujoco.mj_forward(model, reference_data)
    return reference_data


def set_actuated_joint_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_id: int,
    value: float,
) -> None:
    joint_id = model.actuator_trnid[actuator_id, 0]
    if joint_id < 0:
        return

    qpos_id = model.jnt_qposadr[joint_id]
    data.qpos[qpos_id] = value


def rotation_matrix_to_rpy(matrix: np.ndarray) -> np.ndarray:
    pitch = np.arctan2(-matrix[2, 0], np.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2))
    roll = np.arctan2(matrix[2, 1], matrix[2, 2])
    yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    return np.array([roll, pitch, yaw])


class GraspStatePlot:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        reference_data: mujoco.MjData,
    ) -> None:
        self.sensor_ids = []
        self.labels = []
        for sensor_name in TACTILE_SENSOR_NAMES:
            sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
            if sensor_id < 0:
                raise ValueError(f"Tactile sensor not found: {sensor_name}")
            self.sensor_ids.append(sensor_id)
            self.labels.append(sensor_name.removesuffix("_touch"))

        self.hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_mount")
        if self.hand_body_id < 0:
            raise ValueError("Body not found: hand_mount")

        self.base_pos = reference_data.xpos[self.hand_body_id].copy()
        self.base_mat = reference_data.xmat[self.hand_body_id].reshape(3, 3).copy()
        self.prev_pos = data.xpos[self.hand_body_id].copy()
        self.prev_vel = np.zeros(3)
        self.accel = np.zeros(3)
        plt.ion()
        self.fig, (self.force_ax, self.state_ax) = plt.subplots(
            2,
            1,
            figsize=(10, 7),
            gridspec_kw={"height_ratios": [1, 1.7]},
        )
        self.bars = self.force_ax.bar(self.labels, np.zeros(len(self.labels)), color="#2f8fd8")
        self.force_ax.set_title("AP001 Fingertip Touch Force")
        self.force_ax.set_ylabel("Force (N)")
        self.force_ax.set_ylim(0, 5)
        self.force_ax.grid(axis="y", alpha=0.25)
        self.state_ax.axis("off")
        self.state_text = self.state_ax.text(
            0.01,
            0.98,
            "",
            va="top",
            family="monospace",
            fontsize=9,
        )
        self.fig.tight_layout()
        self.last_update_time = time.time()
        self.last_draw_time = 0.0

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        now = time.time()
        dt = max(now - self.last_update_time, 1e-6)
        current_pos = data.xpos[self.hand_body_id].copy()
        current_vel = (current_pos - self.prev_pos) / dt
        raw_accel = (current_vel - self.prev_vel) / dt
        self.accel = 0.8 * self.accel + 0.2 * raw_accel
        self.prev_pos = current_pos
        self.prev_vel = current_vel
        self.last_update_time = now

        if now - self.last_draw_time < PLOT_UPDATE_INTERVAL:
            return

        forces = []
        for sensor_id in self.sensor_ids:
            adr = model.sensor_adr[sensor_id]
            dim = model.sensor_dim[sensor_id]
            forces.append(float(np.linalg.norm(data.sensordata[adr : adr + dim])))

        max_force = max(5.0, max(forces) * 1.25 if forces else 5.0)
        self.force_ax.set_ylim(0, max_force)
        for bar, force in zip(self.bars, forces):
            bar.set_height(force)

        self.state_text.set_text(self.format_state(model, data))
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.last_draw_time = now

    def format_state(self, model: mujoco.MjModel, data: mujoco.MjData) -> str:
        rows = [
            "reference: hand_x/y/z/roll/pitch/yaw = 0",
            "touch force (N): "
            + "  ".join(
                f"{label}={self.force_value(model, data, sensor_id):.3f}"
                for label, sensor_id in zip(self.labels, self.sensor_ids)
            ),
            "",
            "body        dx      dy      dz    roll   pitch     yaw      ax      ay      az",
            "           (m)     (m)     (m)    (rad)   (rad)   (rad)   (m/s2)  (m/s2)  (m/s2)",
        ]
        pos = data.xpos[self.hand_body_id]
        mat = data.xmat[self.hand_body_id].reshape(3, 3)
        rel_pos = self.base_mat.T @ (pos - self.base_pos)
        rel_mat = self.base_mat.T @ mat
        rel_rpy = rotation_matrix_to_rpy(rel_mat)
        rel_accel = self.base_mat.T @ self.accel
        rows.append(
            f"hand_mount {rel_pos[0]:7.4f} {rel_pos[1]:7.4f} {rel_pos[2]:7.4f} "
            f"{rel_rpy[0]:7.3f} {rel_rpy[1]:7.3f} {rel_rpy[2]:7.3f} "
            f"{rel_accel[0]:7.2f} {rel_accel[1]:7.2f} {rel_accel[2]:7.2f}"
        )
        return "\n".join(rows)

    @staticmethod
    def force_value(model: mujoco.MjModel, data: mujoco.MjData, sensor_id: int) -> float:
        adr = model.sensor_adr[sensor_id]
        dim = model.sensor_dim[sensor_id]
        return float(np.linalg.norm(data.sensordata[adr : adr + dim]))


class FingerForceMPC:
    def __init__(self, ctrl_range: np.ndarray) -> None:
        self.ctrl_low = float(ctrl_range[0])
        self.ctrl_high = float(ctrl_range[1])
        self.force_gain = MPC_FORCE_GAIN_INIT
        self.accel_gain = MPC_ACCEL_GAIN_INIT
        self.prev_force: float | None = None
        self.prev_ctrl: float | None = None
        self.prev_delta = 0.0

    def update_model(self, force: float, ctrl: float) -> None:
        if self.prev_force is not None and self.prev_ctrl is not None:
            delta_ctrl = ctrl - self.prev_ctrl
            delta_force = force - self.prev_force
            if abs(delta_ctrl) > 1e-5:
                estimated_gain = delta_force / delta_ctrl
                if np.isfinite(estimated_gain) and estimated_gain > 0.0:
                    estimated_gain = float(np.clip(estimated_gain, 0.5, 80.0))
                    self.force_gain = (
                        (1.0 - MPC_GAIN_UPDATE_RATE) * self.force_gain
                        + MPC_GAIN_UPDATE_RATE * estimated_gain
                    )

        self.prev_force = force
        self.prev_ctrl = ctrl

    def solve(self, force: float, ctrl: float, future_hand_accels: np.ndarray) -> float:
        best_delta = 0.0
        best_cost = float("inf")

        for delta in MPC_DELTA_CANDIDATES:
            predicted_force = force
            predicted_ctrl = ctrl
            last_delta = self.prev_delta
            cost = 0.0

            for hand_accel in future_hand_accels[:MPC_HORIZON]:
                predicted_ctrl = float(
                    np.clip(predicted_ctrl + delta, self.ctrl_low, self.ctrl_high)
                )
                predicted_force = max(
                    0.0,
                    predicted_force
                    + self.force_gain * delta
                    + self.accel_gain * hand_accel * MPC_DT,
                )
                force_error = predicted_force - TARGET_GRASP_FORCE
                cost += MPC_FORCE_WEIGHT * force_error * force_error
                cost += MPC_CTRL_WEIGHT * delta * delta
                cost += MPC_SMOOTH_WEIGHT * (delta - last_delta) ** 2
                last_delta = delta

            if cost < best_cost:
                best_cost = cost
                best_delta = float(delta)

        self.prev_delta = best_delta
        return float(np.clip(ctrl + best_delta, self.ctrl_low, self.ctrl_high))


class ThreeFingerMPCGraspController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.finger_actuator_ids = {
            finger: self.required_actuator_id(actuator_name)
            for finger, actuator_name in ACTUATOR_BY_FORCE_FINGER.items()
        }
        self.sensor_ids = {
            finger: self.required_sensor_id(sensor_name)
            for finger, sensor_name in FORCE_SENSOR_BY_FINGER.items()
        }
        self.hand_z_actuator_id = self.required_actuator_id("hand_z")
        self.initial_hand_z = float(data.ctrl[self.hand_z_actuator_id])
        self.target_hand_z = min(
            self.model.actuator_ctrlrange[self.hand_z_actuator_id, 1],
            self.initial_hand_z + LIFT_HEIGHT,
        )
        self.lift_amplitude = 0.5 * (self.target_hand_z - self.initial_hand_z)
        self.lift_omega = np.sqrt(
            END_EFFECTOR_MAX_ACCEL / max(self.lift_amplitude, 1e-6)
        )
        self.lift_start_time = 0.0
        self.next_mpc_time = 0.0
        self.mpcs = {
            finger: FingerForceMPC(self.model.actuator_ctrlrange[actuator_id])
            for finger, actuator_id in self.finger_actuator_ids.items()
        }
        self.filtered_forces = {finger: 0.0 for finger in self.finger_actuator_ids}
        self.phase = "closing"

    def required_actuator_id(self, name: str) -> int:
        actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            raise ValueError(f"Actuator not found: {name}")
        return actuator_id

    def required_sensor_id(self, name: str) -> int:
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sensor_id < 0:
            raise ValueError(f"Touch sensor not found: {name}")
        return sensor_id

    def update(self, data: mujoco.MjData) -> None:
        timestep = float(self.model.opt.timestep)
        forces = self.filtered_finger_forces(data)

        if self.phase == "closing":
            self.close_until_contact(data, forces, timestep)
            if all(force >= TARGET_GRASP_FORCE - FORCE_TOLERANCE for force in forces.values()):
                self.lift_start_time = float(data.time)
                self.phase = "lifting"
        else:
            self.cycle_hand_z(data)
            if float(data.time) >= self.next_mpc_time:
                self.regulate_force_with_mpc(data, forces)
                self.next_mpc_time = float(data.time) + MPC_DT

    def finger_forces(self, data: mujoco.MjData) -> dict[str, float]:
        return {
            finger: GraspStatePlot.force_value(self.model, data, sensor_id)
            for finger, sensor_id in self.sensor_ids.items()
        }

    def filtered_finger_forces(self, data: mujoco.MjData) -> dict[str, float]:
        raw_forces = self.finger_forces(data)
        for finger, force in raw_forces.items():
            self.filtered_forces[finger] = (
                FORCE_FILTER_ALPHA * self.filtered_forces[finger]
                + (1.0 - FORCE_FILTER_ALPHA) * force
            )
        return dict(self.filtered_forces)

    def close_until_contact(
        self,
        data: mujoco.MjData,
        forces: dict[str, float],
        timestep: float,
    ) -> None:
        for finger, actuator_id in self.finger_actuator_ids.items():
            if forces[finger] < TARGET_GRASP_FORCE - FORCE_TOLERANCE:
                self.add_ctrl(data, actuator_id, FINGER_CLOSE_RATE * timestep)
            else:
                self.add_ctrl(data, actuator_id, -0.25 * FINGER_CLOSE_RATE * timestep)

    def regulate_force_with_mpc(self, data: mujoco.MjData, forces: dict[str, float]) -> None:
        future_hand_accels = self.future_hand_z_accels(float(data.time))
        for finger, actuator_id in self.finger_actuator_ids.items():
            current_ctrl = float(data.ctrl[actuator_id])
            current_force = forces[finger]
            mpc = self.mpcs[finger]
            mpc.update_model(current_force, current_ctrl)
            data.ctrl[actuator_id] = mpc.solve(
                current_force,
                current_ctrl,
                future_hand_accels,
            )

    def cycle_hand_z(self, data: mujoco.MjData) -> None:
        elapsed = max(0.0, float(data.time) - self.lift_start_time)
        data.ctrl[self.hand_z_actuator_id] = (
            self.initial_hand_z
            + self.lift_amplitude * (1.0 - np.cos(self.lift_omega * elapsed))
        )

    def future_hand_z_accels(self, current_time: float) -> np.ndarray:
        future_accels = []
        for step in range(MPC_HORIZON):
            elapsed = max(
                0.0,
                current_time - self.lift_start_time + step * MPC_DT,
            )
            future_accels.append(
                self.lift_amplitude * self.lift_omega**2 * np.cos(self.lift_omega * elapsed)
            )
        return np.array(future_accels)

    def add_ctrl(self, data: mujoco.MjData, actuator_id: int, delta: float) -> None:
        low, high = self.model.actuator_ctrlrange[actuator_id]
        data.ctrl[actuator_id] = min(high, max(low, data.ctrl[actuator_id] + delta))


def main() -> None:
    args = parse_args()
    scene_path = args.scene.expanduser().resolve()
    source_hand_xml = args.source_hand.expanduser().resolve()
    if not source_hand_xml.exists():
        raise FileNotFoundError(f"AP001 left-hand XML not found: {source_hand_xml}")

    build_manual_scene(source_hand_xml, scene_path)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    set_initial_controls(model, data)
    reference_data = make_zero_hand_reference_data(model)
    tactile_plot = GraspStatePlot(model, data, reference_data)
    grasp_controller = None if args.manual else ThreeFingerMPCGraspController(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 设置为默认自由相机模式
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

        while viewer.is_running():
            for _ in range(SIM_STEPS_PER_RENDER):
                if grasp_controller is not None:
                    grasp_controller.update(data)
                mujoco.mj_step(model, data)
            tactile_plot.update(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep * SIM_STEPS_PER_RENDER)


if __name__ == "__main__":
    main()
