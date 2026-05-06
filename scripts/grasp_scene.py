#!/usr/bin/env python3
"""Launch a manual AP001 left-hand grasp scene for MuJoCo Control tuning.
python /mnt/DOCUMENT/ap001_mujoco/scripts/grasp_scene.py

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
DEFAULT_SCENE_PATH = SCENE_DIR / "ap001_left_grasp_manual_scene.generated.xml"
FINGER_ACTUATORS = (
    "index_finger",
    "middle_finger",
    "ring_finger",
    "little_finger",
    "thumb_root",
    "thumb",
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
    "index_finger": "12",
    "middle_finger": "12",
    "ring_finger": "12",
    "little_finger": "12",
    "thumb_root": "10",
    "thumb": "10",
}
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
TACTILE_SITES = {
    "index": ("if_distal_link", "0.007 0.05 -0.01", "0.03"),
    "middle": ("mf_distal_link", "0.007 0.06 -0.01", "0.03"),
    "ring": ("rf_distal_link", "0.007 0.05 -0.01", "0.03"),
    "little": ("lf_distal_link", "0.007 0.04 -0.01", "0.03"),
    "thumb": ("th_distal_link", "0.007 0.000 -0.020", "0.03"),
}
# TACTILE_SITES = {
#     "index": ("if_distal_link", "0.007 0.026 -0.003", "0.012"),
#     "middle": ("mf_distal_link", "0.007 0.030 -0.003", "0.012"),
#     "ring": ("rf_distal_link", "0.007 0.026 -0.003", "0.012"),
#     "little": ("lf_distal_link", "0.007 0.022 -0.003", "0.011"),
#     "thumb": ("th_distal_link", "0.007 0.000 -0.020", "0.012"),
# }
TACTILE_SENSOR_NAMES = tuple(f"{name}_touch" for name in TACTILE_SITES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a manual AP001 cube grasp scene. Use the viewer Control panel."
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
            joint.set("damping", "0.45")
            joint.set("armature", "0.004")
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
        ("hand_x", "hand_x", "500", "-0.50 0.50"),
        ("hand_y", "hand_y", "500", "-0.40 0.60"),
        ("hand_z", "hand_z", "500", "-0.20 0.80"),
        ("hand_roll", "hand_roll", "120", "-0.8 0.8"),
        ("hand_pitch", "hand_pitch", "120", "-0.8 0.8"),
        ("hand_yaw", "hand_yaw", "120", "-1.2 1.2"),
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

    scene = ET.Element("mujoco", {"model": "ap001_left_manual_grasp_scene"})
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
    cube = ET.SubElement(worldbody, "body", {"name": "grasp_cube", "pos": "0.001 0.145 -0.005"})
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
            "friction": "3.0 0.1 0.01",
            "solimp": "0.995 0.999 0.0005",
            "solref": "0.003 1",
            "margin": "0.0002",
            "gap": "0",
        },
    )

    hand_mount = ET.SubElement(worldbody, "body", {"name": "hand_mount", "pos": "0 0 0"})
    ET.SubElement(hand_mount, "joint", {"name": "hand_x", "type": "slide", "axis": "1 0 0", "damping": "220", "armature": "0.25"})
    ET.SubElement(hand_mount, "joint", {"name": "hand_y", "type": "slide", "axis": "0 1 0", "damping": "220", "armature": "0.25"})
    ET.SubElement(hand_mount, "joint", {"name": "hand_z", "type": "slide", "axis": "0 0 1", "damping": "220", "armature": "0.25"})
    ET.SubElement(hand_mount, "joint", {"name": "hand_roll", "type": "hinge", "axis": "1 0 0", "damping": "55", "armature": "0.08"})
    ET.SubElement(hand_mount, "joint", {"name": "hand_pitch", "type": "hinge", "axis": "0 1 0", "damping": "55", "armature": "0.08"})
    ET.SubElement(hand_mount, "joint", {"name": "hand_yaw", "type": "hinge", "axis": "0 0 1", "damping": "55", "armature": "0.08"})
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


class TactilePlot:
    def __init__(self, model: mujoco.MjModel) -> None:
        self.sensor_ids = []
        self.labels = []
        for sensor_name in TACTILE_SENSOR_NAMES:
            sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
            if sensor_id < 0:
                raise ValueError(f"Tactile sensor not found: {sensor_name}")
            self.sensor_ids.append(sensor_id)
            self.labels.append(sensor_name.removesuffix("_touch"))

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7, 4))
        self.bars = self.ax.bar(self.labels, np.zeros(len(self.labels)), color="#2f8fd8")
        self.ax.set_title("AP001 Fingertip Normal Force")
        self.ax.set_ylabel("Force (N)")
        self.ax.set_ylim(0, 5)
        self.ax.grid(axis="y", alpha=0.25)
        self.fig.tight_layout()
        self.last_update = 0.0

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        now = time.time()
        if now - self.last_update < 0.03:
            return

        forces = []
        for sensor_id in self.sensor_ids:
            adr = model.sensor_adr[sensor_id]
            dim = model.sensor_dim[sensor_id]
            forces.append(float(np.linalg.norm(data.sensordata[adr : adr + dim])))

        max_force = max(5.0, max(forces) * 1.25 if forces else 5.0)
        self.ax.set_ylim(0, max_force)
        for bar, force in zip(self.bars, forces):
            bar.set_height(force)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.last_update = now


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
    tactile_plot = TactilePlot(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 设置为默认自由相机模式
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

        while viewer.is_running():
            mujoco.mj_step(model, data)
            tactile_plot.update(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
