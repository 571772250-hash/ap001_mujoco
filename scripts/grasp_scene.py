#!/usr/bin/env python3
"""Launch a manual AP001 left-hand grasp scene for MuJoCo Control tuning."""

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

import mujoco
import mujoco.viewer


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
OPEN_FRACTIONS = (0.04, 0.04, 0.04, 0.04, 0.16, 0.08)


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


def add_hand_pose_actuators(actuator: ET.Element) -> None:
    pose_actuators = (
        ("hand_x", "hand_x", "80", "-0.08 0.08"),
        ("hand_y", "hand_y", "80", "0.02 0.20"),
        ("hand_z", "hand_z", "80", "-0.01 0.12"),
        ("hand_roll", "hand_roll", "20", "-0.8 0.8"),
        ("hand_pitch", "hand_pitch", "20", "-0.8 0.8"),
        ("hand_yaw", "hand_yaw", "20", "-1.2 1.2"),
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

    compiler = hand_root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler", {"angle": "radian", "autolimits": "true"})
    compiler.set("autolimits", "true")

    scene = ET.Element("mujoco", {"model": "ap001_left_manual_grasp_scene"})
    scene.append(compiler)
    ET.SubElement(scene, "option", {"timestep": "0.002", "gravity": "0 0 -9.81"})
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
            "friction": "1.2 0.02 0.002",
        },
    )

    hand_mount = ET.SubElement(worldbody, "body", {"name": "hand_mount", "pos": "0 0 0"})
    ET.SubElement(hand_mount, "joint", {"name": "hand_x", "type": "slide", "axis": "1 0 0", "damping": "2"})
    ET.SubElement(hand_mount, "joint", {"name": "hand_y", "type": "slide", "axis": "0 1 0", "damping": "2"})
    ET.SubElement(hand_mount, "joint", {"name": "hand_z", "type": "slide", "axis": "0 0 1", "damping": "2"})
    ET.SubElement(hand_mount, "joint", {"name": "hand_roll", "type": "hinge", "axis": "1 0 0", "damping": "0.5"})
    ET.SubElement(hand_mount, "joint", {"name": "hand_pitch", "type": "hinge", "axis": "0 1 0", "damping": "0.5"})
    ET.SubElement(hand_mount, "joint", {"name": "hand_yaw", "type": "hinge", "axis": "0 0 1", "damping": "0.5"})
    source_worldbody = hand_root.find("worldbody")
    if source_worldbody is not None:
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
            actuator.append(child)

    indent(scene)
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(scene).write(scene_path, encoding="utf-8", xml_declaration=True)


def set_initial_controls(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    initial_ctrl = {
        "hand_x": 0.0,
        "hand_y": 0.02,
        "hand_z": 0.0,
        "hand_roll": 0.0,
        "hand_pitch": 0.0,
        "hand_yaw": 0.0,
    }
    for name, value in initial_ctrl.items():
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id >= 0:
            data.ctrl[actuator_id] = value

    for name, fraction in zip(FINGER_ACTUATORS, OPEN_FRACTIONS):
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id >= 0:
            low, high = model.actuator_ctrlrange[actuator_id]
            data.ctrl[actuator_id] = low + fraction * (high - low)


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

    with mujoco.viewer.launch_passive(model, data) as viewer:
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "grasp_view")
        if camera_id >= 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = camera_id

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
