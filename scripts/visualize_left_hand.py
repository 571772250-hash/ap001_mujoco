#!/usr/bin/env python3
"""Visualize the AP001 left hand model in MuJoCo.
/mnt/DOCUMENT/ap001_mujoco/.venv/bin/python -m py_compile ...
/mnt/DOCUMENT/ap001_mujoco/scripts/run_left_hand_viewer.sh --animate
/mnt/DOCUMENT/ap001_mujoco/scripts/run_left_hand_viewer.sh


"""

from __future__ import annotations

import argparse
import math
import tempfile
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import mujoco
import mujoco.viewer


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = (
    REPO_ROOT / "assets" / "AP001" / "model" / "rohand_gen2_left.xml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a MuJoCo viewer for the AP001 left hand model."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the MuJoCo XML model. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Drive the hand actuators with a slow open-close motion.",
    )
    return parser.parse_args()


def set_default_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Start from a lightly flexed pose so the model is easier to inspect."""
    for actuator_id in range(model.nu):
        low, high = model.actuator_ctrlrange[actuator_id]
        data.ctrl[actuator_id] = low + 0.35 * (high - low)


def update_animation(model: mujoco.MjModel, data: mujoco.MjData, elapsed: float) -> None:
    phase = 0.5 + 0.5 * math.sin(elapsed * 1.2)
    for actuator_id in range(model.nu):
        low, high = model.actuator_ctrlrange[actuator_id]
        data.ctrl[actuator_id] = low + phase * (high - low)


@contextmanager
def compatible_xml(model_path: Path) -> Iterator[Path]:
    """Create a temporary XML copy compatible with MuJoCo 2.3.x."""
    tree = ET.parse(model_path)
    root = tree.getroot()
    removed = False
    for mesh in root.findall(".//mesh"):
        if "content_type" in mesh.attrib:
            del mesh.attrib["content_type"]
            removed = True

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    if compiler.attrib.get("autolimits") != "true":
        compiler.set("autolimits", "true")
        removed = True

    if not removed:
        raise RuntimeError(f"No compatibility changes were needed for {model_path}")

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".xml",
        prefix=f"{model_path.stem}_compat_",
        dir=model_path.parent,
        delete=True,
    ) as temp_file:
        tree.write(temp_file, encoding="unicode")
        temp_file.flush()
        yield Path(temp_file.name)


def load_model(model_path: Path) -> mujoco.MjModel:
    try:
        return mujoco.MjModel.from_xml_path(str(model_path))
    except ValueError as error:
        if "unrecognized attribute: 'content_type'" not in str(error):
            raise

        with compatible_xml(model_path) as xml_path:
            return mujoco.MjModel.from_xml_path(str(xml_path))


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo model XML not found: {model_path}")

    model = load_model(model_path)
    data = mujoco.MjData(model)
    set_default_pose(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 0.45
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
        viewer.cam.lookat[:] = [0.0, 0.07, 0.02]

        start_time = time.time()
        while viewer.is_running():
            if args.animate:
                update_animation(model, data, time.time() - start_time)

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
