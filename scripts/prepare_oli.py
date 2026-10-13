"""Build an experimental serial HU_D04 model with frozen end effectors.

Sources are pinned to the official LimX description repository. This is NOT
the manufacturer's closed-chain motor-space MJCF and is for simulation only.
"""

import argparse
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
import time
from pathlib import Path
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET

import mujoco

from common.paths import PROJECT_ROOT

DESCRIPTION_REV = "80fbd2d650c43c38916979c2a6eab7a4c79e38de"
DEPLOY_REV = "bc76b0fe8399ecb84ce5492a6c385feb7d7a17a5"
SOURCE = f"https://raw.githubusercontent.com/limxdynamics/humanoid-description/{DESCRIPTION_REV}"


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    data = response.read()
                break
            except (urllib.error.URLError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        path.write_bytes(data)


def joint_parameters(name, limit, rotor):
    if any(part in name for part in ("hip_", "knee")):
        kp, kd = 139.41, 17.75
    elif any(part in name for part in ("ankle_", "waist_")):
        kp, kd = 93.65, 11.92
    elif any(part in name for part in ("shoulder_", "elbow")):
        kp, kd = 87.51, 11.14
    else:
        kp, kd = 15.12, 1.93
    defaults = {
        "hip_pitch": -0.15, "knee": 0.30, "ankle_pitch": -0.16,
        "shoulder_pitch": 0.10, "elbow": -0.20,
    }
    q0 = next((value for part, value in defaults.items() if part in name), 0.0)
    sign = 1 if name.startswith("left_") else -1
    for part, value in (("hip_yaw", -0.05), ("shoulder_roll", 0.10), ("shoulder_yaw", -0.20)):
        if part in name:
            q0 = sign * value
    effort = float(limit.attrib["effort"])
    return dict(
        stiffness=kp, damping=kd, effort_limit=effort,
        armature=float(rotor.attrib["mass"]) * float(rotor.attrib["gear_ratio"]) ** 2,
        velocity_limit=float(limit.attrib["velocity"]),
        default_position=q0, action_scale=0.25 * effort / kp,
    )


def prepare(variant="gripper"):
    target = PROJECT_ROOT / "models" / "oli" / variant
    source_dir = target / "source"
    source_name = f"HU_D04_01_with_{variant}.urdf"
    download(f"{SOURCE}/HU_D04_description/urdf/{source_name}", source_dir / source_name)
    download(f"{SOURCE}/HU_D04_description/urdf/HU_D04_01.srdf", source_dir / "HU_D04_01.srdf")
    download(f"{SOURCE}/HU_D04_description/xml/HU_D04_01.xml", source_dir / "HU_D04_01.xml")
    download(f"{SOURCE}/LICENSE", target / "LICENSE")
    walk_url = (f"https://raw.githubusercontent.com/limxdynamics/humanoid-rl-deploy-python/"
                f"{DEPLOY_REV}/controllers/HU_D04_01/walk_controller/walk_param.yaml")
    download(walk_url, source_dir / "walk_param.yaml")
    download(f"https://raw.githubusercontent.com/limxdynamics/humanoid-rl-deploy-python/{DEPLOY_REV}/LICENSE",
             source_dir / "DEPLOY_LICENSE")
    root = ET.parse(source_dir / source_name).getroot()
    rotors = {j.attrib["name"]: j for j in ET.parse(source_dir / "HU_D04_01.srdf").getroot().iter("joint")}
    mesh_names = {Path(m.attrib["filename"]).name for m in root.iter("mesh")}
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda name: download(
            f"{SOURCE}/HU_D04_description/meshes/HU_D04_01/{name}", target / "meshes" / name
        ), sorted(mesh_names)))
    joints = {}
    frozen = []
    for joint in root.findall("joint"):
        name = joint.attrib["name"]
        if joint.attrib["type"] == "fixed":
            continue
        if name not in rotors:
            joint.set("type", "fixed")
            frozen.append(name)
            for tag in ("axis", "limit", "mimic", "dynamics"):
                for child in joint.findall(tag):
                    joint.remove(child)
        else:
            joints[name] = joint_parameters(name, joint.find("limit"), rotors[name])
    # Keep URDF foot contacts, inertials and visuals; add official body primitives below.
    for link in root.findall("link"):
        if link.attrib["name"] not in ("left_ankle_roll_link", "right_ankle_roll_link"):
            for collision in link.findall("collision"):
                link.remove(collision)
    for mesh in root.iter("mesh"):
        mesh.set("filename", Path(mesh.attrib["filename"]).name)
    ext = ET.SubElement(root, "mujoco")
    ET.SubElement(ext, "compiler", discardvisual="false", fusestatic="false", balanceinertia="true")
    assets = {name: (target / "meshes" / name).read_bytes() for name in mesh_names}
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"), assets=assets)
    base = spec.body("base_link")
    base.add_freejoint(name="floating_base_joint")
    spec.option.timestep = 0.005
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
    for body in spec.bodies:
        for i, geom in enumerate(body.geoms):
            collision = bool(geom.contype or geom.conaffinity)
            geom.name = f"{body.name}_{'collision' if collision else 'visual'}_{i}"
            if not collision:
                geom.contype = geom.conaffinity = 0
                geom.group = 1
            else:
                geom.group = 0
                geom.contype, geom.conaffinity = 0, 1
                geom.condim = 3
                geom.priority = 1
                geom.friction = [0.6, 0.005, 0.0001]
    for name, params in joints.items():
        j = spec.joint(name)
        j.armature = params["armature"]
        actuator = spec.add_actuator(name=name, target=name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT)
        actuator.set_to_position(kp=params["stiffness"], kv=params["damping"])
        actuator.forcelimited = True
        actuator.forcerange = [-params["effort_limit"], params["effort_limit"]]
    base.add_site(name="imu", pos=[0, 0, 0])
    for side in ("left", "right"):
        spec.body(f"{side}_ankle_roll_link").add_site(name=f"{side}_foot", pos=[0.018, 0, -0.0635])
    for name, sensor_type, obj_type, obj_name in (
        ("imu_lin_vel", mujoco.mjtSensor.mjSENS_VELOCIMETER, mujoco.mjtObj.mjOBJ_SITE, "imu"),
        ("imu_ang_vel", mujoco.mjtSensor.mjSENS_GYRO, mujoco.mjtObj.mjOBJ_SITE, "imu"),
        ("root_angmom", mujoco.mjtSensor.mjSENS_SUBTREEANGMOM, mujoco.mjtObj.mjOBJ_BODY, "base_link"),
    ):
        spec.add_sensor(name=name, type=sensor_type, objtype=obj_type, objname=obj_name)
    model = spec.compile()
    if (model.nq, model.nv, model.nu) != (38, 37, 31):
        raise ValueError(f"Unexpected serial OLI dimensions: {model.nq}, {model.nv}, {model.nu}")
    data = mujoco.MjData(model)
    data.qpos[3:7] = [1, 0, 0, 0]
    for name, params in joints.items():
        data.qpos[model.joint(name).qposadr[0]] = params["default_position"]
    mujoco.mj_forward(model, data)
    bottom = min(data.site_xpos[model.site(f"{side}_foot").id, 2] for side in ("left", "right"))
    height = float(-bottom + 0.001)
    # Make serialized assets relocatable, including on the Linux training host.
    xml = ET.fromstring(spec.to_xml())
    compiler = xml.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(xml, "compiler")
    compiler.set("meshdir", "meshes")
    for mesh in xml.iter("mesh"):
        mesh.set("file", Path(mesh.attrib["file"]).name)
    # The URDF has mostly foot collisions. Borrow named body-local primitives
    # from the official MJCF so a fallen torso cannot pass through the ground.
    # Self-collision remains disabled in this first migration.
    official_bodies = {b.attrib["name"]: b for b in ET.parse(source_dir / "HU_D04_01.xml").getroot().iter("body")}
    for body in xml.iter("body"):
        name = body.attrib["name"]
        original = official_bodies.get(name)
        if original is None or name in ("left_ankle_roll_link", "right_ankle_roll_link"):
            continue
        for i, geom in enumerate(original.findall("geom")):
            if geom.attrib.get("type") not in ("box", "capsule", "cylinder", "sphere") or geom.attrib.get("class") == "visual":
                continue
            added = deepcopy(geom)
            added.attrib.pop("class", None)
            added.attrib.update(name=f"{name}_collision_extra_{i}", contype="0", conaffinity="1",
                group="0", condim="1", priority="1", density="0", friction="0.6 0.005 0.0001")
            body.append(added)
    ET.indent(xml)
    ET.ElementTree(xml).write(target / "robot.xml", encoding="utf-8", xml_declaration=True)
    scene = ('<mujoco model="oli_scene"><include file="robot.xml"/>'
             '<worldbody><geom name="floor" type="plane" size="0 0 0.1" '
             'contype="1" conaffinity="1" friction="0.6 0.005 0.0001"/>'
             '<light pos="0 0 3"/></worldbody></mujoco>\n')
    (target / "scene.xml").write_text(scene, encoding="utf-8")
    metadata = dict(robot="oli", variant=variant, base_body="base_link", torso_body="waist_pitch_link",
        hand_bodies=["left_wrist_roll_link", "right_wrist_roll_link"], default_base_position=[0, 0, height],
        joints=joints, frozen_joints=frozen, description_revision=DESCRIPTION_REV, deployment_revision=DEPLOY_REV,
        model_kind="serial_urdf_frozen_end_effectors", contact_mode="ground_only_no_self_collision")
    (target / "robot.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    mujoco.MjModel.from_xml_string(scene, assets={
        "robot.xml": (target / "robot.xml").read_bytes(),
        **{f"meshes/{name}": data for name, data in assets.items()},
    })
    print(f"Prepared {target}: 31 actuators, {len(frozen)} frozen end-effector joints, height={height:.4f}m")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("gripper", "hand"), default="gripper")
    prepare(parser.parse_args().variant)


if __name__ == "__main__":
    main()
