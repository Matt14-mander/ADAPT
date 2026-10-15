"""Pre-attached OLI box carrying: a robot pair, or one robot using both arms."""

import argparse
from contextlib import nullcontext
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mode", choices=("pair", "solo"), help="Two-robot carry or one OLI holding the box with both arms")
    parser.add_argument("--solo-arm-test", action="store_true", help="Diagnostic: keep solo IK arm pose but park the unconnected box on the floor beside the robot")
    parser.add_argument("--solo-arm-pose-scale", type=float, help="Arm diagnostic pose: 0=policy defaults, 1=full IK pose; requires --solo-arm-test")
    parser.add_argument("--checkpoint", type=Path, help="OLI ONNX with a matching JSON; used independently by both robots")
    parser.add_argument("--checkpoint-b", type=Path, help="Optional second compatible OLI ONNX")
    parser.add_argument("--pd-check", action="store_true", help="Fixed-default-position diagnostic; does not validate locomotion")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--command", nargs=3, type=float, metavar=("VX", "VY", "WZ"))
    parser.add_argument("--box-mass", type=float)
    parser.add_argument("--separation", type=float)
    parser.add_argument("--connection", choices=("weld", "connect", "none"))
    parser.add_argument("--arm-hold", type=float, help="Blend carrying arm targets toward default (pair) or IK pose (solo), in [0,1]")
    parser.add_argument("--no-formation", action="store_true")
    parser.add_argument("--encoder-noise", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--realtime", action="store_true", help="Pace headless simulation to wall clock")
    parser.add_argument("--output", type=Path, help="New output directory; must not already exist")
    parser.add_argument("--video", action="store_true", help="Record MP4 at 30 FPS; requires imageio and ffmpeg")
    parser.add_argument("--pause-on-failure", action="store_true", help="Keep failed scene visible until window is closed")
    args = parser.parse_args()
    if args.pd_check and (args.checkpoint or args.checkpoint_b):
        parser.error("--pd-check cannot be combined with checkpoints")
    if not args.pd_check and args.checkpoint is None:
        parser.error("Provide --checkpoint, or explicitly select --pd-check")
    for name in ("checkpoint", "checkpoint_b"):
        path = getattr(args, name)
        if path is None:
            continue
        if not path.name:
            parser.error(f"--{name.replace('_', '-')} is empty or points to a directory; check that your shell policy variable is set")
        if path.suffix.lower() != ".onnx" or not path.is_file():
            parser.error(f"Expected an existing ONNX file: {path}")
        if not path.with_suffix(".json").is_file():
            parser.error(f"Missing matching policy JSON: {path.with_suffix('.json')}")
    if args.headless and args.pause_on_failure:
        parser.error("--pause-on-failure requires the viewer")
    if args.mode == "solo" and args.checkpoint_b:
        parser.error("--checkpoint-b is only available in pair mode")
    return parser, args


def update_viewer(viewer, simulation, record):
    import mujoco
    import numpy as np

    viewer.cam.lookat[:] = np.mean([s.qpos[:3] for s in simulation.states], axis=0)
    with viewer.lock():
        scene = viewer.user_scn
        scene.ngeom = 0
        for start, end in simulation.grasp.arrow_segments():
            if np.linalg.norm(end - start) < 1e-5 or scene.ngeom >= scene.maxgeom:
                continue
            end = start + np.clip(np.linalg.norm(end - start), 0, 0.5) * (end - start) / np.linalg.norm(end - start)
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                               np.array([1, 0.25, 0.1, 1], dtype=np.float32))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.008, start, end)
            scene.ngeom += 1
    viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
        f"OLI {simulation.cfg.mode}\nTime\nPhase\nBox mass\nFormation error\nBox tilt\nGrasp error\nState",
        f"Pre-attached carry\n{record['time']:.1f}s\n{record['phase']}\n{simulation.cfg.box_mass:.2f} kg\n"
        f"{record['formation_error']:.3f} m\n{record['box_tilt_deg']:.1f} deg\n"
        f"{max(record['grasp_position_error']):.3f} m\n{record['failure'] or 'running'}"))
    viewer.sync()


def main():
    parser, args = parse_args()
    if args.headless and args.video and sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    import numpy as np
    from deployment.cooperative.config import CarryConfig
    from deployment.cooperative.simulation import CarrySimulation
    from deployment.cooperative.telemetry import CarryTelemetry
    from deployment.policy_manager import ZeroPolicy, load_policy

    cfg = CarryConfig.load(args.config) if args.config else CarryConfig()
    if args.mode is not None:
        cfg.mode = args.mode
    if args.solo_arm_test:
        if cfg.mode != "solo":
            parser.error("--solo-arm-test requires --mode solo or the solo config")
        cfg.solo_arm_test = True
        cfg.connection = "none"
    if cfg.mode == "solo":
        if args.checkpoint_b:
            parser.error("--checkpoint-b is only available in pair mode")
        if args.arm_hold is None and (not args.config or args.mode == "solo"):
            cfg.arm_hold = 1.0
        if not args.config:
            cfg.box_size = (0.28, 0.30, 0.20)
            cfg.grasp_torque_scale = 0.5
            cfg.formation = False
    for name in ("seconds", "command", "box_mass", "separation", "connection", "arm_hold", "seed", "solo_arm_pose_scale"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    cfg.formation = cfg.formation and not args.no_formation
    cfg.encoder_noise = cfg.encoder_noise or args.encoder_noise
    cfg.validate()
    output = args.output or Path("outputs") / ("carry_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    if output.exists():
        parser.error(f"Output directory already exists: {output}")
    if args.pd_check:
        from scripts.check_oli import make_config
        policies = [ZeroPolicy(make_config(), name=f"pd_{i}") for i in range(1 if cfg.mode == "solo" else 2)]
        print("DIAGNOSTIC: fixed PD targets; this does not demonstrate trained locomotion.")
    else:
        policies = [load_policy(args.checkpoint, name="robot_a")]
        if cfg.mode == "pair":
            policies.append(load_policy(args.checkpoint_b or args.checkpoint, name="robot_b"))
    simulation = CarrySimulation(policies, cfg)
    if cfg.mode == "solo" and not cfg.solo_arm_test:
        print(f"SOLO: two pre-attached hands, IK arm hold={cfg.arm_hold:.2f}; walking with this pose requires policy validation.")
    if cfg.solo_arm_test:
        print("ARM-POSE DIAGNOSTIC: box parked beside robot, no grasp or payload support; grasp gaps and box metrics are not carry results.")
        print(f"Arm pose scale={cfg.solo_arm_pose_scale:.2f}, hold blend={cfg.arm_hold:.2f}")
    metadata = dict(config=cfg.as_dict(), diagnostic=args.pd_check or cfg.solo_arm_test,
        checkpoints=([str(args.checkpoint.resolve())] +
            ([str((args.checkpoint_b or args.checkpoint).resolve())] if cfg.mode == "pair" else [])) if args.checkpoint else [],
        policy_hashes=[p.cfg.model_sha256 for p in policies], mujoco_version=mujoco.__version__,
        joint_names=[p.cfg.joint_names for p in policies],
        arm_hold_targets=[a.hold_pose.tolist() for a in simulation.agents],
        robot="OLI HU_D04 gripper serial model", units="m, s, N, Nm; wrench in world frame about payload origin")
    telemetry = CarryTelemetry(output, metadata)
    renderer = writer = None
    steps_done = 0
    completed = False
    error = None
    interrupted = False
    try:
        if args.video:
            import imageio.v2 as imageio
            renderer = mujoco.Renderer(simulation.model, height=720, width=1280)
            writer = imageio.get_writer(str(output / "demo.mp4"), fps=30)
        if args.headless:
            context = nullcontext(None)
        else:
            import mujoco.viewer
            context = mujoco.viewer.launch_passive(simulation.model, simulation.data,
                                                   show_left_ui=False, show_right_ui=False)
        camera = mujoco.MjvCamera()
        camera.distance, camera.azimuth, camera.elevation = 3.6, 135, -15
        camera.lookat[:] = [0, 0, 0.9]
        next_video_time = 0.0
        next_print_time = 0.0
        record = simulation.record()
        telemetry.write(record)
        with context as viewer:
            if viewer:
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.6, 135, -15
            target_steps = int(np.floor(cfg.seconds / simulation.dt + 1e-8))
            while steps_done < target_steps:
                begin = time.monotonic()
                if viewer and not viewer.is_running():
                    break
                record = simulation.step()
                steps_done += 1
                telemetry.write(record)
                if viewer and viewer.is_running():
                    update_viewer(viewer, simulation, record)
                if renderer and simulation.data.time + 1e-8 >= next_video_time:
                    camera.lookat[:] = np.mean([s.qpos[:3] for s in simulation.states], axis=0)
                    renderer.update_scene(simulation.data, camera=camera)
                    writer.append_data(renderer.render())
                    next_video_time += 1 / 30
                if simulation.data.time >= next_print_time:
                    if cfg.mode == "solo":
                        payload_text = "box disconnected" if cfg.solo_arm_test else f"box tilt={record['box_tilt_deg']:.1f}deg grasp={max(record['grasp_position_error']):.3f}m"
                        print(f"t={record['time']:.1f}s {record['phase']} | {payload_text} "
                              f"robot tilt={max(record['robot_tilt_deg']):.1f}deg base={min(record['base_heights']):.3f}m "
                              f"arm error={max(record['arm_max_position_error']):.3f}rad effort={max(record['arm_max_effort_ratio']):.2f}")
                    else:
                        print(f"t={record['time']:.1f}s {record['phase']} | formation={record['formation_error']:.3f}m "
                              f"box tilt={record['box_tilt_deg']:.1f}deg grasp={max(record['grasp_position_error']):.3f}m")
                    next_print_time += 1
                if simulation.failure:
                    print(f"STOP: {simulation.failure}; failed state preserved, no automatic reset.")
                    if viewer and args.pause_on_failure:
                        while viewer.is_running():
                            update_viewer(viewer, simulation, record)
                            time.sleep(0.03)
                    break
                if viewer or args.realtime:
                    time.sleep(max(0, simulation.dt - (time.monotonic() - begin)))
            completed = steps_done == target_steps and simulation.failure is None
    except KeyboardInterrupt:
        interrupted = True
        print("Stopped by user; recording an incomplete run.")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if writer:
            writer.close()
        if renderer:
            renderer.close()
        summary = simulation.summary(completed, diagnostic=args.pd_check or cfg.solo_arm_test)
        summary.update(error=error, interrupted=interrupted)
        telemetry.finish(summary)
        print(json.dumps(summary, indent=2))
        print(f"Saved: {output}")
    if simulation.failure:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
