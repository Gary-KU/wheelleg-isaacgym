"""Fine tune the existing Actor in MuJoCo, with a fresh Critic/optimizer.

Run the physical audit first. Checkpoints remain separate from Gym and include
simulator/model hashes, sensor/control config, and initialization provenance.
"""

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim2sim.environment import MujocoWheelLeg
from legged_gym.learning.ppo_trainer import PPOTrainer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "exports/h723_small/checkpoint.pt",
    )
    p.add_argument("--stage", choices=("stand", "drive", "ramp"), default="stand")
    p.add_argument(
        "--leg-length", type=float, default=None, help="Default: checkpoint leg length"
    )
    p.add_argument("--learning-rate", type=float, default=3e-6)
    p.add_argument("--ramp-height", type=float, default=0.03)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--rollout-steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-root", type=Path, default=ROOT / "logs/mujoco_training")
    p.add_argument("--audit", type=Path, default=ROOT / "logs/mujoco/model_audit.json")
    args = p.parse_args()
    if min(args.iterations, args.num_envs, args.rollout_steps) < 1:
        p.error("iterations, num-envs and rollout-steps must be positive")
    audit = json.loads(args.audit.read_text())
    if not audit.get("passed"):
        raise RuntimeError(
            "Physical audit has not passed; training is intentionally not started."
        )
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = deepcopy(source["legged_gym_env_config"])
    if args.leg_length is not None:
        config["env"]["initial_leg_length"] = args.leg_length
    env = MujocoWheelLeg(
        config,
        args.num_envs,
        args.seed,
        training=args.stage,
        terrain="ramp" if args.stage == "ramp" else "flat",
        ramp_height=args.ramp_height,
        episode_seconds=6 if args.stage == "ramp" else 16,
    )
    model_hash = hashlib.sha256(
        (
            ROOT
            / (
                "assets/wheelleg/mujoco/ramp.xml"
                if args.stage == "ramp"
                else "assets/wheelleg/mujoco/flat.xml"
            )
        ).read_bytes()
    ).hexdigest()
    if args.stage != "ramp" and audit.get("model_sha256") != model_hash:
        raise RuntimeError("Audit/model mismatch; rerun check_model.py")
    train = deepcopy(source["training_config"])
    train["num_steps_per_env"] = args.rollout_steps
    train["save_interval"] = 10
    train["algorithm"].update(
        learning_rate=args.learning_rate,
        critic_learning_rate=3e-4,
        max_learning_rate=max(1e-5, args.learning_rate),
        schedule="fixed",
    )
    if args.stage == "ramp":
        # At 500 Hz, longer GAE credit assignment spans the ramp/landing transient.
        train["algorithm"].update(gamma=0.999, lam=0.995)
    provenance = dict(
        mode="actor_only_mujoco_finetune",
        checkpoint=str(args.checkpoint.resolve()),
        sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        model_sha256=model_hash,
        seed=args.seed,
        stage=args.stage,
        ramp_height_m=args.ramp_height if args.stage == "ramp" else None,
        reward_version=(
            "ramp_pitch_margin_v5"
            if args.stage == "ramp"
            else "mujoco_sensor_adaptation_v1"
        ),
        source_sha256={
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "sim2sim/environment.py",
                "sim2sim/train.py",
                "sim2sim/build_model.py",
            )
        },
        physics_dt=env.physics_dt,
        control_dt=env.dt,
    )
    log = args.log_root / datetime.now().strftime("%Y%m%d_%H%M%S")

    def metadata():
        return dict(
            simulator="mujoco",
            mujoco_provenance=provenance,
            legged_gym_env_config=config,
        )

    trainer = PPOTrainer(
        env, log, train, environment_config=config, checkpoint_metadata=metadata
    )
    weights = trainer.policy.state_dict()
    for name, value in source["model_state_dict"].items():
        if name.startswith("actor.") or name == "std":
            weights[name] = value
    trainer.policy.load_state_dict(weights)
    assert all(
        torch.equal(v, trainer.policy.state_dict()[k])
        for k, v in source["model_state_dict"].items()
        if k.startswith("actor.")
    )
    print(
        "Actor initialized exactly; fresh Critic and optimizer. Log:", log, flush=True
    )
    trainer.save(log / "model_0.pt", infos=provenance)
    try:
        trainer.learn(args.iterations)
    finally:
        env.close()


if __name__ == "__main__":
    main()
