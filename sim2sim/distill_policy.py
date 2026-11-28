"""Distill an actor into an embedded-size MLP using sensor-only rollouts.

Training and validation data use different noise seeds. Normalization is folded
into Linear layers before saving, preserving the 132-input/6-output contract.
Offline imitation errors do not establish closed-loop acceptance: run the same
MuJoCo ramp and flat regression checks on the resulting checkpoint separately.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim2sim.environment import MujocoWheelLeg
from legged_gym.learning.policy import make_policy


def wheel_modes(actions):
    """Separate forward drive and steering, so small yaw signals get a scale."""
    result = actions.clone()
    result[:, 4] = (actions[:, 5] - actions[:, 4]) * 0.5
    result[:, 5] = (actions[:, 5] + actions[:, 4]) * 0.5
    return result


def motor_actions(modes):
    result = modes.clone()
    result[:, 4] = modes[:, 5] - modes[:, 4]
    result[:, 5] = modes[:, 5] + modes[:, 4]
    return result


def collect(source, seed, driver=None):
    teacher = make_policy(132, 6, "cpu", source["training_config"]["policy"])
    teacher.load_state_dict(source["model_state_dict"])
    teacher.eval()
    observations, labels = [], []
    # Include stationary support and both driving directions, plus the validated
    # forward ramp task. No world state is appended to observations.
    for terrain, speed in (("flat", 0), ("flat", 1), ("flat", -1), ("ramp", 1)):
        env = MujocoWheelLeg(
            source["legged_gym_env_config"],
            4,
            seed,
            episode_seconds=7,
            terrain=terrain,
            ramp_height=0.01,
        )
        with torch.inference_mode():
            for step in range(3000):
                if step >= 1000:
                    env.targets[:, 0] = speed
                x = env.obs.clone()
                y = teacher.act_inference(x)
                if step % 2 == 0:
                    observations.append(x)
                    labels.append(y.clone())
                env.step(driver(x) if driver is not None else y)
        env.close()
        print(f"Collected seed={seed} terrain={terrain} speed={speed}", flush=True)
    return torch.cat(observations), torch.cat(labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--hidden", type=int, nargs="+", default=(128, 64))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--direct-actions",
        action="store_true",
        help="Fit original motor actions instead of wheel modes",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dagger-checkpoint", type=Path)
    parser.add_argument(
        "--extra-dataset",
        type=Path,
        help="Mix original teacher trajectories with DAgger data",
    )
    args = parser.parse_args()
    if min(*args.hidden, args.epochs) < 1:
        parser.error("hidden sizes and epochs must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    teacher_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    driver = None
    if args.dagger_checkpoint:
        state = torch.load(
            args.dagger_checkpoint, map_location="cpu", weights_only=False
        )
        driver = make_policy(132, 6, "cpu", state["training_config"]["policy"])
        driver.load_state_dict(state["model_state_dict"])
        driver = driver.actor.eval()
    driver_hash = (
        hashlib.sha256(args.dagger_checkpoint.read_bytes()).hexdigest()
        if driver is not None
        else None
    )
    if args.dataset.exists():
        dataset = torch.load(args.dataset, weights_only=False)
        if (
            dataset["teacher_sha256"] != teacher_hash
            or dataset.get("driver_sha256") != driver_hash
        ):
            raise RuntimeError(
                "Cached dataset provenance differs from requested teacher/driver"
            )
    else:
        train_x, train_y = collect(source, 42, driver)
        val_x, val_y = collect(source, 123, driver)
        dataset = dict(
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            teacher_sha256=teacher_hash,
            driver_sha256=driver_hash,
        )
        args.dataset.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, args.dataset)
    x, y = dataset["train_x"], dataset["train_y"]
    vx, vy = dataset["val_x"], dataset["val_y"]
    if args.extra_dataset:
        extra = torch.load(args.extra_dataset, weights_only=False)
        if extra["teacher_sha256"] != teacher_hash:
            raise RuntimeError("Extra dataset uses a different teacher")
        x, y = torch.cat((x, extra["train_x"])), torch.cat((y, extra["train_y"]))
        vx, vy = torch.cat((vx, extra["val_x"])), torch.cat((vy, extra["val_y"]))
    encoded_y, encoded_vy = (
        (y, vy) if args.direct_actions else (wheel_modes(y), wheel_modes(vy))
    )
    mx, sx = x.mean(0), x.std(0).clamp(min=0.01)
    my, sy = encoded_y.mean(0), encoded_y.std(0).clamp(min=0.01)
    if not args.direct_actions:
        sy[5] = encoded_y[:, 5].std().clamp(min=0.0001)
    nx, ny = (x - mx) / sx, (encoded_y - my) / sy
    nvx, nvy = (vx - mx) / sx, (encoded_vy - my) / sy
    sizes = [132, *args.hidden, 6]
    layers = []
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        layers.append(torch.nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*layers)
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=1e-5
    )
    best, best_loss, best_epoch = None, float("inf"), None
    for epoch in range(args.epochs):
        actor.train()
        for batch in torch.randperm(len(nx)).split(512):
            loss = (actor(nx[batch]) - ny[batch]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        actor.eval()
        with torch.no_grad():
            val_loss = (actor(nvx) - nvy).square().mean().item()
        if val_loss < best_loss:
            best_loss, best = val_loss, deepcopy(actor.state_dict())
            best_epoch = epoch + 1
        if (epoch + 1) % 20 == 0:
            print(f"epoch={epoch + 1} normalized_val_mse={val_loss:.7f}", flush=True)
        scheduler.step()
    actor.load_state_dict(best)
    with torch.no_grad():
        reference = actor(nvx) * sy + my
        if not args.direct_actions:
            reference = motor_actions(reference)
        # Fold input/output normalization so firmware needs only the MLP itself.
        actor[0].weight.div_(sx[None, :])
        actor[0].bias.sub_(actor[0].weight @ mx)
        actor[-1].weight.mul_(sy[:, None])
        actor[-1].bias.mul_(sy).add_(my)
        # Fold the inverse wheel-mode transform into the last Linear as well.
        if not args.direct_actions:
            drive_w, yaw_w = actor[-1].weight[4:].clone()
            drive_b, yaw_b = actor[-1].bias[4:].clone()
            actor[-1].weight[4], actor[-1].weight[5] = yaw_w - drive_w, yaw_w + drive_w
            actor[-1].bias[4], actor[-1].bias[5] = yaw_b - drive_b, yaw_b + drive_b
        prediction = actor(vx)
        fold_error = (prediction - reference).abs().max().item()
        assert fold_error < 1e-4, fold_error
        error = prediction - vy
    config = deepcopy(source["training_config"])
    config["policy"]["actor_hidden_dims"] = list(args.hidden)
    policy = make_policy(132, 6, "cpu", config["policy"])
    # Copy unchanged critic and exploration std only when dimensions agree.
    weights = policy.state_dict()
    for key, value in source["model_state_dict"].items():
        if (
            not key.startswith("actor.")
            and key in weights
            and weights[key].shape == value.shape
        ):
            weights[key] = value
    weights.update({"actor." + key: value for key, value in actor.state_dict().items()})
    policy.load_state_dict(weights)
    parameters = sum(p.numel() for p in actor.parameters())
    report = dict(
        teacher=str(args.checkpoint.resolve()),
        teacher_sha256=teacher_hash,
        driver_sha256=driver_hash,
        dataset=str(args.dataset.resolve()),
        dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        extra_dataset=str(args.extra_dataset.resolve()) if args.extra_dataset else None,
        extra_dataset_sha256=(
            hashlib.sha256(args.extra_dataset.read_bytes()).hexdigest()
            if args.extra_dataset
            else None
        ),
        training_noise_seed=42,
        validation_noise_seed=123,
        best_epoch=best_epoch,
        hidden=list(args.hidden),
        parameters=parameters,
        float32_weight_bytes=parameters * 4,
        train_samples=len(x),
        validation_samples=len(vx),
        normalized_validation_mse=best_loss,
        wheel_modes_normalized=not args.direct_actions,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        learning_rate_schedule="cosine_to_1e-5",
        action_rmse=error.square().mean(0).sqrt().tolist(),
        action_max_error=error.abs().max(0).values.tolist(),
        normalization_fold_max_error=fold_error,
        closed_loop_passed=False,
    )
    checkpoint = dict(
        model_state_dict=policy.state_dict(),
        training_config=config,
        legged_gym_env_config=deepcopy(source["legged_gym_env_config"]),
        environment_config=deepcopy(source["legged_gym_env_config"]),
        simulator="mujoco",
        iter=0,
        distillation=report,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output / "model_0.pt")
    torch.jit.script(actor.eval()).save(str(args.output / "actor.pt"))
    (args.output / "distillation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
