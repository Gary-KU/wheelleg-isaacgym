#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import isaacgym
from legged_gym.envs import register_tasks
register_tasks()
from legged_gym.utils import task_registry

for name in task_registry.task_classes:
    cfg, train = task_registry.get_cfgs(name)
    print(
        "{}: obs={} actions={} terrain={} experiment={}".format(
            name,
            cfg.env.num_observations,
            cfg.env.num_actions,
            cfg.terrain.terrain,
            train.runner.experiment_name,
        )
    )
