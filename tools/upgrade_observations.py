#!/usr/bin/env python3
"""Preserve a legacy checkpoint while adding actual-leg-length observations."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wheelleg_gym.observation_upgrade import save_upgraded_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    try:
        metadata = save_upgraded_checkpoint(args.source, args.target)
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(dict(target_checkpoint=str(args.target.resolve()), **metadata), indent=2))


if __name__ == "__main__":
    main()
