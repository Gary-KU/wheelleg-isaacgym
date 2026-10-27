"""Optional editable install; existing ./run.sh entrypoints need no pip changes."""

from setuptools import find_packages, setup

setup(
    name="wheelleg-isaacgym",
    version="0.2.0",
    python_requires=">=3.8,<3.9",
    packages=find_packages(include=["legged_gym*", "wheelleg_gym*"]),
    description="Five-bar wheelleg tasks for Isaac Gym / legged_gym",
)
