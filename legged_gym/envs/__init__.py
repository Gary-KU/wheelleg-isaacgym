"""Task registration is explicit; importing helpers never loads a robot or Gym."""


def register_tasks(registry=None):
    """Register built-in tasks. Add another robot's register() call here."""
    from legged_gym.utils.task_registry import task_registry
    from .wheelleg.register import register

    register(registry if registry is not None else task_registry)
