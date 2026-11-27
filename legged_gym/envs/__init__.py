"""统一注册入口，导入本模块不会启动仿真。"""
import importlib
import os


def register_tasks(registry=None):
    """加载内置任务与研究员提供的模块。模块须公开 register(registry)。"""
    from legged_gym.utils.task_registry import task_registry
    from .wheelleg.register import register

    target = registry if registry is not None else task_registry
    register(target)
    for module in os.environ.get('LEGGED_TASK_MODULES', '').split(','):
        if module.strip():
            importlib.import_module(module.strip()).register(target)
