"""共用检查点校验；只比较调用方提供的模型标识，不导入机器人代码。"""


def validate_model_fingerprint(state, expected):
    """防止同维度策略被误用于另一个机器人；expected=None 表示调用方不校验。"""
    if expected is not None and state.get("asset_fingerprint") != expected:
        raise ValueError(
            "Checkpoint belongs to a different/unidentified robot model. "
            "Train a policy for the active model; use model preview to inspect it."
        )
