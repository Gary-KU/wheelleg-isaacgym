"""Export deterministic actor; outer controller is required separately."""

import argparse
import hashlib
import json
import inspect
from pathlib import Path
import torch

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True, type=Path)
p.add_argument("--output", required=True, type=Path)
p.add_argument("--validation-inputs", type=Path, help="Recorded distillation dataset")
p.add_argument("--require-onnx", action="store_true")
a = p.parse_args()
load_options = (
    {"weights_only": False}
    if "weights_only" in inspect.signature(torch.load).parameters
    else {}
)
s = torch.load(a.checkpoint, map_location="cpu", **load_options)
if s["training_config"]["policy"]["activation"].lower() != "elu":
    raise ValueError("Exporter currently supports ELU actors only")
w = s["model_state_dict"]
layers = []
indices = sorted(
    int(k.split(".")[1]) for k in w if k.startswith("actor.") and k.endswith("weight")
)
for i in indices:
    layers.append(
        torch.nn.Linear(
            w["actor.%s.weight" % i].shape[1], w["actor.%s.weight" % i].shape[0]
        )
    )
    if i != indices[-1]:
        layers.append(torch.nn.ELU())
actor = torch.nn.Sequential(*layers)
actor.load_state_dict({k[6:]: v for k, v in w.items() if k.startswith("actor.")})
actor.eval()
a.output.mkdir(parents=True, exist_ok=True)
torch.jit.script(actor).save(str(a.output / "actor.pt"))
x = torch.randn(128, w["actor.0.weight"].shape[1])
if a.validation_inputs:
    recorded = torch.load(a.validation_inputs, map_location="cpu", **load_options)[
        "val_x"
    ]
    x = recorded[torch.linspace(0, len(recorded) - 1, 128).long()].clone()
assert torch.equal(actor(x), torch.jit.load(str(a.output / "actor.pt"))(x))
meta = dict(
    checkpoint=str(a.checkpoint.resolve()),
    sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
    input_size=w["actor.0.weight"].shape[1],
    output_size=w["actor.%s.weight" % indices[-1]].shape[0],
    torchscript_verified=True,
    environment=s["legged_gym_env_config"],
)
meta.update(
    input_shape=[1, meta["input_size"]],
    output_shape=[1, meta["output_size"]],
    dtype="float32",
    activation="ELU",
    parameters=sum(v.numel() for v in actor.parameters()),
    weight_bytes_fp32=sum(v.numel() * v.element_size() for v in actor.parameters()),
    macs_per_inference=sum(
        layer.weight.numel() for layer in actor if isinstance(layer, torch.nn.Linear)
    ),
    stm32cube_ai_analyzed=False,
    board_latency_measured=False,
)
try:
    options = (
        {"dynamo": False}
        if "dynamo" in inspect.signature(torch.onnx.export).parameters
        else {}
    )
    torch.onnx.export(
        actor,
        torch.zeros(1, w["actor.0.weight"].shape[1]),
        str(a.output / "actor.onnx"),
        input_names=["observations"],
        output_names=["actions"],
        opset_version=11,
        **options
    )
    import onnx, onnxruntime as ort, numpy as np

    model = onnx.load(str(a.output / "actor.onnx"))
    onnx.checker.check_model(model)
    session = ort.InferenceSession(
        str(a.output / "actor.onnx"), providers=["CPUExecutionProvider"]
    )
    prediction = np.concatenate(
        [session.run(None, {"observations": row[None].numpy()})[0] for row in x]
    )
    with torch.inference_mode():
        reference = actor(x).numpy()
    error = float(np.max(np.abs(prediction - reference)))
    assert np.allclose(prediction, reference, atol=1e-5, rtol=1e-5), error
    meta.update(
        onnx_verified=True,
        onnx_max_abs_error=error,
        onnx_opset=11,
        onnx_operators=sorted({node.op_type for node in model.graph.node}),
    )
except Exception as e:
    meta["onnx_error"] = str(e)
    if a.require_onnx:
        raise
(a.output / "manifest.json").write_text(json.dumps(meta, indent=2))
print("Exported", a.output)
(a.output / "metadata.json").write_text(json.dumps(meta, indent=2))
