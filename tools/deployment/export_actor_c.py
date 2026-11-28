"""Export an ELU actor as portable float32 C with no heap allocation.

Compile actor.c once and include actor.h from firmware. The caller supplies the
same observation history and external controller as Python. Host equivalence is
checked against PyTorch; STM32 timing/memory must be measured with its toolchain.
"""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if source["training_config"]["policy"]["activation"].lower() != "elu":
        raise ValueError("C exporter currently supports ELU actors only")
    weights = source["model_state_dict"]
    indices = sorted(
        int(k.split(".")[1])
        for k in weights
        if k.startswith("actor.") and k.endswith(".weight")
    )
    layers = []
    code = ['#include "actor.h"', "#include <math.h>", ""]
    sizes = []
    parameters = 0
    macs = 0
    for n, index in enumerate(indices):
        w, b = weights[f"actor.{index}.weight"], weights[f"actor.{index}.bias"]
        output_size, input_size = w.shape
        sizes.append(output_size)
        parameters += w.numel() + b.numel()
        macs += w.numel()
        linear = torch.nn.Linear(input_size, output_size)
        linear.load_state_dict(dict(weight=w, bias=b))
        layers.append(linear)
        if n < len(indices) - 1:
            layers.append(torch.nn.ELU())
        for name, tensor in ((f"w{n}", w), (f"b{n}", b)):
            values = [float(v).hex() + "f" for v in tensor.flatten()]
            lines = [", ".join(values[i : i + 8]) for i in range(0, len(values), 8)]
            code.append(
                f"static const float {name}[{len(values)}] = {{\n"
                + ",\n".join(lines)
                + "\n};\n"
            )
    inputs = weights[f"actor.{indices[0]}.weight"].shape[1]
    outputs = sizes[-1]
    width = max(sizes[:-1])
    header = f"""#ifndef WHEELLEG_ACTOR_H
#define WHEELLEG_ACTOR_H
#define WHEELLEG_ACTOR_INPUTS {inputs}
#define WHEELLEG_ACTOR_OUTPUTS {outputs}
#ifdef __cplusplus
extern "C" {{
#endif
void wheelleg_actor(const float input[{inputs}], float output[{outputs}]);
#ifdef __cplusplus
}}
#endif
#endif
"""
    code.append(
        f"void wheelleg_actor(const float input[{inputs}], float output[{outputs}]) {{\n  float scratch[2][{width}];"
    )
    previous = inputs
    for n, size in enumerate(sizes):
        src = "input" if n == 0 else f"scratch[{(n - 1) % 2}]"
        dest = "output" if n == len(sizes) - 1 else f"scratch[{n % 2}]"
        code.append(
            f"  for (int i = 0; i < {size}; ++i) {{\n    float v = b{n}[i];\n    for (int j = 0; j < {previous}; ++j) v += w{n}[i * {previous} + j] * {src}[j];"
        )
        if n < len(sizes) - 1:
            code.append("    if (v < 0.0f) v = expm1f(v);")
        code.append(f"    {dest}[i] = v;\n  }}")
        previous = size
    code.append("}\n")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "actor.h").write_text(header)
    (args.output / "actor.c").write_text("\n".join(code))
    actor = torch.nn.Sequential(*layers).eval()
    torch.jit.script(actor).save(str(args.output / "actor.pt"))
    rng = np.random.default_rng(42)
    x = rng.normal(size=(128, inputs)).astype(np.float32)
    # Include large wheel-speed-like inputs to exercise the ELU negative branch.
    x[64:] *= 20
    with tempfile.TemporaryDirectory() as temporary:
        library_path = Path(temporary) / "actor.so"
        subprocess.run(
            [
                "cc",
                "-std=c99",
                "-O2",
                "-ffp-contract=off",
                "-shared",
                "-fPIC",
                str((args.output / "actor.c").resolve()),
                "-o",
                str(library_path),
                "-lm",
            ],
            check=True,
        )
        library = ctypes.CDLL(str(library_path))
        pointer = ctypes.POINTER(ctypes.c_float)
        library.wheelleg_actor.argtypes = [pointer, pointer]
        library.wheelleg_actor.restype = None
        result = np.zeros((len(x), outputs), dtype=np.float32)
        for i in range(len(x)):
            library.wheelleg_actor(
                x[i].ctypes.data_as(pointer), result[i].ctypes.data_as(pointer)
            )
        with torch.inference_mode():
            reference = actor(torch.from_numpy(x)).numpy()
        error = float(np.max(np.abs(reference - result)))
        if not np.allclose(reference, result, atol=2e-4, rtol=2e-4):
            raise RuntimeError(f"C/PyTorch equivalence failed: max error {error}")
    manifest = dict(
        checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        input_size=inputs,
        output_size=outputs,
        parameters=parameters,
        weight_bytes_fp32=parameters * 4,
        macs_per_inference=macs,
        explicit_scratch_bytes=2 * width * 4,
        scratch_note="Excludes caller input/output, stack overhead and libm workspace; not a measured STM32 RAM total.",
        activation="ELU",
        host_c_max_abs_error=error,
        stm32_latency_measured=False,
        environment=source["legged_gym_env_config"],
        distillation=source.get("distillation"),
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                key: value
                for key, value in manifest.items()
                if key not in ("environment", "distillation")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
