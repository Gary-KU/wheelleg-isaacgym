"""Export deterministic actor; outer controller is required separately."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True,type=Path)
p.add_argument('--output',required=True,type=Path)
a=p.parse_args()
s=torch.load(a.checkpoint,map_location='cpu')
w=s['model_state_dict']; layers=[]
indices=sorted(int(k.split(".")[1]) for k in w if k.startswith("actor.") and k.endswith("weight"))
for i in indices:
 layers.append(torch.nn.Linear(w['actor.%s.weight'%i].shape[1],w['actor.%s.weight'%i].shape[0]))
 if i!=indices[-1]: layers.append(torch.nn.ELU())
actor=torch.nn.Sequential(*layers)
actor.load_state_dict({k[6:]:v for k,v in w.items() if k.startswith('actor.')})
actor.eval();a.output.mkdir(parents=True,exist_ok=True)
torch.jit.script(actor).save(str(a.output/'actor.pt'))
x=torch.randn(128,w["actor.0.weight"].shape[1])
assert torch.equal(actor(x),torch.jit.load(str(a.output/'actor.pt'))(x))
meta=dict(checkpoint=str(a.checkpoint.resolve()),sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),input_size=w["actor.0.weight"].shape[1],output_size=w["actor.%s.weight"%indices[-1]].shape[0],torchscript_verified=True,environment=s['legged_gym_env_config'])
try:
 torch.onnx.export(actor,torch.zeros(1,w["actor.0.weight"].shape[1]),str(a.output/'actor.onnx'),input_names=['observations'],output_names=['actions'],opset_version=11)
except Exception as e: meta['onnx_error']=str(e)
(a.output/'manifest.json').write_text(json.dumps(meta,indent=2))
print('Exported',a.output)
(a.output/'metadata.json').write_text(json.dumps(meta,indent=2))
