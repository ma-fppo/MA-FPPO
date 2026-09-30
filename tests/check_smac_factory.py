"""Check portable SMAC factory setup without launching StarCraft II."""
import importlib.util
import os
from pathlib import Path
import sys
import types

path = Path(__file__).resolve().parents[1] / "main_experiments/runtime/smac/project/mac_flow_pytorch_smacv2/macflow_torch/smacv1_env.py"
class FakeEnv:
    def __init__(self, **kwargs): self.kwargs = kwargs
fake = types.ModuleType("smac.env")
fake.StarCraft2Env = FakeEnv
sys.modules["smac.env"] = fake
spec = importlib.util.spec_from_file_location("factory", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
os.environ["SC2PATH"] = "/explicit/sc2"
for task in ("3m", "2s3z", "5m_vs_6m", "8m"):
    env = module.make_smacv1_env(task, seed=123)
    assert env.kwargs["seed"] == 123 and env.kwargs["map_name"] == task
    assert env._ma_fppo_environment_kind == "smac_v1"
assert os.environ["SC2PATH"] == "/explicit/sc2"
print("SMAC factory import, seed and explicit path passed.")
