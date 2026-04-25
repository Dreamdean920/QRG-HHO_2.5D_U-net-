from __future__ import annotations
import json, math, random, hashlib
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import yaml

def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def space_keys(space: Dict[str, Any]) -> List[str]:
    return list(space.keys())

def sample_from_space(space: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    out = {}
    for k, spec in space.items():
        if spec["type"] == "float":
            low, high = float(spec["low"]), float(spec["high"])
            if spec.get("log", False):
                val = math.exp(rng.uniform(math.log(low), math.log(high)))
            else:
                val = rng.uniform(low, high)
            out[k] = float(val)
        elif spec["type"] == "int":
            out[k] = int(rng.randint(int(spec["low"]), int(spec["high"])))
        elif spec["type"] == "categorical":
            out[k] = rng.choice(list(spec["choices"]))
        else:
            raise ValueError(f"Unsupported space type: {spec['type']}")
    return out

def clip_to_space(params: Dict[str, Any], space: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, spec in space.items():
        v = params[k]
        if spec["type"] == "float":
            v = float(v)
            v = max(float(spec["low"]), min(float(spec["high"]), v))
            out[k] = v
        elif spec["type"] == "int":
            v = int(round(v))
            v = max(int(spec["low"]), min(int(spec["high"]), v))
            out[k] = v
        elif spec["type"] == "categorical":
            choices = list(spec["choices"])
            out[k] = v if v in choices else choices[0]
        else:
            raise ValueError(f"Unsupported space type: {spec['type']}")
    return out

def params_to_run_name(method: str, params: Dict[str, Any], suffix: str = "") -> str:
    payload = json.dumps(params, sort_keys=True, ensure_ascii=False)
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]
    return f"{method}_{digest}{suffix}"

def save_json(path: str | Path, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
