from __future__ import annotations
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional
from exp_utils import ensure_dir


class ObjectiveAdapter:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.output_root = ensure_dir(cfg["output_root"])
        self.maximize_metric = cfg.get("maximize_metric", "val_dice")

    def _build_command(
        self,
        mode: str,
        params: Dict[str, Any],
        seed: int,
        epochs: int,
        run_name: str,
        output_json: Path,
    ) -> str:
        template = self.cfg["train_command"][mode]
        kwargs = dict(params)
        kwargs.update(
            seed=seed,
            epochs=epochs,
            run_name=run_name,
            output_json=str(output_json),
        )
        return template.format(**kwargs)

    def _try_resume_from_json(
        self,
        output_json: Path,
        params: Dict[str, Any],
        seed: int,
        run_name: str,
        mode: str,
    ) -> Optional[Dict[str, Any]]:
        if not output_json.exists():
            return None

        try:
            payload = json.loads(output_json.read_text(encoding="utf-8"))
        except Exception:
            return None

        if payload.get("status", "") != "OK":
            return None

        score_key = self.maximize_metric
        score = payload.get(score_key)
        if score is None:
            return None

        resumed = {
            "status": "OK",
            "command": "[RESUMED] existing output_json",
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "elapsed_sec": 0.0,
            "params": params,
            "seed": seed,
            "run_name": run_name,
            "mode": mode,
            "output_json": str(output_json),
        }
        resumed.update(payload)
        resumed["score"] = float(score)
        resumed["fitness"] = -float(score)
        resumed["resumed"] = True
        return resumed

    def evaluate(
        self,
        params: Dict[str, Any],
        *,
        seed: int,
        run_name: str,
        mode: str = "short_eval",
        timeout_sec: Optional[int] = None,
    ) -> Dict[str, Any]:
        trial_dir = ensure_dir(self.output_root / "trial_json")
        output_json = trial_dir / f"{run_name}.json"

        # ========== 断点续跑：已有成功结果则直接跳过 ==========
        resumed = self._try_resume_from_json(
            output_json=output_json,
            params=params,
            seed=seed,
            run_name=run_name,
            mode=mode,
        )
        if resumed is not None:
            print("\n" + "=" * 80)
            print(f"[ObjectiveAdapter] SKIP finished run_name={run_name}")
            print(f"[ObjectiveAdapter] reused json={output_json}")
            print(f"[ObjectiveAdapter] score={resumed['score']:.6f}")
            print("=" * 80)
            return resumed

        epochs = int(
            self.cfg["search"]["short_eval_epochs"]
            if mode == "short_eval"
            else self.cfg["search"]["confirm_epochs"]
        )

        cmd = self._build_command(
            mode=mode,
            params=params,
            seed=seed,
            epochs=epochs,
            run_name=run_name,
            output_json=output_json,
        )

        print("\n" + "=" * 80)
        print(f"[ObjectiveAdapter] START run_name={run_name}")
        print(f"[ObjectiveAdapter] mode={mode}, seed={seed}, epochs={epochs}")
        print(f"[ObjectiveAdapter] params={params}")
        print(f"[ObjectiveAdapter] cmd={cmd}")
        print("=" * 80)

        start = time.time()

        stdout_lines = []
        stderr_lines = []
        returncode = -1

        try:
            proc = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            while True:
                line = proc.stdout.readline() if proc.stdout is not None else ""
                if line:
                    line = line.rstrip("\n")
                    stdout_lines.append(line)
                    print(f"[{run_name}] {line}", flush=True)

                if proc.poll() is not None:
                    if proc.stdout is not None:
                        rest = proc.stdout.read()
                        if rest:
                            for extra_line in rest.splitlines():
                                stdout_lines.append(extra_line)
                                print(f"[{run_name}] {extra_line}", flush=True)
                    returncode = proc.returncode
                    break

                if timeout_sec is not None and (time.time() - start) > timeout_sec:
                    proc.kill()
                    raise TimeoutError(f"subprocess timeout after {timeout_sec} sec")

        except Exception as e:
            elapsed = time.time() - start
            return {
                "status": "ERROR",
                "command": cmd,
                "returncode": returncode,
                "stdout": "\n".join(stdout_lines)[-4000:],
                "stderr": "\n".join(stderr_lines)[-4000:],
                "elapsed_sec": elapsed,
                "params": params,
                "seed": seed,
                "run_name": run_name,
                "mode": mode,
                "output_json": str(output_json),
                "error_message": f"subprocess_exception: {e}",
            }

        elapsed = time.time() - start

        result: Dict[str, Any] = {
            "status": "ERROR",
            "command": cmd,
            "returncode": returncode,
            "stdout": "\n".join(stdout_lines)[-4000:],
            "stderr": "\n".join(stderr_lines)[-4000:],
            "elapsed_sec": elapsed,
            "params": params,
            "seed": seed,
            "run_name": run_name,
            "mode": mode,
            "output_json": str(output_json),
        }

        if returncode != 0:
            result["error_message"] = "subprocess_failed"
            print(f"[ObjectiveAdapter] FAIL run_name={run_name}, returncode={returncode}")
            return result

        if not output_json.exists():
            result["error_message"] = "missing_output_json"
            print(f"[ObjectiveAdapter] FAIL run_name={run_name}, missing output_json={output_json}")
            return result

        try:
            payload = json.loads(output_json.read_text(encoding="utf-8"))
        except Exception as e:
            result["error_message"] = f"invalid_output_json: {e}"
            print(f"[ObjectiveAdapter] FAIL run_name={run_name}, invalid json: {e}")
            return result

        result.update(payload)
        result["status"] = payload.get("status", "OK")
        result["elapsed_sec"] = elapsed

        score_key = self.maximize_metric
        score = payload.get(score_key)
        if score is None:
            result["error_message"] = f"missing_score_key:{score_key}"
            result["status"] = "ERROR"
            print(f"[ObjectiveAdapter] FAIL run_name={run_name}, missing score key={score_key}")
            return result

        result["score"] = float(score)
        result["fitness"] = -float(score)
        result["resumed"] = False

        print(f"[ObjectiveAdapter] DONE run_name={run_name}, score={result['score']:.6f}, elapsed={elapsed:.2f}s")
        return result