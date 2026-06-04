#!/usr/bin/env python3
"""Rerun Gal_split_nolang flat evaluation with the correct prompt in 3 shards."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
REMOTE = SCRIPT_DIR / "remote.sh"
MODEL_ID = "MasterProject2026/Gal_split_nolang"
MODEL_NAME = "Gal_split_nolang"
PROMPT = "Place the orange into plate"
TOTAL_EPISODES = 100
N_SHARDS = 3
CONDA_ENV = os.environ.get("CONDA_ENV", "leisaac_envhub")
MAX_RETRIES = int(os.environ.get("GAL_SPLIT_RERUN_MAX_RETRIES", "2"))
STALL_SECONDS = int(os.environ.get("GAL_SPLIT_RERUN_STALL_SECONDS", str(75 * 60)))
STARTUP_GRACE_SECONDS = int(os.environ.get("GAL_SPLIT_RERUN_STARTUP_GRACE_SECONDS", str(20 * 60)))
BASE_HTTP_PORT = int(os.environ.get("GAL_SPLIT_RERUN_BASE_HTTP_PORT", "8011"))
BASE_LIVESTREAM_PORT = int(os.environ.get("GAL_SPLIT_RERUN_BASE_LIVESTREAM_PORT", "49100"))


@dataclass(frozen=True)
class Shard:
    index: int
    n_runs: int

    @property
    def name(self) -> str:
        return f"shard{self.index}"


def now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def iso_now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def run_root(run_id: str) -> Path:
    return SCRIPT_DIR / "logs" / "gal_split_nolang_correct_prompt" / run_id


def result_dir() -> Path:
    return RESULTS_DIR / MODEL_NAME


def final_checkpoint_path() -> Path:
    return result_dir() / "flat_checkpoint.json"


def final_summary_path() -> Path:
    return result_dir() / "flat_latest.txt"


def shard_checkpoint(run_id: str, shard: Shard) -> Path:
    return run_root(run_id) / shard.name / "flat_checkpoint.json"


def shard_summary(run_id: str, shard: Shard) -> Path:
    return run_root(run_id) / shard.name / "flat_latest.txt"


def shard_log(run_id: str, shard: Shard, attempt: int) -> Path:
    return run_root(run_id) / shard.name / f"attempt{attempt}.log"


def shard_rc(run_id: str, shard: Shard, attempt: int) -> Path:
    return run_root(run_id) / shard.name / f"attempt{attempt}.rc"


def shard_http_port(shard: Shard) -> int:
    return BASE_HTTP_PORT + shard.index


def shard_livestream_port(shard: Shard) -> int:
    return BASE_LIVESTREAM_PORT + shard.index


def shards() -> list[Shard]:
    base, extra = divmod(TOTAL_EPISODES, N_SHARDS)
    return [Shard(i, base + (1 if i < extra else 0)) for i in range(N_SHARDS)]


def checkpoint_completed(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text())
    except Exception:
        return 0
    completed = data.get("completed_episodes")
    if isinstance(completed, int):
        return completed
    episodes = data.get("episodes", [])
    return len(episodes) if isinstance(episodes, list) else 0


def latest_mtime(paths: list[Path]) -> float:
    mtimes = [path.stat().st_mtime for path in paths if path.exists()]
    return max(mtimes) if mtimes else 0.0


def run_checked(cmd: list[str], *, timeout=120) -> str:
    result = subprocess.run(
        cmd,
        cwd=SCRIPT_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {shlex.join(cmd)}\n{result.stdout}")
    return result.stdout


def conda_python_cmd(*args: str) -> list[str]:
    return ["conda", "run", "--no-capture-output", "-n", CONDA_ENV, "python", *args]


def static_preflight() -> None:
    print("Static preflight")
    run_checked(
        conda_python_cmd("-m", "py_compile", "inference_flat_prompt.py", "rerun_gal_split_nolang_correct_prompt.py"),
        timeout=120,
    )
    run_checked(
        conda_python_cmd(
            "-c",
            "import torch, lerobot; "
            "assert torch.cuda.is_available(), 'CUDA unavailable'; "
            "print(torch.cuda.get_device_name(0))",
        ),
        timeout=120,
    )
    run_checked(
        conda_python_cmd(
            "-c",
            "from huggingface_hub import HfApi; "
            f"info=HfApi().model_info({MODEL_ID!r}); "
            "files=[s.rfilename for s in (info.siblings or [])]; "
            "assert 'config.json' in files, files; "
            "print('model ok')",
        ),
        timeout=120,
    )
    run_checked(["tmux", "-V"], timeout=30)


def archive_existing_results(run_id: str) -> None:
    src = result_dir()
    if not src.exists():
        return
    archive_root = RESULTS_DIR / "_archive" / run_id
    archive_root.mkdir(parents=True, exist_ok=True)
    dest = archive_root / f"{MODEL_NAME}_wrong_prompt"
    suffix = 1
    while dest.exists():
        suffix += 1
        dest = archive_root / f"{MODEL_NAME}_wrong_prompt_{suffix}"
    print(f"Archiving wrong-prompt results: {src} -> {dest}")
    shutil.move(str(src), str(dest))


def tmux_run(cmd: list[str], *, check=False) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["tmux", *cmd],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"tmux {' '.join(cmd)} failed:\n{result.stdout}")
    return result


def tmux_alive(target: str | None) -> bool:
    if not target:
        return False
    return tmux_run(["display-message", "-p", "-t", target, "#{window_id}"]).returncode == 0


def tmux_interrupt(target: str | None) -> None:
    if target and tmux_alive(target):
        tmux_run(["send-keys", "-t", target, "C-c"])


def tmux_kill(target: str | None) -> None:
    if target and tmux_alive(target):
        tmux_run(["kill-window", "-t", target])


def shard_done(run_id: str, shard: Shard) -> bool:
    return checkpoint_completed(shard_checkpoint(run_id, shard)) >= shard.n_runs and shard_summary(run_id, shard).exists()


def write_state(run_id: str, states: dict[str, dict]) -> None:
    path = run_root(run_id) / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "updated_at": iso_now(),
        "model_id": MODEL_ID,
        "prompt": PROMPT,
        "target_episodes": TOTAL_EPISODES,
        "states": states,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def start_shard(run_id: str, session: str, shard: Shard, state: dict) -> None:
    state["attempts"] += 1
    attempt = state["attempts"]
    log_path = shard_log(run_id, shard, attempt)
    rc_path = shard_rc(run_id, shard, attempt)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for path in (log_path, rc_path):
        if path.exists():
            path.unlink()

    env = {
        "CONDA_ENV": CONDA_ENV,
        "MODEL_ID": MODEL_ID,
        "INSTRUCTION": PROMPT,
        "N_INFERENCE_RUNS": str(shard.n_runs),
        "MAX_STEPS": "5000",
        "EVAL_RESUME": "1" if shard_checkpoint(run_id, shard).exists() else "0",
        "EVAL_CHECKPOINT_PATH": str(shard_checkpoint(run_id, shard)),
        "EVAL_SUMMARY_PATH": str(shard_summary(run_id, shard)),
        "SAVE_CAMERA_SNAPSHOTS": "0",
        "LIVESTREAM": "2",
        "ENABLE_LIVESTREAM": "1",
        "ENABLE_CAMERAS": "1",
        "REMOTE_LOG_FILE": str(log_path),
    }
    exports = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env.items()))
    kit_args = [
        f"--/exts/omni.services.transport.server.http/port={shard_http_port(shard)}",
        f"--/exts/omni.services.transport.server.http/https/port={8433 + shard.index}",
        f"--/app/livestream/port={shard_livestream_port(shard)}",
    ]
    remote_cmd = " ".join(shlex.quote(arg) for arg in ["./remote.sh", "inference_flat_prompt.py", *kit_args])
    inner = (
        "set -o pipefail; "
        f"cd {shlex.quote(str(SCRIPT_DIR))}; "
        f"{exports} {remote_cmd}; "
        "rc=$?; "
        f"echo $rc > {shlex.quote(str(rc_path))}; "
        "exit $rc"
    )
    result = tmux_run(
        [
            "new-window",
            "-d",
            "-P",
            "-F",
            "#{window_id}",
            "-t",
            session,
            "-n",
            f"{shard.name}_{attempt}",
            "bash",
            "-lc",
            inner,
        ],
        check=True,
    )
    window_id = result.stdout.strip().splitlines()[-1]
    now = time.time()
    state.update(
        {
            "status": "running",
            "window_id": window_id,
            "started_at": iso_now(),
            "started_ts": now,
            "last_progress_ts": now,
            "last_seen_mtime": 0.0,
            "completed": checkpoint_completed(shard_checkpoint(run_id, shard)),
            "log": str(log_path),
            "rc": str(rc_path),
            "http_port": shard_http_port(shard),
            "livestream_port": shard_livestream_port(shard),
            "message": f"started attempt {attempt}",
        }
    )
    print(
        f"Started {shard.name} attempt {attempt} in {window_id} "
        f"(http={shard_http_port(shard)}, livestream={shard_livestream_port(shard)})"
    )


def mark_done_or_retry(run_id: str, session: str, shard: Shard, state: dict, reason: str) -> None:
    completed = checkpoint_completed(shard_checkpoint(run_id, shard))
    state["completed"] = completed
    if completed >= shard.n_runs:
        state.update({"status": "done", "finished_at": iso_now(), "message": reason})
        print(f"{shard.name} done: {completed}/{shard.n_runs}")
        return
    if state["attempts"] <= MAX_RETRIES:
        print(f"Retrying {shard.name}: {reason}; completed={completed}/{shard.n_runs}")
        state.update({"status": "pending", "message": reason, "window_id": None})
        return
    state.update({"status": "failed", "finished_at": iso_now(), "message": reason})
    print(f"{shard.name} failed: {reason}; completed={completed}/{shard.n_runs}")


def monitor_shard(run_id: str, session: str, shard: Shard, state: dict) -> None:
    checkpoint_path = shard_checkpoint(run_id, shard)
    summary_path = shard_summary(run_id, shard)
    log_path = Path(state["log"])
    rc_path = Path(state["rc"])
    progress_mtime = latest_mtime([checkpoint_path, summary_path, log_path])
    if progress_mtime > float(state.get("last_seen_mtime", 0.0)):
        state["last_seen_mtime"] = progress_mtime
        state["last_progress_ts"] = time.time()

    state["completed"] = checkpoint_completed(checkpoint_path)
    alive = tmux_alive(state.get("window_id"))
    if alive and shard_done(run_id, shard):
        tmux_interrupt(state.get("window_id"))
        time.sleep(15)
        tmux_kill(state.get("window_id"))
        state.update({"status": "done", "finished_at": iso_now(), "message": "shard target reached"})
        print(f"{shard.name} reached target")
        return

    if not alive:
        rc = rc_path.read_text().strip() if rc_path.exists() else "missing"
        mark_done_or_retry(run_id, session, shard, state, f"process exited rc={rc}")
        return

    now = time.time()
    if now - float(state.get("started_ts", now)) < STARTUP_GRACE_SECONDS:
        return
    if now - float(state.get("last_progress_ts", now)) <= STALL_SECONDS:
        return

    tmux_interrupt(state.get("window_id"))
    time.sleep(60)
    tmux_kill(state.get("window_id"))
    mark_done_or_retry(run_id, session, shard, state, "stalled without progress")


def summary_text(records: list[dict]) -> str:
    n_eval = len(records)
    oranges = [record["oranges_in_plate"] for record in records]
    successes = sum(1 for count in oranges if count == 3)
    success_steps = [record["step_count"] for record in records if record["oranges_in_plate"] == 3]
    pct = lambda count: (count / n_eval * 100) if n_eval else 0
    avg_oranges = sum(oranges) / n_eval if n_eval else 0
    mean_success_steps = sum(success_steps) / len(success_steps) if success_steps else float("nan")
    header = "FLAT-PROMPT EVALUATION COMPLETE" if n_eval == TOTAL_EPISODES else (
        f"FLAT-PROMPT EVALUATION SUMMARY (stopped after {n_eval}/{TOTAL_EPISODES} runs)"
    )
    return (
        f"\n========================================\n"
        f"{header}\n"
        f"Model ID:             {MODEL_ID}\n"
        f"Prompt:               {PROMPT}\n"
        f"Success Rate:         {successes}/{n_eval} ({pct(successes):.2f}%)\n"
        f"Avg oranges in plate: {avg_oranges:.2f}/3\n"
        f"Mean steps (success): {mean_success_steps:.1f}\n"
        f"3/3 oranges:          {oranges.count(3)}/{n_eval} ({pct(oranges.count(3)):.1f}%)\n"
        f"2/3 oranges:          {oranges.count(2)}/{n_eval} ({pct(oranges.count(2)):.1f}%)\n"
        f"1/3 oranges:          {oranges.count(1)}/{n_eval} ({pct(oranges.count(1)):.1f}%)\n"
        f"0/3 oranges:          {oranges.count(0)}/{n_eval} ({pct(oranges.count(0)):.1f}%)\n"
        f"Per-episode oranges:  {oranges}\n"
        f"========================================\n"
    )


def merge_shards(run_id: str) -> None:
    records: list[dict] = []
    next_episode = 0
    for shard in shards():
        path = shard_checkpoint(run_id, shard)
        data = json.loads(path.read_text())
        shard_records = sorted(data.get("episodes", []), key=lambda record: record.get("episode", 0))
        if len(shard_records) != shard.n_runs:
            raise RuntimeError(f"{shard.name} has {len(shard_records)} records, expected {shard.n_runs}")
        for record in shard_records:
            merged = dict(record)
            merged["episode"] = next_episode
            merged["source_shard"] = shard.index
            records.append(merged)
            next_episode += 1
    if len(records) != TOTAL_EPISODES:
        raise RuntimeError(f"Merged {len(records)} records, expected {TOTAL_EPISODES}")

    result_dir().mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_id": MODEL_ID,
        "prompt": PROMPT,
        "target_n_episodes": TOTAL_EPISODES,
        "completed_episodes": len(records),
        "last_update": iso_now(),
        "episodes": records,
        "merged_from_shards": {
            "run_id": run_id,
            "n_shards": N_SHARDS,
            "shard_sizes": [shard.n_runs for shard in shards()],
        },
    }
    checkpoint_tmp = final_checkpoint_path().with_suffix(final_checkpoint_path().suffix + ".tmp")
    checkpoint_tmp.write_text(json.dumps(checkpoint, indent=2))
    checkpoint_tmp.replace(final_checkpoint_path())

    summary_tmp = final_summary_path().with_suffix(final_summary_path().suffix + ".tmp")
    summary_tmp.write_text(summary_text(records))
    summary_tmp.replace(final_summary_path())
    print(f"Merged final checkpoint: {final_checkpoint_path()}")
    print(f"Merged final summary:    {final_summary_path()}")


def manager(run_id: str, session: str) -> None:
    states = {
        shard.name: {
            "status": "pending",
            "attempts": 0,
            "completed": checkpoint_completed(shard_checkpoint(run_id, shard)),
            "target": shard.n_runs,
        }
        for shard in shards()
    }
    while True:
        for shard in shards():
            state = states[shard.name]
            if state["status"] == "running":
                monitor_shard(run_id, session, shard, state)
        for shard in shards():
            state = states[shard.name]
            if state["status"] == "pending":
                start_shard(run_id, session, shard, state)
        write_state(run_id, states)
        statuses = {state["status"] for state in states.values()}
        if statuses <= {"done"}:
            merge_shards(run_id)
            write_state(run_id, states)
            print("All shards complete and merged")
            return
        if "failed" in statuses and not (statuses & {"pending", "running"}):
            raise RuntimeError("At least one shard failed")
        time.sleep(60)


def start_queue(args) -> None:
    run_id = args.run_id or now_stamp()
    run_root(run_id).mkdir(parents=True, exist_ok=True)
    static_preflight()
    archive_existing_results(run_id)
    session = f"gal_split_nolang_{run_id}"
    if tmux_run(["has-session", "-t", session]).returncode == 0:
        raise RuntimeError(f"tmux session already exists: {session}")
    manager_log = run_root(run_id) / "manager.log"
    inner = (
        f"cd {shlex.quote(str(SCRIPT_DIR))}; "
        f"conda run --no-capture-output -n {shlex.quote(CONDA_ENV)} "
        f"python -u rerun_gal_split_nolang_correct_prompt.py manager "
        f"--run-id {shlex.quote(run_id)} --session {shlex.quote(session)} "
        f"2>&1 | tee -a {shlex.quote(str(manager_log))}"
    )
    tmux_run(["new-session", "-d", "-s", session, "-n", "manager", "bash", "-lc", inner], check=True)
    print(f"Started {session}")
    print(f"Run dir: {run_root(run_id)}")
    print(f"Status:  conda run --no-capture-output -n {CONDA_ENV} python rerun_gal_split_nolang_correct_prompt.py status --run-id {run_id}")


def status(run_id: str | None) -> None:
    if run_id is None:
        states = sorted((SCRIPT_DIR / "logs" / "gal_split_nolang_correct_prompt").glob("*/state.json"))
        if not states:
            raise RuntimeError("No rerun state files found")
        state_path = states[-1]
    else:
        state_path = run_root(run_id) / "state.json"
    data = json.loads(state_path.read_text())
    print(f"Run {data['run_id']} updated {data['updated_at']}")
    for name, state in data["states"].items():
        ports = ""
        if state.get("http_port"):
            ports = f" http={state.get('http_port')} livestream={state.get('livestream_port')}"
        print(
            f"{name:<8} {state.get('status'):<8} "
            f"{state.get('completed', 0):>3}/{state.get('target')} "
            f"{state.get('message', '')}{ports}"
        )
    if final_checkpoint_path().exists():
        print(f"Final completed: {checkpoint_completed(final_checkpoint_path())}/{TOTAL_EPISODES}")


def parse_args(argv: list[str]):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start")
    start.add_argument("--run-id")
    manager_parser = sub.add_parser("manager")
    manager_parser.add_argument("--run-id", required=True)
    manager_parser.add_argument("--session", required=True)
    status_parser = sub.add_parser("status")
    status_parser.add_argument("--run-id")
    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--run-id", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        if args.command == "start":
            start_queue(args)
        elif args.command == "manager":
            manager(args.run_id, args.session)
        elif args.command == "status":
            status(args.run_id)
        elif args.command == "merge":
            merge_shards(args.run_id)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
