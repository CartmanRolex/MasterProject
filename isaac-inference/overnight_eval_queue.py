#!/usr/bin/env python3
"""Preflight and tmux queue for overnight Isaac evaluations."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
REMOTE = SCRIPT_DIR / "remote.sh"
QUEUE_ROOT = SCRIPT_DIR / "logs" / "overnight_queue"
SEED_LIST_PATH = SCRIPT_DIR / "eval_seeds" / "pick_orange_reference_100_v1.json"
CONDA_ENV = os.environ.get("CONDA_ENV", "leisaac_envhub")
MAX_CONCURRENT = int(os.environ.get("QUEUE_MAX_CONCURRENT", "3"))
STARTUP_GRACE_SECONDS = int(os.environ.get("QUEUE_STARTUP_GRACE_SECONDS", str(20 * 60)))
STALL_SECONDS = int(os.environ.get("QUEUE_STALL_SECONDS", str(75 * 60)))
MAX_RETRIES = int(os.environ.get("QUEUE_MAX_RETRIES", "2"))
SMOKE_MAX_STEPS = int(os.environ.get("SMOKE_MAX_STEPS", "30"))
SMOKE_TIMEOUT_SECONDS = int(os.environ.get("SMOKE_TIMEOUT_SECONDS", str(30 * 60)))
CONCURRENCY_SMOKE_TIMEOUT_SECONDS = int(os.environ.get("CONCURRENCY_SMOKE_TIMEOUT_SECONDS", str(40 * 60)))
MIN_FREE_GB = int(os.environ.get("QUEUE_MIN_FREE_GB", "50"))


@dataclass(frozen=True)
class EvalJob:
    name: str
    script: str
    model_id: str
    kind: str
    n_runs: int = 100
    resume: bool = True
    fresh: bool = False
    instruction: str | None = None
    actions_per_chunk: int = 20
    result_name: str | None = None

    @property
    def result_dir(self) -> Path:
        name = self.result_name or self.model_id.rstrip("/").split("/")[-1]
        return RESULTS_DIR / name

    @property
    def checkpoint_path(self) -> Path:
        if self.kind == "autonomous":
            return self.result_dir / "checkpoint.json"
        if self.kind == "act":
            return self.result_dir / "act_checkpoint.json"
        return self.result_dir / "flat_checkpoint.json"

    @property
    def summary_path(self) -> Path:
        if self.kind == "autonomous":
            return self.result_dir / "latest.txt"
        if self.kind == "act":
            return self.result_dir / "act_latest.txt"
        return self.result_dir / "flat_latest.txt"


JOBS = [
    EvalJob(
        name="gal_ch20",
        script="inference_autonomous_orders.py",
        model_id="MasterProject2026/Gal-pick-orange-tailedCH20",
        kind="autonomous",
        resume=False,
        fresh=True,
        actions_per_chunk=20,
    ),
    EvalJob(
        name="gal_merged_auto",
        script="inference_autonomous_orders.py",
        model_id="MasterProject2026/Gal-merged-tailed-auto",
        kind="autonomous",
        resume=False,
        fresh=True,
        actions_per_chunk=20,
    ),
    EvalJob(
        name="gal_split_nolang",
        script="inference_flat_prompt.py",
        model_id="MasterProject2026/Gal_split_nolang",
        kind="flat",
        resume=False,
        fresh=True,
        instruction="Place the orange into plate",
        actions_per_chunk=20,
    ),
    EvalJob(
        name="gal_merged_nolang_nohome",
        script="inference_flat_prompt.py",
        model_id="MasterProject2026/Gal-merged-tailed-auto-no-lang-no-home",
        kind="flat",
        resume=False,
        fresh=True,
        instruction="Place the orange into plate",
        actions_per_chunk=20,
    ),
    EvalJob(
        name="pick_orange_mimic",
        script="inference_flat_prompt.py",
        model_id="MasterProject2026/pick-orange-mimic",
        kind="flat",
        resume=False,
        fresh=True,
        instruction="Grab orange and place into plate",
        actions_per_chunk=20,
    ),
    EvalJob(
        name="act_pick_orange_ch20",
        script="inference_act_flat_prompt.py",
        model_id="MasterProject2026/ACT-pick-orange",
        kind="act",
        resume=False,
        fresh=True,
        actions_per_chunk=20,
        result_name="ACT-pick-orange-chunk20",
    ),
    EvalJob(
        name="act_pick_orange_ch100",
        script="inference_act_flat_prompt.py",
        model_id="MasterProject2026/ACT-pick-orange",
        kind="act",
        resume=False,
        fresh=True,
        actions_per_chunk=100,
        result_name="ACT-pick-orange-chunk100",
    ),
]
JOBS_BY_NAME = {job.name: job for job in JOBS}
ARCHIVE_RESULT_NAMES = [
    "Gal-pick-orange-tailedCH20",
    "Gal-merged-tailed-auto",
    "Gal_split_nolang",
    "Gal-merged-tailed-auto-no-lang-no-home",
    "pick-orange-mimic",
    "ACT-pick-orange",
    "ACT-pick-orange-chunk20",
    "ACT-pick-orange-chunk100",
]


def now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def iso_now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def run_id_dir(run_id: str) -> Path:
    return QUEUE_ROOT / run_id


def run_checked(cmd, *, timeout=120, cwd=SCRIPT_DIR, env=None):
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {shlex.join(map(str, cmd))}\n{result.stdout}"
        )
    return result.stdout


def conda_python_cmd(*args: str) -> list[str]:
    return ["conda", "run", "--no-capture-output", "-n", CONDA_ENV, "python", *args]


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


def base_job_env(
    job: EvalJob,
    *,
    n_runs: int,
    max_steps: int,
    resume: bool,
    checkpoint_path: Path | None = None,
    summary_path: Path | None = None,
    log_path: Path | None = None,
    save_snapshots: bool = True,
    result_name: str | None = None,
) -> dict[str, str]:
    effective_result_name = result_name or job.result_name
    env = os.environ.copy()
    env.update(
        {
            "CONDA_ENV": CONDA_ENV,
            "MODEL_ID": job.model_id,
            "N_INFERENCE_RUNS": str(n_runs),
            "MAX_STEPS": str(max_steps),
            "ACTIONS_PER_CHUNK": str(job.actions_per_chunk),
            "EVAL_RESUME": "1" if resume else "0",
            "EVAL_SEED_LIST_PATH": str(SEED_LIST_PATH),
            "SAVE_CAMERA_SNAPSHOTS": "1" if save_snapshots else "0",
            "LIVESTREAM": "2",
            "ENABLE_LIVESTREAM": "1",
            "ENABLE_CAMERAS": "1",
        }
    )
    if checkpoint_path:
        env["EVAL_CHECKPOINT_PATH"] = str(checkpoint_path)
    if summary_path:
        env["EVAL_SUMMARY_PATH"] = str(summary_path)
    if log_path:
        env["REMOTE_LOG_FILE"] = str(log_path)
    if effective_result_name:
        env["EVAL_RESULT_NAME"] = effective_result_name
    if job.instruction is not None:
        env["INSTRUCTION"] = job.instruction
    return env


def static_preflight() -> None:
    print("Static preflight: compile scripts")
    scripts = [
        "inference_autonomous_orders.py",
        "inference_flat_prompt.py",
        "inference_act_flat_prompt.py",
        "eval_utils.py",
        "overnight_eval_queue.py",
    ]
    run_checked(conda_python_cmd("-m", "py_compile", *scripts), timeout=120)

    print("Static preflight: reference seed list")
    seed_code = (
        "from eval_utils import load_eval_seed_set; "
        f"s=load_eval_seed_set({str(SEED_LIST_PATH)!r}, min_count=100); "
        "assert s['count'] == 100, s; "
        "print(s['name'], s['sha256'])"
    )
    run_checked(conda_python_cmd("-c", seed_code), timeout=120)

    print("Static preflight: imports in leisaac_envhub")
    run_checked(
        conda_python_cmd(
            "-c",
            "import torch, lerobot, huggingface_hub; "
            "print(torch.__version__); print('imports ok')",
        ),
        timeout=120,
    )

    print("Static preflight: CUDA device")
    cuda_code = (
        "import torch; "
        "assert torch.cuda.is_available(), 'CUDA is not available'; "
        "name=torch.cuda.get_device_name(0); "
        "print(name); "
        "assert 'RTX 5090' in name, name"
    )
    run_checked(conda_python_cmd("-c", cuda_code), timeout=120)

    print("Static preflight: Hugging Face model configs")
    models = sorted({job.model_id for job in JOBS})
    hf_code = (
        "from huggingface_hub import HfApi; "
        f"models={models!r}; "
        "api=HfApi(); "
        "missing=[]; "
        "\nfor m in models:\n"
        "    info=api.model_info(m)\n"
        "    files=[s.rfilename for s in (info.siblings or [])]\n"
        "    print(f'{m}: {len(files)} files')\n"
        "    if 'config.json' not in files:\n"
        "        missing.append(m)\n"
        "assert not missing, missing\n"
    )
    run_checked(conda_python_cmd("-c", hf_code), timeout=180)

    print("Static preflight: tmux and disk")
    run_checked(["tmux", "-V"], timeout=30)
    free_gb = shutil.disk_usage(SCRIPT_DIR).free / (1024**3)
    if free_gb < MIN_FREE_GB:
        raise RuntimeError(f"Only {free_gb:.1f} GB free; need at least {MIN_FREE_GB} GB")
    print(f"Static preflight OK ({free_gb:.1f} GB free)")


def kill_process_group(proc: subprocess.Popen, grace_seconds=60) -> None:
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=10)


def smoke_start_snapshot_path(result_name: str, kind: str) -> Path:
    return RESULTS_DIR / result_name / "snapshots" / kind / "episode_000000_start_front.png"


def smoke_success_ready(
    checkpoint_path: Path,
    summary_path: Path,
    log_path: Path,
    *,
    job: EvalJob | None = None,
    result_name: str | None = None,
    require_snapshot: bool = False,
) -> tuple[bool, str]:
    if checkpoint_completed(checkpoint_path) < 1:
        return False, "checkpoint has no completed episode"
    try:
        checkpoint = json.loads(checkpoint_path.read_text())
    except Exception as exc:
        return False, f"checkpoint is unreadable: {exc}"
    episodes = checkpoint.get("episodes", [])
    first_episode = episodes[0] if episodes and isinstance(episodes[0], dict) else {}
    required_record_fields = ["seed", "seed_index", "seed_set_name", "seed_set_hash"]
    missing_record_fields = [field for field in required_record_fields if field not in first_episode]
    if missing_record_fields:
        return False, f"episode record missing {missing_record_fields}"
    if "initial_scene" not in first_episode and "initial_scene_audit" not in first_episode:
        return False, "episode record missing initial scene audit"
    if not checkpoint.get("seed_set", {}).get("sha256"):
        return False, "checkpoint missing seed_set hash"
    if not summary_path.exists():
        return False, "summary file is missing"
    if require_snapshot:
        if not job or not result_name:
            return False, "snapshot check missing job/result name"
        snapshot_path = smoke_start_snapshot_path(result_name, job.kind)
        if not snapshot_path.exists():
            return False, f"start snapshot is missing: {snapshot_path}"
    text = log_path.read_text(errors="ignore") if log_path.exists() else ""
    required = ["Loading LeIsaac Environment", "Loading trained", "Summary saved to"]
    missing = [needle for needle in required if needle not in text]
    if missing:
        return False, f"log missing {missing}"
    return True, "success markers found"


def run_remote_for_smoke(job: EvalJob, run_id: str, label: str, timeout: int) -> tuple[bool, str]:
    preflight_dir = run_id_dir(run_id) / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = preflight_dir / f"{job.name}_{label}_checkpoint.json"
    summary_path = preflight_dir / f"{job.name}_{label}_summary.txt"
    log_path = preflight_dir / f"{job.name}_{label}.log"
    smoke_result_name = f"_preflight_{run_id}_{job.name}_{label}"[:96]
    smoke_result_dir = RESULTS_DIR / smoke_result_name
    for path in (checkpoint_path, summary_path, log_path):
        if path.exists():
            path.unlink()
    if smoke_result_dir.exists():
        shutil.rmtree(smoke_result_dir)

    env = base_job_env(
        job,
        n_runs=1,
        max_steps=SMOKE_MAX_STEPS,
        resume=False,
        checkpoint_path=checkpoint_path,
        summary_path=summary_path,
        log_path=log_path,
        save_snapshots=True,
        result_name=smoke_result_name,
    )
    proc = subprocess.Popen(
        ["bash", str(REMOTE), job.script],
        cwd=SCRIPT_DIR,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        ready, _ = smoke_success_ready(
            checkpoint_path,
            summary_path,
            log_path,
            job=job,
            result_name=smoke_result_name,
            require_snapshot=True,
        )
        if ready:
            if proc.poll() is None:
                kill_process_group(proc, grace_seconds=10)
            shutil.rmtree(smoke_result_dir, ignore_errors=True)
            return True, f"{job.name} smoke OK; log: {log_path}"
        if proc.poll() is not None:
            break
        time.sleep(5)

    if proc.poll() is None:
        kill_process_group(proc)
        return False, f"{job.name} smoke timed out; log: {log_path}"
    if proc.returncode != 0:
        return False, f"{job.name} smoke exited {proc.returncode}; log: {log_path}"
    ready, reason = smoke_success_ready(
        checkpoint_path,
        summary_path,
        log_path,
        job=job,
        result_name=smoke_result_name,
        require_snapshot=True,
    )
    if not ready:
        return False, f"{job.name} smoke did not finish cleanly ({reason}); log: {log_path}"
    shutil.rmtree(smoke_result_dir, ignore_errors=True)
    return True, f"{job.name} smoke OK; log: {log_path}"


def individual_smoke_tests(run_id: str) -> None:
    print("Individual smoke tests")
    for job in JOBS:
        ok, message = run_remote_for_smoke(job, run_id, "individual", SMOKE_TIMEOUT_SECONDS)
        print(message)
        if not ok:
            raise RuntimeError(message)


def concurrency_smoke_test(run_id: str) -> None:
    print("Concurrency smoke test: first 3 jobs")
    jobs = JOBS[:MAX_CONCURRENT]
    procs = []
    preflight_dir = run_id_dir(run_id) / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)

    for job in jobs:
        checkpoint_path = preflight_dir / f"{job.name}_concurrent_checkpoint.json"
        summary_path = preflight_dir / f"{job.name}_concurrent_summary.txt"
        log_path = preflight_dir / f"{job.name}_concurrent.log"
        smoke_result_name = f"_preflight_{run_id}_{job.name}_concurrent"[:96]
        smoke_result_dir = RESULTS_DIR / smoke_result_name
        for path in (checkpoint_path, summary_path, log_path):
            if path.exists():
                path.unlink()
        if smoke_result_dir.exists():
            shutil.rmtree(smoke_result_dir)
        env = base_job_env(
            job,
            n_runs=1,
            max_steps=SMOKE_MAX_STEPS,
            resume=False,
            checkpoint_path=checkpoint_path,
            summary_path=summary_path,
            log_path=log_path,
            save_snapshots=True,
            result_name=smoke_result_name,
        )
        proc = subprocess.Popen(
            ["bash", str(REMOTE), job.script],
            cwd=SCRIPT_DIR,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        procs.append((job, proc, checkpoint_path, summary_path, log_path, smoke_result_name, smoke_result_dir))

    succeeded = set()
    failed = {}
    deadline = time.time() + CONCURRENCY_SMOKE_TIMEOUT_SECONDS
    while time.time() < deadline:
        for job, proc, checkpoint_path, summary_path, log_path, smoke_result_name, smoke_result_dir in procs:
            if job.name in succeeded or job.name in failed:
                continue
            ready, reason = smoke_success_ready(
                checkpoint_path,
                summary_path,
                log_path,
                job=job,
                result_name=smoke_result_name,
                require_snapshot=True,
            )
            if ready:
                succeeded.add(job.name)
                if proc.poll() is None:
                    kill_process_group(proc, grace_seconds=10)
                shutil.rmtree(smoke_result_dir, ignore_errors=True)
                print(f"{job.name} concurrent smoke OK; log: {log_path}")
                continue
            if proc.poll() is not None:
                failed[job.name] = f"{job.name} exited before success markers ({reason}); log: {log_path}"
        if len(succeeded) + len(failed) == len(procs):
            break
        time.sleep(10)
    else:
        for _, proc, _, _, _, _, _ in procs:
            if proc.poll() is None:
                kill_process_group(proc)
        raise RuntimeError("Concurrency smoke timed out")

    failures = list(failed.values())
    for job, proc, checkpoint_path, summary_path, log_path, smoke_result_name, smoke_result_dir in procs:
        if job.name in succeeded or job.name in failed:
            continue
        ready, reason = smoke_success_ready(
            checkpoint_path,
            summary_path,
            log_path,
            job=job,
            result_name=smoke_result_name,
            require_snapshot=True,
        )
        if ready:
            succeeded.add(job.name)
            shutil.rmtree(smoke_result_dir, ignore_errors=True)
            print(f"{job.name} concurrent smoke OK; log: {log_path}")
        else:
            failures.append(f"{job.name} did not finish cleanly ({reason}); log: {log_path}")
    if failures:
        raise RuntimeError("\n".join(failures))


def archive_fresh_results(run_id: str) -> None:
    archive_root = RESULTS_DIR / "_archive" / run_id
    for result_name in ARCHIVE_RESULT_NAMES:
        result_dir = RESULTS_DIR / result_name
        if not result_dir.exists():
            continue
        archive_root.mkdir(parents=True, exist_ok=True)
        dest = archive_root / result_dir.name
        suffix = 1
        while dest.exists():
            suffix += 1
            dest = archive_root / f"{result_dir.name}_{suffix}"
        print(f"Archiving {result_dir} -> {dest}")
        shutil.move(str(result_dir), str(dest))


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


def tmux_window_alive(window_id: str | None) -> bool:
    if not window_id:
        return False
    return tmux_run(["display-message", "-p", "-t", window_id, "#{window_id}"]).returncode == 0


def tmux_kill_window(window_id: str | None) -> None:
    if window_id and tmux_window_alive(window_id):
        tmux_run(["kill-window", "-t", window_id])


def tmux_interrupt_window(window_id: str | None) -> None:
    if window_id and tmux_window_alive(window_id):
        tmux_run(["send-keys", "-t", window_id, "C-c"])


def manager_state_path(run_id: str) -> Path:
    return run_id_dir(run_id) / "queue_state.json"


def write_manager_state(run_id: str, states: dict[str, dict]) -> None:
    path = manager_state_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "updated_at": iso_now(),
        "max_concurrent": MAX_CONCURRENT,
        "max_retries": MAX_RETRIES,
        "jobs": states,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def real_job_resume(job: EvalJob, state: dict) -> bool:
    if job.fresh:
        return job.checkpoint_path.exists()
    return job.resume


def start_real_job(job: EvalJob, state: dict, run_id: str, session: str) -> None:
    state["attempts"] += 1
    attempt = state["attempts"]
    window_name = f"{job.name}_{attempt}"[:32]
    job_log = run_id_dir(run_id) / f"{job.name}_attempt{attempt}.log"
    rc_file = run_id_dir(run_id) / f"{job.name}_attempt{attempt}.rc"
    for path in (job_log, rc_file):
        if path.exists():
            path.unlink()
    job_log.parent.mkdir(parents=True, exist_ok=True)

    env = base_job_env(
        job,
        n_runs=job.n_runs,
        max_steps=5000,
        resume=real_job_resume(job, state),
        log_path=job_log,
        save_snapshots=True,
    )
    exports = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env.items()) if key in {
        "CONDA_ENV",
        "MODEL_ID",
        "N_INFERENCE_RUNS",
        "MAX_STEPS",
        "ACTIONS_PER_CHUNK",
        "EVAL_RESUME",
        "EVAL_RESULT_NAME",
        "EVAL_SEED_LIST_PATH",
        "SAVE_CAMERA_SNAPSHOTS",
        "LIVESTREAM",
        "ENABLE_LIVESTREAM",
        "ENABLE_CAMERAS",
        "REMOTE_LOG_FILE",
        "INSTRUCTION",
    })
    inner = (
        "set -o pipefail; "
        f"cd {shlex.quote(str(SCRIPT_DIR))}; "
        f"{exports} ./remote.sh {shlex.quote(job.script)}; "
        "rc=$?; "
        f"echo $rc > {shlex.quote(str(rc_file))}; "
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
            window_name,
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
            "started_at": iso_now(),
            "started_ts": now,
            "last_progress_ts": now,
            "last_seen_mtime": 0.0,
            "window_id": window_id,
            "window_name": window_name,
            "log_file": str(job_log),
            "rc_file": str(rc_file),
            "checkpoint_file": str(job.checkpoint_path),
            "message": f"started attempt {attempt}",
        }
    )
    print(f"Started {job.name} attempt {attempt} in tmux window {window_id}")


def finish_or_retry_job(job: EvalJob, state: dict, reason: str, run_id: str, session: str) -> None:
    completed = checkpoint_completed(job.checkpoint_path)
    if completed >= job.n_runs:
        state.update({"status": "done", "finished_at": iso_now(), "completed": completed, "message": reason})
        print(f"Done {job.name}: {completed}/{job.n_runs}")
        return

    state["completed"] = completed
    if state["attempts"] <= MAX_RETRIES:
        print(f"Retrying {job.name}: {reason}; completed={completed}/{job.n_runs}")
        state.update({"status": "pending", "message": reason, "window_id": None})
        return

    state.update({"status": "failed", "finished_at": iso_now(), "message": reason})
    print(f"Failed {job.name}: {reason}; completed={completed}/{job.n_runs}")


def monitor_running_job(job: EvalJob, state: dict, run_id: str, session: str) -> None:
    checkpoint_path = job.checkpoint_path
    log_path = Path(state["log_file"]) if state.get("log_file") else None
    rc_file = Path(state["rc_file"]) if state.get("rc_file") else None
    progress_mtime = latest_mtime([p for p in (checkpoint_path, log_path) if p])
    if progress_mtime > float(state.get("last_seen_mtime", 0.0)):
        state["last_seen_mtime"] = progress_mtime
        state["last_progress_ts"] = time.time()

    completed = checkpoint_completed(checkpoint_path)
    state["completed"] = completed

    alive = tmux_window_alive(state.get("window_id"))
    if alive and completed >= job.n_runs and job.summary_path.exists():
        print(f"{job.name} reached {completed}/{job.n_runs}; stopping finished window")
        tmux_interrupt_window(state.get("window_id"))
        time.sleep(15)
        tmux_kill_window(state.get("window_id"))
        state.update(
            {
                "status": "done",
                "finished_at": iso_now(),
                "completed": completed,
                "message": "checkpoint target reached",
            }
        )
        return

    if not alive:
        rc = rc_file.read_text().strip() if rc_file and rc_file.exists() else "missing"
        finish_or_retry_job(job, state, f"process exited rc={rc}", run_id, session)
        return

    now = time.time()
    if now - float(state.get("started_ts", now)) < STARTUP_GRACE_SECONDS:
        return
    if now - float(state.get("last_progress_ts", now)) <= STALL_SECONDS:
        return

    print(f"Stall detected for {job.name}; interrupting window {state.get('window_id')}")
    tmux_interrupt_window(state.get("window_id"))
    time.sleep(60)
    tmux_kill_window(state.get("window_id"))
    finish_or_retry_job(job, state, "stalled without checkpoint/log progress", run_id, session)


def run_manager(run_id: str, session: str) -> None:
    print(f"Manager started for {run_id} in tmux session {session}")
    states = {
        job.name: {
            "status": "pending",
            "attempts": 0,
            "completed": checkpoint_completed(job.checkpoint_path),
            "model_id": job.model_id,
            "result_name": job.result_dir.name,
            "script": job.script,
            "target_runs": job.n_runs,
            "actions_per_chunk": job.actions_per_chunk,
            "fresh": job.fresh,
        }
        for job in JOBS
    }

    while True:
        for job in JOBS:
            state = states[job.name]
            if state["status"] == "running":
                monitor_running_job(job, state, run_id, session)

        running = sum(1 for state in states.values() if state["status"] == "running")
        for job in JOBS:
            if running >= MAX_CONCURRENT:
                break
            state = states[job.name]
            if state["status"] != "pending":
                continue
            if checkpoint_completed(job.checkpoint_path) >= job.n_runs:
                state.update(
                    {
                        "status": "done",
                        "finished_at": iso_now(),
                        "completed": checkpoint_completed(job.checkpoint_path),
                        "message": "already complete",
                    }
                )
                continue
            start_real_job(job, state, run_id, session)
            running += 1

        write_manager_state(run_id, states)
        statuses = {state["status"] for state in states.values()}
        if not (statuses & {"pending", "running"}):
            print("Queue finished")
            break
        time.sleep(60)


def start_manager_session(run_id: str) -> str:
    session = f"overnight_eval_{run_id}"
    if tmux_run(["has-session", "-t", session]).returncode == 0:
        raise RuntimeError(f"tmux session already exists: {session}")

    manager_log = run_id_dir(run_id) / "manager.log"
    manager_log.parent.mkdir(parents=True, exist_ok=True)
    inner = (
        f"cd {shlex.quote(str(SCRIPT_DIR))}; "
        f"conda run --no-capture-output -n {shlex.quote(CONDA_ENV)} "
        f"python -u overnight_eval_queue.py manager --run-id {shlex.quote(run_id)} "
        f"--session {shlex.quote(session)} 2>&1 | tee -a {shlex.quote(str(manager_log))}"
    )
    tmux_run(["new-session", "-d", "-s", session, "-n", "manager", "bash", "-lc", inner], check=True)
    return session


def run_preflight(run_id: str, *, static_only: bool) -> None:
    static_preflight()
    if static_only:
        return
    individual_smoke_tests(run_id)
    concurrency_smoke_test(run_id)


def start_queue(args) -> None:
    run_id = args.run_id or now_stamp()
    run_id_dir(run_id).mkdir(parents=True, exist_ok=True)
    if not args.skip_preflight:
        run_preflight(run_id, static_only=args.static_only)
        if args.static_only:
            print("Static-only preflight complete; not starting queue.")
            return
    if args.no_start:
        print("Preflight complete; --no-start requested.")
        return
    archive_fresh_results(run_id)
    session = start_manager_session(run_id)
    print(f"Queue started in tmux session: {session}")
    print(f"Manager log: {run_id_dir(run_id) / 'manager.log'}")
    print(f"Status file: {manager_state_path(run_id)}")
    print(f"Attach: tmux attach -t {session}")


def print_status(run_id: str | None) -> None:
    if run_id is None:
        states = sorted(QUEUE_ROOT.glob("*/queue_state.json"), key=lambda p: p.stat().st_mtime)
        if not states:
            raise RuntimeError("No queue state files found")
        state_path = states[-1]
    else:
        state_path = manager_state_path(run_id)
    data = json.loads(state_path.read_text())
    print(f"Queue {data['run_id']} updated {data['updated_at']}")
    for name, state in data["jobs"].items():
        print(
            f"{name:<18} {state.get('status'):<8} "
            f"{state.get('completed', 0):>3}/{state.get('target_runs', '?')} "
            f"attempts={state.get('attempts')} {state.get('message', '')}"
        )


def parse_args(argv: list[str]):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="run preflight, archive fresh results, then start tmux queue")
    start.add_argument("--run-id")
    start.add_argument("--skip-preflight", action="store_true")
    start.add_argument("--static-only", action="store_true")
    start.add_argument("--no-start", action="store_true")

    preflight = sub.add_parser("preflight", help="run preflight checks only")
    preflight.add_argument("--run-id", default=None)
    preflight.add_argument("--static-only", action="store_true")

    manager = sub.add_parser("manager", help="internal tmux manager mode")
    manager.add_argument("--run-id", required=True)
    manager.add_argument("--session", required=True)

    status = sub.add_parser("status", help="print latest queue status")
    status.add_argument("--run-id")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        if args.command == "start":
            start_queue(args)
        elif args.command == "preflight":
            run_preflight(args.run_id or now_stamp(), static_only=args.static_only)
        elif args.command == "manager":
            run_manager(args.run_id, args.session)
        elif args.command == "status":
            print_status(args.run_id)
        return 0
    except KeyboardInterrupt:
        print("Interrupted")
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
