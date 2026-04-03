#!/usr/bin/env python3
"""Live monitor for the GRPO multi-model job chain.

Shows the status of every job in the pipeline (completed, failed, running,
waiting) and, for the active training job, displays the current curriculum
stage and training step in real time.

When the active job finishes, the monitor automatically picks up the next
job's log.

Usage:
    python3 -m src.utils.chain_monitor              # default, auto-detect
    python3 -m src.utils.chain_monitor --poll 30    # poll every 30s (default 15)

Designed to run on the cluster login node (same node as the watcher).
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
PROJ_DIR = Path(os.environ.get("HOME", "~")) / "GRPO-strict-generation"
CHAIN_FILE = PROJ_DIR / ".job_chain"
CHAIN_PID_FILE = PROJ_DIR / ".chain_pid"
LOGS_DIR = PROJ_DIR / "logs"
CHAIN_LOG = LOGS_DIR / "chain_watcher.log"

# Regex patterns for training log parsing
_KV_PATTERN = re.compile(r"step=(\d+)")
_STAGE_START = re.compile(r"\[stage (\d+)\] steps=(\d+)")
_STAGE_DONE = re.compile(r"\[stage (\d+)\] (\S+) completed")
_CURRICULUM_DONE = re.compile(r"CURRICULUM TRAINING COMPLETE")
_EVAL_CHECKPOINT = re.compile(r"Evaluating: (.+)")
_EVAL_PASS = re.compile(r"(\S+) Pass@1: ([\d.]+)")
_EVAL_COMPLETE = re.compile(r"Evaluation complete")


# ── Data structures ───────────────────────────────────────────────────────────
@dataclass
class JobInfo:
    """Info about a single job in the chain."""

    job_type: str  # "train" or "eval"
    config: str  # config path
    tag: str  # model tag (e.g. "smollm2-135m")
    slurm_id: str | None = None
    state: str = "WAITING"  # WAITING, RUNNING, COMPLETED, FAILED, TIMEOUT
    stage: int = 0  # current curriculum stage (1-3), 0 = not started
    stage_name: str = ""
    step: int = 0  # current training step
    stage_total: int = 0  # total steps for current stage
    eval_label: str = ""  # current eval label
    eval_pass: str = ""  # last eval pass@1
    exit_code: str = ""

    @property
    def label(self) -> str:
        return f"{self.job_type}-{self.tag}"


def _run(cmd: str) -> str:
    """Run a shell command and return stdout."""
    try:
        r = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return r.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


# ── SLURM queries ─────────────────────────────────────────────────────────────
def _get_slurm_jobs() -> dict[str, tuple[str, str, str]]:
    """Get recent SLURM jobs. Returns {job_name: (job_id, state, exit_code)}."""
    # sacct for the last 2 days, only our jobs
    out = _run(
        "sacct --me --starttime=$(date -d '2 days ago' +%Y-%m-%d) "
        "--format=JobID%20,JobName%30,State%15,ExitCode%10 --noheader --parsable2"
    )
    jobs: dict[str, tuple[str, str, str]] = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        job_id, name, state, exit_code = (
            parts[0],
            parts[1],
            parts[2],
            parts[3],
        )
        # Skip sub-steps (e.g. "12345.batch", "12345.extern")
        if "." in job_id:
            continue
        jobs[name] = (job_id, state, exit_code)
    return jobs


def _get_active_job() -> tuple[str, str] | None:
    """Return (job_id, job_name) of the currently running job, or None."""
    out = _run('squeue --me --noheader --format="%i %j %T"')
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[2] in ("RUNNING", "PENDING"):
            return parts[0], parts[1]
    return None


# ── Chain file parsing ────────────────────────────────────────────────────────
def _parse_chain_log() -> list[str]:
    """Parse chain_watcher.log to find submitted job names in order."""
    if not CHAIN_LOG.exists():
        return []
    submitted: list[str] = []
    # Lines like: [chain] Sottometto: train smollm2-135m (...) — N rimanenti — date
    pattern = re.compile(r"\[chain\] Sottometto: (\w+) (\S+)")
    for line in CHAIN_LOG.read_text(errors="replace").splitlines():
        m = pattern.search(line)
        if m:
            submitted.append(f"{m.group(1)}-{m.group(2)}")
    return submitted


def _read_pending_chain() -> list[tuple[str, str, str]]:
    """Read remaining entries from .job_chain file."""
    if not CHAIN_FILE.exists():
        return []
    entries = []
    for line in CHAIN_FILE.read_text().strip().splitlines():
        parts = line.strip().split(":")
        if len(parts) == 3:
            entries.append((parts[0], parts[1], parts[2]))
    return entries


def _find_log_file(job_type: str, slurm_id: str) -> Path | None:
    """Find the SLURM log file for a job."""
    log = LOGS_DIR / f"slurm-{job_type}-{slurm_id}.log"
    if log.exists():
        return log
    return None


# ── Log parsing ───────────────────────────────────────────────────────────────
def _parse_training_log(log_path: Path, job: JobInfo) -> None:
    """Parse a training log file and update job state."""
    try:
        content = log_path.read_text(errors="replace")
    except OSError:
        return

    for line in content.splitlines():
        # Stage start: [stage 1] steps=800 temp=0.8 ...
        m = _STAGE_START.search(line)
        if m:
            job.stage = int(m.group(1))
            job.stage_total = int(m.group(2))

        # Stage done: [stage 1] format_basics completed
        m = _STAGE_DONE.search(line)
        if m:
            job.stage_name = m.group(2)

        # Step: step=240 or key=value lines
        m = _KV_PATTERN.search(line)
        if m:
            job.step = int(m.group(1))

        # Curriculum complete
        if _CURRICULUM_DONE.search(line):
            job.stage_name = "COMPLETE"


def _parse_eval_log(log_path: Path, job: JobInfo) -> None:
    """Parse an eval log file and update job state."""
    try:
        content = log_path.read_text(errors="replace")
    except OSError:
        return

    for line in content.splitlines():
        m = _EVAL_CHECKPOINT.search(line)
        if m:
            job.eval_label = m.group(1)

        m = _EVAL_PASS.search(line)
        if m:
            job.eval_label = m.group(1)
            job.eval_pass = m.group(2)

        if _EVAL_COMPLETE.search(line):
            job.eval_label = "COMPLETE"


# ── Build full pipeline view ──────────────────────────────────────────────────
def _build_pipeline() -> list[JobInfo]:
    """Reconstruct the full pipeline from chain log + chain file + sacct."""
    slurm_jobs = _get_slurm_jobs()
    active = _get_active_job()
    submitted_names = _parse_chain_log()
    pending = _read_pending_chain()

    jobs: list[JobInfo] = []
    seen_names: set[str] = set()

    # 1. Already submitted jobs (from chain log)
    for name in submitted_names:
        if name in seen_names:
            continue
        seen_names.add(name)

        parts = name.split("-", 1)
        job_type = parts[0] if parts else "train"
        tag = parts[1] if len(parts) > 1 else name

        job = JobInfo(job_type=job_type, config="", tag=tag)

        # Match to SLURM
        if name in slurm_jobs:
            sid, state, exit_code = slurm_jobs[name]
            job.slurm_id = sid
            job.exit_code = exit_code
            if state == "RUNNING":
                job.state = "RUNNING"
            elif state == "PENDING":
                job.state = "PENDING"
            elif state == "COMPLETED":
                job.state = (
                    "COMPLETED" if exit_code == "0:0" else "FAILED"
                )
            elif state in ("FAILED", "NODE_FAIL", "OUT_OF_MEMORY"):
                job.state = "FAILED"
            elif state == "TIMEOUT":
                job.state = "TIMEOUT"
            elif state == "CANCELLED":
                job.state = "CANCELLED"
            else:
                job.state = state

            # Parse log for details
            log_file = _find_log_file(job_type, sid)
            if log_file:
                if job_type == "train":
                    _parse_training_log(log_file, job)
                else:
                    _parse_eval_log(log_file, job)
        elif active and active[1] == name:
            job.slurm_id = active[0]
            job.state = "RUNNING"

        jobs.append(job)

    # 2. Pending jobs (still in .job_chain)
    for job_type, cfg, tag in pending:
        name = f"{job_type}-{tag}"
        if name in seen_names:
            continue
        seen_names.add(name)
        jobs.append(
            JobInfo(job_type=job_type, config=cfg, tag=tag, state="WAITING")
        )

    return jobs


# ── Display ───────────────────────────────────────────────────────────────────
_STATE_ICONS = {
    "COMPLETED": "\033[32m✓\033[0m",  # green
    "FAILED": "\033[31m✗\033[0m",  # red
    "TIMEOUT": "\033[31m⏱\033[0m",  # red
    "CANCELLED": "\033[33m⊘\033[0m",  # yellow
    "RUNNING": "\033[36m▶\033[0m",  # cyan
    "PENDING": "\033[33m⏳\033[0m",  # yellow
    "WAITING": "\033[90m·\033[0m",  # gray
}


def _format_status(job: JobInfo) -> str:
    """Format a single line for a job."""
    icon = _STATE_ICONS.get(job.state, "?")
    slurm = f"[{job.slurm_id}]" if job.slurm_id else ""

    detail = ""
    if job.state == "RUNNING" and job.job_type == "train":
        if job.stage > 0:
            detail = f" stage {job.stage}/3 step {job.step}/{job.stage_total}"
        elif job.step > 0:
            detail = f" step {job.step}"
        else:
            detail = " starting..."
    elif job.state == "RUNNING" and job.job_type == "eval":
        if job.eval_label:
            detail = f" → {job.eval_label}"
            if job.eval_pass:
                detail += f" (pass@1={job.eval_pass})"
        else:
            detail = " starting..."
    elif job.state == "COMPLETED" and job.job_type == "train":
        if job.stage_name == "COMPLETE":
            detail = " ✓ all 3 stages"
        elif job.stage > 0:
            detail = f" completed at stage {job.stage}"
    elif job.state == "COMPLETED" and job.job_type == "eval":
        if job.eval_pass:
            detail = f" pass@1={job.eval_pass}"
    elif job.state == "FAILED":
        detail = f" exit={job.exit_code}" if job.exit_code else ""

    return f" {icon}  {job.label:<25s} {slurm:<12s} {job.state:<12s}{detail}"


def _watcher_status() -> str:
    """Check if the watcher process is alive."""
    if not CHAIN_PID_FILE.exists():
        return "\033[31mWatcher: OFF\033[0m"
    try:
        pid = CHAIN_PID_FILE.read_text().strip()
        result = _run(f"ps -p {pid} -o pid= 2>/dev/null")
        if result:
            return f"\033[32mWatcher: ON (PID {pid})\033[0m"
        else:
            return f"\033[31mWatcher: DEAD (PID {pid})\033[0m"
    except Exception:
        return "\033[31mWatcher: UNKNOWN\033[0m"


def _display(jobs: list[JobInfo]) -> None:
    """Print the full pipeline status."""
    completed = sum(1 for j in jobs if j.state == "COMPLETED")
    failed = sum(1 for j in jobs if j.state in ("FAILED", "TIMEOUT"))
    total = len(jobs)

    print("\033[2J\033[H", end="")  # clear screen
    print("=" * 65)
    print(
        f"  GRPO Pipeline Monitor — {completed}/{total} done"
        + (f", {failed} failed" if failed else "")
    )
    print(f"  {_watcher_status()}")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    print()

    current_model = ""
    for job in jobs:
        # Group separator by model
        if job.tag != current_model:
            current_model = job.tag
            print(f"  \033[1m{current_model}\033[0m")

        print(_format_status(job))

    print()
    print("-" * 65)
    remaining = sum(1 for j in jobs if j.state in ("WAITING", "PENDING"))
    running = [j for j in jobs if j.state == "RUNNING"]
    if running:
        j = running[0]
        if j.job_type == "train" and j.stage > 0:
            print(
                f"  Active: {j.label} — stage {j.stage}/3, step {j.step}"
            )
        elif j.job_type == "eval" and j.eval_label:
            print(f"  Active: {j.label} — {j.eval_label}")
        else:
            print(f"  Active: {j.label}")
    elif remaining > 0:
        print(f"  Waiting for next job... ({remaining} remaining)")
    else:
        print("  Pipeline finished!")
    print()


# ── Live follow mode ──────────────────────────────────────────────────────────
def _tail_active_job(jobs: list[JobInfo]) -> str | None:
    """Find and return the log path of the currently active job."""
    for job in jobs:
        if job.state == "RUNNING" and job.slurm_id:
            log = _find_log_file(job.job_type, job.slurm_id)
            return str(log) if log else None
    return None


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Live monitor for the GRPO multi-model job chain"
    )
    parser.add_argument(
        "--poll",
        type=int,
        default=15,
        help="Seconds between refresh (default: 15)",
    )
    args = parser.parse_args()

    print("GRPO Pipeline Monitor — Ctrl+C to exit")
    print(f"Polling every {args.poll}s...")
    print()

    try:
        while True:
            jobs = _build_pipeline()
            _display(jobs)

            # Check if pipeline is done
            all_done = all(
                j.state
                in ("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED")
                for j in jobs
            )
            watcher_alive = CHAIN_PID_FILE.exists()
            no_pending = not CHAIN_FILE.exists() or not _read_pending_chain()

            if all_done and not watcher_alive and no_pending and jobs:
                print("Pipeline complete. Exiting.")
                break

            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
