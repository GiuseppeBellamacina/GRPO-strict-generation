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
PROJ_DIR = (
    Path(os.environ.get("HOME", "~")) / "GRPO-strict-generation"
)
CHAIN_FILE = PROJ_DIR / ".job_chain"
CHAIN_PID_FILE = PROJ_DIR / ".chain_pid"
LOGS_DIR = PROJ_DIR / "logs"
CHAIN_LOG = LOGS_DIR / "chain_watcher.log"

# Regex patterns for training log parsing
_KV_STEP = re.compile(r"^\s+step=(\d+)\s+loss=")
_STAGE_START = re.compile(r"\[stage (\d+)\] steps=(\d+)")
_STAGE_DONE = re.compile(r"\[stage (\d+)\] (\S+) completed")
_CURRICULUM_DONE = re.compile(r"CURRICULUM TRAINING COMPLETE")
# tqdm progress bar: " 47%|████▋     | 420/900 [29:23<25:49"
_TQDM_PROGRESS = re.compile(r"\|\s*(\d+)/(\d+)\s*\[")
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
    state: str = (
        "WAITING"  # WAITING, RUNNING, COMPLETED, FAILED, TIMEOUT
    )
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
def _tail_lines(log_path: Path, n: int = 500) -> list[str]:
    """Read the last N lines of a file efficiently using tail."""
    out = _run(f"tail -n {n} '{log_path}'")
    if out:
        return out.splitlines()
    # Fallback: read from Python (slow but works)
    try:
        return log_path.read_text(errors="replace").splitlines()[-n:]
    except OSError:
        return []


def _grep_lines(
    log_path: Path, pattern: str, max_count: int = 20
) -> list[str]:
    """Grep a file for a pattern, returning matching lines."""
    out = _run(
        f"grep -E '{pattern}' '{log_path}' | tail -n {max_count}"
    )
    return out.splitlines() if out else []


def _parse_training_log(log_path: Path, job: JobInfo) -> None:
    """Parse a training log file and update job state.

    Uses grep + tail for efficiency on large files.
    """
    # 1. Find all stage markers (few lines in the whole file)
    stage_lines = _grep_lines(log_path, r"\[stage [0-9]+\]")
    for line in stage_lines:
        m = _STAGE_START.search(line)
        if m:
            job.stage = int(m.group(1))
            job.stage_total = int(m.group(2))

        m = _STAGE_DONE.search(line)
        if m:
            job.stage_name = m.group(2)

    # 2. Check for curriculum complete
    complete_lines = _grep_lines(
        log_path, "CURRICULUM TRAINING COMPLETE", max_count=1
    )
    if complete_lines:
        job.stage_name = "COMPLETE"

    # 3. Get current step from last lines (step= or tqdm progress bar)
    tail = _tail_lines(log_path, n=100)
    for line in reversed(tail):
        # Try key=value format: "  step=420  loss=0.005..."
        m = _KV_STEP.search(line)
        if m:
            job.step = int(m.group(1))
            break

        # Try tqdm progress bar: "47%|████▋| 420/900 ["
        m = _TQDM_PROGRESS.search(line)
        if m:
            job.step = int(m.group(1))
            if job.stage_total == 0:
                job.stage_total = int(m.group(2))
            break


def _parse_eval_log(log_path: Path, job: JobInfo) -> None:
    """Parse an eval log file and update job state."""
    tail = _tail_lines(log_path, n=200)

    for line in tail:
        m = _EVAL_CHECKPOINT.search(line)
        if m:
            job.eval_label = m.group(1)

        m = _EVAL_PASS.search(line)
        if m:
            job.eval_label = m.group(1)
            job.eval_pass = m.group(2)

        if _EVAL_COMPLETE.search(line):
            job.eval_label = "COMPLETE"

    # Get tqdm generation progress from the last lines
    for line in reversed(tail):
        m = _TQDM_PROGRESS.search(line)
        if m:
            job.step = int(m.group(1))
            job.stage_total = int(m.group(2))
            break


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
            # Parse log even when sacct didn't find it
            log_file = _find_log_file(job_type, active[0])
            if log_file:
                if job_type == "train":
                    _parse_training_log(log_file, job)
                else:
                    _parse_eval_log(log_file, job)

        jobs.append(job)

    # 2. Pending jobs (still in .job_chain)
    for job_type, cfg, tag in pending:
        name = f"{job_type}-{tag}"
        if name in seen_names:
            continue
        seen_names.add(name)

        job = JobInfo(
            job_type=job_type,
            config=cfg,
            tag=tag,
            state="PENDING",
        )

        # Check sacct/squeue in case the chain_log missed it
        if name in slurm_jobs:
            sid, state, exit_code = slurm_jobs[name]
            job.slurm_id = sid
            job.exit_code = exit_code
            if state == "COMPLETED":
                job.state = (
                    "COMPLETED" if exit_code == "0:0" else "FAILED"
                )
            elif state == "RUNNING":
                job.state = "RUNNING"
            elif state in ("FAILED", "NODE_FAIL", "OUT_OF_MEMORY"):
                job.state = "FAILED"
            elif state == "TIMEOUT":
                job.state = "TIMEOUT"
            elif state == "CANCELLED":
                job.state = "CANCELLED"

            log_file = _find_log_file(job_type, sid)
            if log_file:
                if job_type == "train":
                    _parse_training_log(log_file, job)
                else:
                    _parse_eval_log(log_file, job)
        elif active and active[1] == name:
            job.slurm_id = active[0]
            job.state = "RUNNING"
            log_file = _find_log_file(job_type, active[0])
            if log_file:
                if job_type == "train":
                    _parse_training_log(log_file, job)
                else:
                    _parse_eval_log(log_file, job)

        jobs.append(job)

    return jobs


# ── ANSI color helpers ────────────────────────────────────────────────────────
_RST = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"
_WHITE = "\033[97m"
_GRAY = "\033[90m"
_BG_GREEN = "\033[42m"
_BG_RED = "\033[41m"
_BG_YELLOW = "\033[43m"
_BG_CYAN = "\033[46m"


# ── Display ───────────────────────────────────────────────────────────────────
_STATE_ICONS = {
    "COMPLETED": f"{_GREEN}✓{_RST}",
    "FAILED": f"{_RED}✗{_RST}",
    "TIMEOUT": f"{_RED}⏱{_RST}",
    "CANCELLED": f"{_YELLOW}⊘{_RST}",
    "RUNNING": f"{_CYAN}▶{_RST}",
    "PENDING": f"{_GRAY}·{_RST}",
}

_STATE_COLORS = {
    "COMPLETED": _GREEN,
    "FAILED": _RED,
    "TIMEOUT": _RED,
    "CANCELLED": _YELLOW,
    "RUNNING": _CYAN,
    "PENDING": _GRAY,
}

_TYPE_COLORS = {
    "train": _BLUE,
    "eval": _MAGENTA,
}


def _format_status(job: JobInfo) -> str:
    """Format a single line for a job."""
    icon = _STATE_ICONS.get(job.state, "?")
    sc = _STATE_COLORS.get(job.state, "")
    tc = _TYPE_COLORS.get(job.job_type, "")
    slurm = f"{_DIM}[{job.slurm_id}]{_RST}" if job.slurm_id else ""

    detail = ""
    if job.state == "RUNNING" and job.job_type == "train":
        if job.stage > 0:
            pct = (
                int(job.step / job.stage_total * 100)
                if job.stage_total
                else 0
            )
            bar_w = 12
            filled = int(bar_w * pct / 100)
            bar = f"{_CYAN}{'█' * filled}{_GRAY}{'░' * (bar_w - filled)}{_RST}"
            detail = f" stage {_WHITE}{_BOLD}{job.stage}/3{_RST} {bar} {_WHITE}{job.step}{_RST}/{job.stage_total} {_DIM}({pct}%){_RST}"
        elif job.step > 0:
            detail = f" step {_WHITE}{job.step}{_RST}"
        else:
            detail = f" {_DIM}starting...{_RST}"
    elif job.state == "RUNNING" and job.job_type == "eval":
        if job.eval_label:
            detail = f" → {_WHITE}{job.eval_label}{_RST}"
            if job.step > 0 and job.stage_total > 0:
                pct = int(job.step / job.stage_total * 100)
                bar_w = 12
                filled = int(bar_w * pct / 100)
                bar = f"{_MAGENTA}{'█' * filled}{_GRAY}{'░' * (bar_w - filled)}{_RST}"
                detail += f" {bar} {_WHITE}{job.step}{_RST}/{job.stage_total} {_DIM}({pct}%){_RST}"
            if job.eval_pass:
                detail += f" {_GREEN}(pass@1={job.eval_pass}){_RST}"
        else:
            detail = f" {_DIM}starting...{_RST}"
    elif job.state == "COMPLETED" and job.job_type == "train":
        if job.stage_name == "COMPLETE":
            detail = f" {_GREEN}✓ all 3 stages{_RST}"
        elif job.stage > 0:
            detail = f" {_GREEN}completed at stage {job.stage}{_RST}"
    elif job.state == "COMPLETED" and job.job_type == "eval":
        if job.eval_pass:
            detail = f" {_GREEN}pass@1={job.eval_pass}{_RST}"
    elif job.state == "FAILED":
        detail = (
            f" {_RED}exit={job.exit_code}{_RST}"
            if job.exit_code
            else ""
        )

    label = f"{tc}{job.job_type}{_RST}-{_BOLD}{job.tag}{_RST}"
    state_str = f"{sc}{job.state}{_RST}"

    return (
        f" {icon}  {label:<40s} {slurm:<20s} {state_str:<22s}{detail}"
    )


def _watcher_status() -> str:
    """Check if the watcher process is alive."""
    if not CHAIN_PID_FILE.exists():
        return f"{_RED}Watcher: OFF{_RST}"
    try:
        pid = CHAIN_PID_FILE.read_text().strip()
        result = _run(f"ps -p {pid} -o pid= 2>/dev/null")
        if result:
            return (
                f"{_GREEN}Watcher: ON{_RST} {_DIM}(PID {pid}){_RST}"
            )
        else:
            return (
                f"{_RED}Watcher: DEAD{_RST} {_DIM}(PID {pid}){_RST}"
            )
    except Exception:
        return f"{_RED}Watcher: UNKNOWN{_RST}"


def _display(jobs: list[JobInfo]) -> None:
    """Print the full pipeline status."""
    completed = sum(1 for j in jobs if j.state == "COMPLETED")
    failed = sum(1 for j in jobs if j.state in ("FAILED", "TIMEOUT"))
    total = len(jobs)

    # Build summary badges
    done_badge = f"{_GREEN}{completed}{_RST}/{total} done"
    fail_badge = f"  {_RED}{failed} failed{_RST}" if failed else ""

    # Move cursor home + clear from cursor to end of screen
    # (avoids full clear → no flicker)
    print("\033[H\033[J", end="")
    print(f"{_CYAN}{'═' * 65}{_RST}")
    print(
        f"  {_BOLD}{_CYAN}GRPO Pipeline Monitor{_RST} — {done_badge}{fail_badge}"
    )
    print(f"  {_watcher_status()}")
    print(f"  {_DIM}{time.strftime('%Y-%m-%d %H:%M:%S')}{_RST}")
    print(f"{_CYAN}{'═' * 65}{_RST}")
    print()

    current_model = ""
    for job in jobs:
        # Group separator by model
        if job.tag != current_model:
            current_model = job.tag
            print(f"  {_BOLD}{_YELLOW}▸ {current_model}{_RST}")

        print(_format_status(job))

    print()
    print(f"{_DIM}{'─' * 65}{_RST}")
    remaining = sum(1 for j in jobs if j.state == "PENDING")
    running = [j for j in jobs if j.state == "RUNNING"]
    if running:
        j = running[0]
        tc = _TYPE_COLORS.get(j.job_type, "")
        if j.job_type == "train" and j.stage > 0:
            print(
                f"  {_CYAN}▶ Active:{_RST} {tc}{j.job_type}{_RST}-{_BOLD}{j.tag}{_RST}"
                f" — stage {_WHITE}{j.stage}/3{_RST}, step {_WHITE}{j.step}{_RST}"
            )
        elif j.job_type == "eval" and j.eval_label:
            print(
                f"  {_CYAN}▶ Active:{_RST} {tc}{j.job_type}{_RST}-{_BOLD}{j.tag}{_RST}"
                f" — {_WHITE}{j.eval_label}{_RST}"
            )
        else:
            print(
                f"  {_CYAN}▶ Active:{_RST} {tc}{j.job_type}{_RST}-{_BOLD}{j.tag}{_RST}"
            )
    elif remaining > 0:
        print(
            f"  {_YELLOW}⏳ Waiting for next job... ({remaining} remaining){_RST}"
        )
    else:
        print(f"  {_GREEN}{_BOLD}✓ Pipeline finished!{_RST}")
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
            no_pending = (
                not CHAIN_FILE.exists() or not _read_pending_chain()
            )

            if all_done and not watcher_alive and no_pending and jobs:
                print("Pipeline complete. Exiting.")
                break

            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
