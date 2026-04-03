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
# tqdm progress bar: " 47%|████▋     | 420/900 [29:23<25:49"
# HF Trainer's bar starts at line beginning (optional whitespace + percentage).
# This avoids matching unrelated tqdm bars like "Saving dataset: 100%|...|1500/1500["
_TQDM_PROGRESS = re.compile(
    r"^\s*\d+%\|.*\|\s*(\d+)/(\d+)\s*\[", re.MULTILINE
)
# Eval generation bar: "Generating:  45%|████▍| 17/38 ["
_TQDM_GENERATING = re.compile(r"Generating.*\|\s*(\d+)/(\d+)\s*\[")
# tqdm time info: "[04:25<37:02, 33.17s/it]" or "[1:23:45<2:03:04"
_TQDM_TIME = re.compile(r"\[([\d:]+)<([\d:]+)")
_EVAL_CHECKPOINT = re.compile(r"Evaluating: (.+)")
_EVAL_STAGE_NUM = re.compile(r"Stage (\d+)")
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
        "PENDING"  # PENDING, STARTING, RUNNING, COMPLETED, FAILED
    )
    stage: int = 0  # current curriculum stage (1-3), 0 = not started
    stage_name: str = ""
    step: int = 0  # current training step
    stage_total: int = 0  # total steps for current stage
    eval_label: str = ""  # current eval label
    eval_pass: str = ""  # last eval pass@1
    eval_step_total: int = 0  # total generation batches for eval
    exit_code: str = ""
    elapsed: str = ""  # elapsed time from squeue (e.g. "12:34")
    tqdm_elapsed: str = ""  # elapsed time from tqdm bar
    tqdm_eta: str = ""  # remaining time from tqdm bar

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
    # sacct for the last 7 days, only our jobs
    out = _run(
        "sacct --me --starttime=$(date -d '7 days ago' +%Y-%m-%d) "
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


def _get_active_job() -> tuple[str, str, str] | None:
    """Return (job_id, job_name, elapsed) of the currently running job, or None."""
    out = _run('squeue --me --noheader --format="%i %j %T %M"')
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[2] in ("RUNNING", "PENDING"):
            return parts[0], parts[1], parts[3]
        elif len(parts) >= 3 and parts[2] in ("RUNNING", "PENDING"):
            return parts[0], parts[1], ""
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
            # Parse tqdm elapsed/ETA
            mt = _TQDM_TIME.search(line)
            if mt:
                job.tqdm_elapsed = mt.group(1)
                job.tqdm_eta = mt.group(2)
            break


def _parse_eval_log(log_path: Path, job: JobInfo) -> None:
    """Parse an eval log file and update job state."""
    tail = _tail_lines(log_path, n=200)

    eval_stages_seen = (
        0  # count how many "Evaluating: Stage" lines we've seen
    )
    is_baseline = False

    for line in tail:
        m = _EVAL_CHECKPOINT.search(line)
        if m:
            job.eval_label = m.group(1)
            # Extract stage number if present
            sm = _EVAL_STAGE_NUM.search(job.eval_label)
            if sm:
                job.stage = int(sm.group(1))
                eval_stages_seen += 1
                is_baseline = False  # back to stage eval
            elif "baseline" in job.eval_label.lower():
                is_baseline = True
                job.stage = 0

        # Detect "Running baseline evaluation..." (not an "Evaluating:" line)
        if "baseline evaluation" in line.lower():
            is_baseline = True
            job.stage = 0

        m = _EVAL_PASS.search(line)
        if m:
            job.eval_label = m.group(1)
            job.eval_pass = m.group(2)
            if "baseline" in job.eval_label.lower():
                is_baseline = True
                job.stage = 0

        if _EVAL_COMPLETE.search(line):
            job.eval_label = "COMPLETE"

    # Count total stages from curriculum log marker
    stage_count_lines = _grep_lines(
        log_path,
        r"\[curriculum\] Found [0-9]+ stages",
        max_count=1,
    )
    if stage_count_lines:
        m = re.search(r"Found (\d+) stages", stage_count_lines[0])
        if m:
            job.stage_total = int(m.group(1))
    elif eval_stages_seen > 0:
        # Fallback: use max stage number seen
        job.stage_total = max(eval_stages_seen, job.stage)
    if is_baseline:
        job.stage_name = "baseline"

    # Get tqdm generation progress from the last lines
    for line in reversed(tail):
        m = _TQDM_GENERATING.search(line)
        if m:
            job.step = int(m.group(1))
            job.eval_step_total = int(m.group(2))
            # Parse tqdm elapsed/ETA
            mt = _TQDM_TIME.search(line)
            if mt:
                job.tqdm_elapsed = mt.group(1)
                job.tqdm_eta = mt.group(2)
            break


# ── Build full pipeline view ──────────────────────────────────────────────────
def _build_pipeline() -> list[JobInfo]:
    """Reconstruct the full pipeline from chain log + chain file + sacct."""
    slurm_jobs = _get_slurm_jobs()
    active = _get_active_job()
    pending = _read_pending_chain()

    # Only parse chain log if pipeline is active (watcher running or jobs pending)
    has_pipeline = CHAIN_PID_FILE.exists() or (
        CHAIN_FILE.exists() and pending
    )
    submitted_names = _parse_chain_log() if has_pipeline else []

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
            elif state in (
                "FAILED",
                "NODE_FAIL",
                "OUT_OF_MEMORY",
                "TIMEOUT",
                "CANCELLED",
            ):
                job.state = "FAILED"
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
            job.elapsed = active[2]
            # Parse log even when sacct didn't find it
            log_file = _find_log_file(job_type, active[0])
            if log_file:
                if job_type == "train":
                    _parse_training_log(log_file, job)
                else:
                    _parse_eval_log(log_file, job)

        jobs.append(job)

    # 1b. Infer COMPLETED for submitted jobs not found in sacct.
    # The chain only advances on success, so:
    #   - If a later job for the same model (tag) has a non-PENDING state,
    #     earlier same-tag jobs must have completed.
    #   - If any job later in the pipeline (by position) has a non-PENDING
    #     state, ALL earlier jobs must have completed (the chain is serial).
    # Find the last non-PENDING job index in pipeline order.
    last_active_idx = -1
    for i, job in enumerate(jobs):
        if job.state != "PENDING":
            last_active_idx = i
    for i, job in enumerate(jobs):
        if job.state == "PENDING" and i < last_active_idx:
            job.state = "COMPLETED"

    # 1c. Mark RUNNING jobs with no progress yet as STARTING
    for job in jobs:
        if job.state == "RUNNING":
            has_progress = (
                job.stage > 0 or job.step > 0 or job.eval_label
            )
            if not has_progress:
                job.state = "STARTING"

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
            elif state in (
                "FAILED",
                "NODE_FAIL",
                "OUT_OF_MEMORY",
                "TIMEOUT",
                "CANCELLED",
            ):
                job.state = "FAILED"

            log_file = _find_log_file(job_type, sid)
            if log_file:
                if job_type == "train":
                    _parse_training_log(log_file, job)
                else:
                    _parse_eval_log(log_file, job)
        elif active and active[1] == name:
            job.slurm_id = active[0]
            job.state = "RUNNING"
            job.elapsed = active[2]
            log_file = _find_log_file(job_type, active[0])
            if log_file:
                if job_type == "train":
                    _parse_training_log(log_file, job)
                else:
                    _parse_eval_log(log_file, job)

        jobs.append(job)

    # 3. Standalone mode — no pipeline files found.
    #    Discover jobs from squeue/sacct matching train/eval names.
    if not jobs:

        def _parse_job_name(name: str) -> tuple[str, str] | None:
            """Extract (job_type, tag) from a SLURM job name.

            Supports: 'train-smollm2-135m', 'eval-tinyllama-11b',
            'train', 'eval' (no tag).
            """
            parts = name.split("-", 1)
            if parts[0] in ("train", "eval"):
                tag = parts[1] if len(parts) == 2 else ""
                return parts[0], tag
            return None

        for name, (sid, state, exit_code) in sorted(
            slurm_jobs.items(), key=lambda x: x[1][0]
        ):
            parsed = _parse_job_name(name)
            if not parsed:
                continue
            job_type, tag = parsed
            job = JobInfo(job_type=job_type, config="", tag=tag)
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
            elif state in (
                "FAILED",
                "NODE_FAIL",
                "OUT_OF_MEMORY",
                "TIMEOUT",
                "CANCELLED",
            ):
                job.state = "FAILED"
            else:
                job.state = state

            log_file = _find_log_file(job_type, sid)
            if log_file:
                if job_type == "train":
                    _parse_training_log(log_file, job)
                else:
                    _parse_eval_log(log_file, job)
            jobs.append(job)

        # Also check squeue for a running job not yet in sacct
        if active:
            a_name = active[1]
            existing_names = set()
            for j in jobs:
                existing_names.add(
                    f"{j.job_type}-{j.tag}" if j.tag else j.job_type
                )
            full_name = a_name  # e.g. "eval" or "train-smollm2-135m"
            if full_name not in existing_names:
                parsed = _parse_job_name(a_name)
                if parsed:
                    job_type, tag = parsed
                    job = JobInfo(
                        job_type=job_type,
                        config="",
                        tag=tag,
                        state="RUNNING",
                        slurm_id=active[0],
                        elapsed=active[2],
                    )
                    log_file = _find_log_file(job.job_type, active[0])
                    if log_file:
                        if job.job_type == "train":
                            _parse_training_log(log_file, job)
                        else:
                            _parse_eval_log(log_file, job)
                    jobs.append(job)

        # Mark RUNNING jobs with no progress as STARTING
        for job in jobs:
            if job.state == "RUNNING":
                has_progress = (
                    job.stage > 0 or job.step > 0 or job.eval_label
                )
                if not has_progress:
                    job.state = "STARTING"

    return jobs


# ── Time helpers ──────────────────────────────────────────────────────────────
def _parse_elapsed_seconds(elapsed: str) -> int | None:
    """Parse squeue elapsed time (e.g. '12:34', '1:23:45', '1-02:03:04') to seconds."""
    if not elapsed:
        return None
    try:
        parts = elapsed.split("-")
        if len(parts) == 2:
            days = int(parts[0])
            rest = parts[1]
        else:
            days = 0
            rest = parts[0]
        t = rest.split(":")
        if len(t) == 3:
            return (
                days * 86400
                + int(t[0]) * 3600
                + int(t[1]) * 60
                + int(t[2])
            )
        elif len(t) == 2:
            return days * 86400 + int(t[0]) * 60 + int(t[1])
        return None
    except (ValueError, IndexError):
        return None


def _format_duration(seconds: int) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _estimate_eta(job: JobInfo) -> str:
    """Return tqdm ETA if available, otherwise estimate from elapsed."""
    if job.tqdm_eta:
        return job.tqdm_eta

    elapsed_s = _parse_elapsed_seconds(job.elapsed)
    if not elapsed_s or elapsed_s < 30:
        return ""

    if (
        job.job_type == "train"
        and job.step > 0
        and job.stage_total > 0
    ):
        remaining_steps = job.stage_total - job.step
        if remaining_steps <= 0:
            return ""
        secs_per_step = elapsed_s / job.step
        eta = int(secs_per_step * remaining_steps)
        return _format_duration(eta)
    elif (
        job.job_type == "eval"
        and job.step > 0
        and job.eval_step_total > 0
    ):
        remaining = job.eval_step_total - job.step
        if remaining <= 0:
            return ""
        secs_per_step = elapsed_s / job.step
        eta = int(secs_per_step * remaining)
        return _format_duration(eta)
    return ""


def _estimate_total_eta(job: JobInfo) -> str:
    """Estimate total remaining time including future stages.

    For train: uses step speed x remaining steps in all stages.
    For eval: uses batch speed x remaining batches in all stages.
    """
    # Use tqdm timing if available, fallback to squeue elapsed
    speed_elapsed_s = _parse_elapsed_seconds(job.tqdm_elapsed)
    speed_steps = job.step
    if not speed_elapsed_s or speed_elapsed_s < 10:
        # Fallback: use squeue elapsed (less accurate but works for train)
        speed_elapsed_s = _parse_elapsed_seconds(job.elapsed)
        if not speed_elapsed_s or speed_elapsed_s < 30:
            return ""

    if job.job_type == "train" and speed_steps > 0 and job.stage > 0:
        # At last stage, total == stage ETA, skip
        if job.stage >= 3:
            return ""
        secs_per_step = speed_elapsed_s / speed_steps
        # Remaining in current stage
        remaining_current = max(0, job.stage_total - job.step)
        # Future stages: read from log
        future_steps = 0
        if job.slurm_id:
            log_file = _find_log_file(job.job_type, job.slurm_id)
            if log_file:
                stage_lines = _grep_lines(
                    log_file, r"\[stage [0-9]+\] steps="
                )
                all_stages: dict[int, int] = {}
                for line in stage_lines:
                    m = _STAGE_START.search(line)
                    if m:
                        all_stages[int(m.group(1))] = int(m.group(2))
                for s_num, s_steps in all_stages.items():
                    if s_num > job.stage:
                        future_steps += s_steps
        # If we couldn't read future stages, estimate ~same as current
        if future_steps == 0 and job.stage < 3:
            future_steps = job.stage_total * (3 - job.stage)
        total_remaining = remaining_current + future_steps
        if total_remaining <= 0:
            return ""
        eta = int(secs_per_step * total_remaining)
        return _format_duration(eta)

    elif (
        job.job_type == "eval"
        and speed_steps > 0
        and job.stage > 0
        and job.eval_step_total > 0
    ):
        secs_per_batch = speed_elapsed_s / speed_steps
        remaining_current = max(0, job.eval_step_total - job.step)
        # Each stage has same number of batches (same dataset size)
        total_stages = job.stage_total if job.stage_total > 0 else 3
        remaining_stages = total_stages - job.stage
        # +1 round for baseline evaluation
        baseline_batches = job.eval_step_total
        future_batches = (
            remaining_stages * job.eval_step_total + baseline_batches
        )
        # At last stage with no future stages, only baseline remains
        if remaining_stages <= 0 and remaining_current <= 0:
            future_batches = baseline_batches
        total_remaining = remaining_current + future_batches
        if total_remaining <= 0:
            return ""
        eta = int(secs_per_batch * total_remaining)
        return _format_duration(eta)

    return ""


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


# ── Display ───────────────────────────────────────────────────────────────────
_STATE_ICONS = {
    "COMPLETED": f"{_GREEN}✓{_RST}",
    "FAILED": f"{_RED}✗{_RST}",
    "RUNNING": f"{_CYAN}▶{_RST}",
    "STARTING": f"{_YELLOW}»{_RST}",
    "PENDING": f"{_GRAY}·{_RST}",
}

_STATE_COLORS = {
    "COMPLETED": _GREEN,
    "FAILED": _RED,
    "RUNNING": _CYAN,
    "STARTING": _YELLOW,
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
    if job.state in ("RUNNING", "STARTING"):
        pass  # no detail in table row — shown in footer
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

    # Pad with visible-width awareness (ANSI codes don't count)
    def _vpad(s: str, width: int) -> str:
        visible = len(re.sub(r"\033\[[0-9;]*m", "", s))
        return s + " " * max(0, width - visible)

    if job.tag:
        label = f"{tc}{job.job_type}{_RST}-{_BOLD}{job.tag}{_RST}"
    else:
        label = f"{tc}{job.job_type}{_RST}"
    state_str = f"{sc}{job.state}{_RST}"

    return f" {icon}  {_vpad(label, 30)} {_vpad(slurm, 12)} {_vpad(state_str, 12)}{detail}"


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


def _display(jobs: list[JobInfo], show_table: bool = True) -> None:
    """Print the full pipeline status."""
    completed = sum(1 for j in jobs if j.state == "COMPLETED")
    failed = sum(1 for j in jobs if j.state == "FAILED")
    total = len(jobs)

    # Detect standalone mode — pipeline is active only if watcher PID or pending jobs
    is_pipeline = CHAIN_PID_FILE.exists() or (
        CHAIN_FILE.exists() and CHAIN_FILE.stat().st_size > 0
    )

    # Build summary badges
    done_badge = f"{_GREEN}{completed}{_RST}/{total} done"
    fail_badge = f"  {_RED}{failed} failed{_RST}" if failed else ""

    # Move cursor home + clear from cursor to end of screen
    # (avoids full clear → no flicker)
    print("\033[H\033[J", end="")
    print(f"{_CYAN}{'═' * 65}{_RST}")
    if is_pipeline:
        print(
            f"  {_BOLD}{_CYAN}GRPO Pipeline Monitor{_RST} — {done_badge}{fail_badge}"
        )
        print(f"  {_watcher_status()}")
    elif total > 0:
        print(
            f"  {_BOLD}{_CYAN}GRPO Job Monitor{_RST} — {done_badge}{fail_badge}"
        )
        print(f"  {_DIM}Standalone mode (no pipeline){_RST}")
    else:
        print(
            f"  {_BOLD}{_CYAN}GRPO Job Monitor{_RST} — no jobs found"
        )
        print(
            f"  {_DIM}Waiting for jobs matching train-*/eval-*...{_RST}"
        )
    print(f"  {_DIM}{time.strftime('%Y-%m-%d %H:%M:%S')}{_RST}")
    print(f"{_CYAN}{'═' * 65}{_RST}")

    if show_table:
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

    print()
    remaining = sum(1 for j in jobs if j.state == "PENDING")
    running = [j for j in jobs if j.state in ("RUNNING", "STARTING")]
    if running:
        j = running[0]
        tc = _TYPE_COLORS.get(j.job_type, "")
        bar_color = _CYAN if j.job_type == "train" else _MAGENTA

        # Build description line
        if j.job_type == "train" and j.stage > 0:
            desc = f"stage {_WHITE}{j.stage}/3{_RST}, step {_WHITE}{j.step}{_RST}/{j.stage_total}"
        elif j.job_type == "eval" and j.eval_label:
            if j.stage > 0:
                stg = f"stage {j.stage}"
                if j.stage_total > 0:
                    stg += f"/{j.stage_total}"
                desc = f"{stg}"
            elif j.stage_name == "baseline":
                desc = "baseline"
            else:
                desc = j.eval_label
            if j.step > 0 and j.eval_step_total > 0:
                desc += f", batch {_WHITE}{j.step}{_RST}/{j.eval_step_total}"
        else:
            desc = ""

        job_label = (
            f"{tc}{j.job_type}{_RST}-{_BOLD}{j.tag}{_RST}"
            if j.tag
            else f"{tc}{j.job_type}{_RST}"
        )
        print(
            f"  {_CYAN}▶ Active:{_RST} {job_label}"
            + (f" {_DIM}[{j.slurm_id}]{_RST}" if j.slurm_id else "")
            + (f" — {desc}" if desc else "")
        )

        # Progress bar + time on second line
        cur = j.step
        tot = (
            j.stage_total
            if j.job_type == "train"
            else j.eval_step_total
        )
        if cur > 0 and tot > 0:
            pct = int(cur / tot * 100)
            bar_w = 20
            filled = int(bar_w * pct / 100)
            bar = f"{bar_color}{'█' * filled}{_GRAY}{'░' * (bar_w - filled)}{_RST}"
            eta = _estimate_eta(j)
            total_eta = _estimate_total_eta(j)
            time_parts = ""
            if j.elapsed:
                time_parts += f" ⏱  {_DIM}{j.elapsed}{_RST}"
            if eta:
                time_parts += f" ⏳ {_DIM}~{eta}{_RST}"
            if total_eta and total_eta != eta:
                time_parts += f" {_DIM}(job ~{total_eta}){_RST}"
            print(f"  {bar} {_WHITE}{pct}%{_RST}{time_parts}")
    elif remaining > 0:
        # Check if watcher is alive
        watcher_alive = False
        if CHAIN_PID_FILE.exists():
            try:
                pid = CHAIN_PID_FILE.read_text().strip()
                result = _run(f"ps -p {pid} -o pid= 2>/dev/null")
                watcher_alive = bool(result)
            except Exception:
                pass
        if watcher_alive:
            print(
                f"  {_YELLOW}⏳ Waiting for next job... ({remaining} remaining){_RST}"
            )
        else:
            print(
                f"  {_RED}⚠ Pipeline stalled{_RST} — {remaining} jobs pending but watcher is dead"
            )
            print(
                f"  {_DIM}Restart: bash cluster/run_all.sh   |   Clean: rm .job_chain{_RST}"
            )
    elif not jobs:
        print(f"  {_DIM}No jobs found.{_RST}")
    else:
        print(f"  {_GREEN}{_BOLD}✓ Pipeline finished!{_RST}")

        # Show summary table of eval results
        eval_jobs = [j for j in jobs if j.job_type == "eval"]
        if eval_jobs:
            print()
            print(
                f"  {_BOLD}{'Model':<25s} {'Status':<12s} {'Pass@1'}{_RST}"
            )
            print(f"  {'─' * 50}")
            for ej in eval_jobs:
                sc = _STATE_COLORS.get(ej.state, "")
                status = f"{sc}{ej.state}{_RST}"
                p1 = ej.eval_pass if ej.eval_pass else "-"
                if ej.state == "COMPLETED" and ej.eval_pass:
                    p1 = f"{_GREEN}{_BOLD}{ej.eval_pass}{_RST}"
                elif ej.state == "FAILED":
                    p1 = f"{_RED}—{_RST}"
                print(f"  {ej.tag:<25s} {status:<22s} {p1}")

    print()


# ── Live follow mode ──────────────────────────────────────────────────────────
def _tail_active_job(jobs: list[JobInfo]) -> str | None:
    """Find and return the log path of the currently active job."""
    for job in jobs:
        if job.state in ("RUNNING", "STARTING") and job.slurm_id:
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
    parser.add_argument(
        "--tab",
        action="store_true",
        help="Show the full job table (default: compact view)",
    )
    args = parser.parse_args()

    print("GRPO Monitor — Ctrl+C to exit")
    print(f"Polling every {args.poll}s...")
    print()

    try:
        while True:
            jobs = _build_pipeline()
            _display(jobs, show_table=args.tab)

            # Check if pipeline/job is done
            is_pipeline = CHAIN_PID_FILE.exists() or (
                CHAIN_FILE.exists() and CHAIN_FILE.stat().st_size > 0
            )
            all_done = jobs and all(
                j.state in ("COMPLETED", "FAILED") for j in jobs
            )
            watcher_alive = CHAIN_PID_FILE.exists()
            no_pending = (
                not CHAIN_FILE.exists() or not _read_pending_chain()
            )

            if (
                is_pipeline
                and all_done
                and not watcher_alive
                and no_pending
            ):
                print("Pipeline complete. Exiting.")
                break
            elif not is_pipeline and all_done:
                print("Job complete. Exiting.")
                break

            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("\n💀 Monitor stopped.")


if __name__ == "__main__":
    main()
