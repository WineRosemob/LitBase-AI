from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any

from litbase_ai.utils.logging import get_logger


logger = get_logger(__name__)


try:  # pragma: no cover
    from rich.console import Console
    from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
except Exception:  # pragma: no cover
    Console = None
    Progress = None
    BarColumn = None
    TaskProgressColumn = None
    TextColumn = None
    TimeElapsedColumn = None

try:  # pragma: no cover
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


@dataclass
class _PlainTask:
    description: str
    total: int | None
    completed: int = 0


class _StageContext:
    def __init__(self, manager: "ProgressManager", name: str, message: str | None = None):
        self.manager = manager
        self.name = name
        self.message = message
        self.start: float | None = None

    def __enter__(self):
        self.start = time.perf_counter()
        self.manager.log(f"{self.name} - started" + (f" | {self.message}" if self.message else ""))
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.perf_counter() - (self.start or time.perf_counter())
        if exc is None:
            self.manager.log(f"{self.name} - completed in {elapsed:.2f}s")
        else:
            self.manager.log(f"{self.name} - failed in {elapsed:.2f}s: {exc}", level="error")
        return False


class ProgressManager:
    """Unified progress manager with rich/tqdm/plain fallbacks."""

    def __init__(
        self,
        enabled: bool = True,
        use_rich: bool = True,
        style: str = "rich",
        verbose: bool = False,
        quiet: bool = False,
    ):
        self.enabled = enabled and not quiet
        self.verbose = verbose
        self.quiet = quiet
        self.style = style
        if style == "rich" and not use_rich:
            self.style = "plain"
        self._task_counter = 0
        self._plain_tasks: dict[int, _PlainTask] = {}
        self._tqdm_tasks: dict[int, Any] = {}
        self._rich_task_ids: dict[int, int] = {}
        self._lock = threading.RLock()

        self.console = Console() if (Console and self.style == "rich") else None
        self._rich_progress = None
        if self.enabled and self.style == "rich" and Progress is not None:
            try:
                self._rich_progress = Progress(
                    TextColumn("{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeElapsedColumn(),
                    console=self.console,
                    transient=False,
                )
                self._rich_progress.start()
            except Exception:  # pragma: no cover
                self._rich_progress = None
                self.style = "tqdm" if tqdm is not None else "plain"
        elif self.style == "tqdm" and tqdm is None:
            self.style = "plain"

    def stage(self, name: str, message: str | None = None):
        """Create a timed stage context manager."""
        return _StageContext(self, name=name, message=message)

    def task(self, description: str, total: int | None = None):
        """Create a progress task and return its identifier."""
        with self._lock:
            self._task_counter += 1
            task_id = self._task_counter
            if not self.enabled:
                self._plain_tasks[task_id] = _PlainTask(description=description, total=total)
                return task_id

            if self.style == "rich" and self._rich_progress is not None:
                rid = self._rich_progress.add_task(description=description, total=total)
                self._rich_task_ids[task_id] = rid
                self._plain_tasks[task_id] = _PlainTask(description=description, total=total)
                return task_id

            if self.style == "tqdm" and tqdm is not None:
                bar = tqdm(total=total, desc=description, leave=False)
                self._tqdm_tasks[task_id] = bar
                self._plain_tasks[task_id] = _PlainTask(description=description, total=total)
                return task_id

            self._plain_tasks[task_id] = _PlainTask(description=description, total=total)
            self.log(f"{description} started (total={total if total is not None else 'unknown'})")
            return task_id

    def update(self, task_id, advance: int = 1, description: str | None = None):
        """Update progress for a task."""
        with self._lock:
            if task_id not in self._plain_tasks:
                return
            plain = self._plain_tasks[task_id]
            plain.completed += advance
            if description:
                plain.description = description

            if not self.enabled:
                return

            if self.style == "rich" and self._rich_progress is not None:
                rid = self._rich_task_ids.get(task_id)
                if isinstance(rid, int) and rid in self._rich_progress.task_ids:
                    self._rich_progress.update(rid, advance=advance, description=description)
                return

            if self.style == "tqdm" and task_id in self._tqdm_tasks:
                bar = self._tqdm_tasks[task_id]
                if description:
                    bar.set_description_str(description)
                bar.update(advance)
                return

            total = plain.total or "?"
            self.log(f"{plain.description}: {plain.completed}/{total}")

    def log(self, message: str, level: str = "info"):
        """Write message through logger and optional rich console."""
        with self._lock:
            fn = getattr(logger, level.lower(), logger.info)
            fn(message)
            if self.enabled and self.style == "rich" and self.console is not None and level.lower() == "info":
                if self.verbose:
                    self.console.print(message)

    def summary(self, title: str, data: dict):
        """Print a compact summary block."""
        text = json.dumps(data, ensure_ascii=False, indent=2)
        self.log(f"{title}\n{text}")

    def close(self):
        """Close active progress resources."""
        with self._lock:
            if self._rich_progress is not None:
                try:
                    self._rich_progress.stop()
                except Exception:  # pragma: no cover
                    pass
            for bar in self._tqdm_tasks.values():
                try:
                    bar.close()
                except Exception:  # pragma: no cover
                    pass
            self._tqdm_tasks.clear()
