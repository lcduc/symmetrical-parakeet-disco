import time
from collections import defaultdict
from threading import Lock


class InMemoryRateLimiter:
    """Per-key sliding-window rate limiter, held in process memory.

    Appropriate for this project's single-process deployment (no external
    broker/queue, see README "Why no job queue"); a multi-instance deployment
    would need a shared store (e.g. Redis) instead.
    """

    def __init__(self, limit_per_minute: int):
        self._limit = limit_per_minute
        self._window_seconds = 60.0
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        """Record a hit for `key` and return whether it's within the per-minute limit."""
        now = time.monotonic()
        cutoff = now - self._window_seconds
        with self._lock:
            recent_hits = [hit for hit in self._hits[key] if hit > cutoff]
            if len(recent_hits) >= self._limit:
                self._hits[key] = recent_hits
                return False
            recent_hits.append(now)
            self._hits[key] = recent_hits
            return True
