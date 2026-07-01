"""Lightweight in-memory sliding-window rate limiter.

Protects the public demo's shared LLM quota: caps requests per session and
globally. In-process only (fine for a single-container deployment); swap for
Redis if you ever scale horizontally.
"""
import threading
import time
from collections import defaultdict, deque


class SlidingWindowRateLimiter:
    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str = "global") -> bool:
        """Record a hit and return True if under the limit, False if exceeded."""
        now = time.monotonic()
        with self._lock:
            dq = self._hits[key]
            cutoff = now - self.window
            while dq and dq[0] <= cutoff:
                dq.popleft()
            if len(dq) >= self.max_requests:
                return False
            dq.append(now)
            return True
