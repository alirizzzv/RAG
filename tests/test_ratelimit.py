"""Sliding-window rate limiter."""
import time

from app.ratelimit import SlidingWindowRateLimiter


def test_allows_up_to_limit_then_blocks():
    rl = SlidingWindowRateLimiter(max_requests=3, window_seconds=60)
    assert [rl.allow("s1") for _ in range(4)] == [True, True, True, False]


def test_keys_are_independent():
    rl = SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
    assert rl.allow("a") is True
    assert rl.allow("b") is True     # different key, own budget
    assert rl.allow("a") is False    # first key exhausted


def test_window_resets():
    rl = SlidingWindowRateLimiter(max_requests=1, window_seconds=0.3)
    assert rl.allow("s") is True
    assert rl.allow("s") is False
    time.sleep(0.35)
    assert rl.allow("s") is True     # window elapsed
