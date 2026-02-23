import time

_LAST_TS_MS = -1

def get_monotonic_ts() -> int:
    """
    Returns a monotonically increasing millisecond timestamp.
    Ensures that subsequent calls always return a value at least 1ms 
    greater than the previous call, even if the system clock hasn't moved.
    """
    global _LAST_TS_MS
    current_ts = int(time.perf_counter() * 1000)
    
    if current_ts <= _LAST_TS_MS:
        current_ts = _LAST_TS_MS + 1
        
    _LAST_TS_MS = current_ts
    return current_ts