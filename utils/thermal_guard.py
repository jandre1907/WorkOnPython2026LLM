"""thermal_guard.py

Small helper to monitor CPU temperature and usage and inject sleep
when thresholds are exceeded to allow cooling.

API:
  check_and_cool(threshold_usage=85, threshold_temp=92, check_interval=2, max_wait=300)
    - returns total_sleep_time (seconds) if it slept, 0 otherwise

This uses psutil where available and falls back to reading
/sys/class/thermal/thermal_zone*/temp on Linux.
"""
from __future__ import annotations
import time
import glob
import os
from typing import Optional
import json
import re
import urllib.request

try:
    import psutil
except Exception:
    psutil = None


def _read_thermal_zones() -> Optional[float]:
    """Attempt to read CPU temperature (Celsius) from thermal zones.
    Returns the max temperature found, or None if unavailable."""
    temps = []
    for path in glob.glob('/sys/class/thermal/thermal_zone*/temp'):
        try:
            with open(path, 'r') as f:
                v = f.read().strip()
            if not v:
                continue
            # Some zones report millidegrees
            t = float(v) / 1000.0 if len(v) > 3 else float(v)
            temps.append(t)
        except Exception:
            continue
    if not temps:
        return None
    return max(temps)


def read_cpu_temp() -> Optional[float]:
    """Return CPU temperature in Celsius, or None if not available."""
    # Try a local HTTP API first if provided
    # Try a list of candidate API endpoints (env var first)
    env_url = os.environ.get('THERMAL_API_URL')
    candidates = []
    if env_url:
        candidates.append(env_url)
    candidates.extend([
        'http://127.0.0.1:8085/data.json',
        'http://localhost:8085/data.json',
        'http://[::1]:8085/data.json',
    ])
    for api_url in candidates:
        try:
            with urllib.request.urlopen(api_url, timeout=2.0) as resp:
                raw = resp.read()
                data = json.loads(raw.decode('utf-8', errors='ignore'))
            # scan JSON for temperature entries
            temps = []

            def _scan(node):
                if isinstance(node, dict):
                    # common pattern: Type == 'Temperature' and 'Value' contains 'xx,0 \u00B0C' or 'xx.x °C'
                    t = None
                    if node.get('Type') == 'Temperature' and 'Value' in node:
                        t = node.get('Value')
                    elif 'Text' in node and 'Temperature' in node.get('Text', '') and 'Value' in node:
                        t = node.get('Value')
                    if t:
                        # extract numeric part
                        m = re.search(r"([0-9]+[.,]?[0-9]*)", str(t))
                        if m:
                            val = float(m.group(1).replace(',', '.'))
                            temps.append(val)
                    for v in node.values():
                        _scan(v)
                elif isinstance(node, list):
                    for item in node:
                        _scan(item)

            _scan(data)
            if temps:
                return max(temps)
        except Exception:
            # try next candidate
            continue

    # Try psutil sensors first
    if psutil is not None:
        try:
            st = psutil.sensors_temperatures()
            if st:
                # Look for common keys
                for key in ('coretemp', 'cpu-thermal', 'acpitz'):
                    if key in st and st[key]:
                        # pick highest current reading
                        vals = [x.current for x in st[key] if hasattr(x, 'current')]
                        if vals:
                            return max(vals)
                # fallback: take any temperature entries
                for entries in st.values():
                    vals = [x.current for x in entries if hasattr(x, 'current')]
                    if vals:
                        return max(vals)
        except Exception:
            pass

    # Fallback to thermal zones
    try:
        return _read_thermal_zones()
    except Exception:
        return None


def read_cpu_usage(interval: float = 1.0) -> float:
    """Return CPU usage percentage over given interval."""
    if psutil is not None:
        try:
            return psutil.cpu_percent(interval=interval)
        except Exception:
            pass
    # Fallback: approximate using /proc/stat (very simple)
    try:
        def _read_proc():
            with open('/proc/stat', 'r') as f:
                for line in f:
                    if line.startswith('cpu '):
                        parts = line.split()
                        vals = [float(x) for x in parts[1:]]
                        total = sum(vals)
                        idle = vals[3]
                        return total, idle
            return None

        t1 = _read_proc()
        time.sleep(interval)
        t2 = _read_proc()
        if t1 and t2:
            total_diff = t2[0] - t1[0]
            idle_diff = t2[1] - t1[1]
            usage = 100.0 * (1.0 - idle_diff / total_diff) if total_diff > 0 else 0.0
            return usage
    except Exception:
        pass
    return 0.0


def check_and_cool(threshold_usage: float = 85.0,
                   threshold_temp: float = 92.0,
                   check_interval: float = 2.0,
                   max_wait: float = 300.0,
                   verbose: bool = False) -> float:
    """Check CPU usage and temperature, and sleep to cool if thresholds exceeded.

    Behavior:
      - Samples CPU usage and temperature.
      - If either usage > threshold_usage or temp > threshold_temp, enters a cooling loop.
      - In the loop it sleeps `check_interval` seconds and re-checks until both are below
        thresholds or `max_wait` is reached.

    Returns total sleep time in seconds (0 if no cooling was needed).
    """
    start = time.time()
    # first measurement
    usage = read_cpu_usage(interval=0.5)
    temp = read_cpu_temp()
    if verbose:
        print(f"[thermal_guard] usage={usage:.1f}% temp={temp if temp is not None else 'N/A'}°C")

    need_cool = False
    if usage is not None and usage > threshold_usage:
        need_cool = True
    if temp is not None and temp > threshold_temp:
        need_cool = True

    total_sleep = 0.0
    if not need_cool:
        return 0.0

    if verbose:
        print(f"[thermal_guard] Threshold exceeded (usage>{threshold_usage}% or temp>{threshold_temp}°C). Cooling...")

    # cooling loop
    while True:
        if time.time() - start >= max_wait:
            if verbose:
                print(f"[thermal_guard] max_wait reached ({max_wait}s). Stopping cooling.")
            break
        time.sleep(check_interval)
        total_sleep += check_interval
        usage = read_cpu_usage(interval=0.1)
        temp = read_cpu_temp()
        if verbose:
            print(f"[thermal_guard] after sleep usage={usage:.1f}% temp={temp if temp is not None else 'N/A'}°C")
        below_usage = (usage <= threshold_usage)
        below_temp = (temp is None) or (temp <= threshold_temp)
        if below_usage and below_temp:
            if verbose:
                print(f"[thermal_guard] Cooling successful after {total_sleep:.1f}s")
            break

    return total_sleep
