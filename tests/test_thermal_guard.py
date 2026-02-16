import importlib


def test_check_and_cool_triggers_sleep():
    """Simulate high usage/temp then recovery; ensure check_and_cool sleeps."""
    tg = importlib.import_module('utils.thermal_guard')

    # Fake time object to avoid real sleeping and to advance time deterministically
    class FakeTime:
        def __init__(self):
            self._t = 0.0

        def time(self):
            return self._t

        def sleep(self, d):
            # advance internal time instead of real sleep
            self._t += d

    fake_time = FakeTime()

    # Usage and temp sequences: first call returns high (trigger), next call returns safe values
    usage_vals = iter([95.0, 40.0])
    temp_vals = iter([100.0, 45.0])

    def mock_usage(interval=1.0):
        try:
            return next(usage_vals)
        except StopIteration:
            return 40.0

    def mock_temp():
        try:
            return next(temp_vals)
        except StopIteration:
            return 45.0

    # Patch the module functions/objects
    orig_time = tg.time
    orig_usage = tg.read_cpu_usage
    orig_temp = tg.read_cpu_temp

    tg.time = fake_time
    tg.read_cpu_usage = mock_usage
    tg.read_cpu_temp = mock_temp

    try:
        slept = tg.check_and_cool(threshold_usage=85.0, threshold_temp=92.0, check_interval=0.1, max_wait=5.0, verbose=False)
        assert slept >= 0.1, f"Expected some cooling sleep, got {slept}"
    finally:
        # restore
        tg.time = orig_time
        tg.read_cpu_usage = orig_usage
        tg.read_cpu_temp = orig_temp


def test_check_and_cool_no_sleep():
    """When sensors are within limits, check_and_cool should return 0."""
    tg = importlib.import_module('utils.thermal_guard')

    class FakeTime:
        def __init__(self):
            self._t = 0.0

        def time(self):
            return self._t

        def sleep(self, d):
            self._t += d

    fake_time = FakeTime()

    def mock_usage(interval=1.0):
        return 10.0

    def mock_temp():
        return 40.0

    orig_time = tg.time
    orig_usage = tg.read_cpu_usage
    orig_temp = tg.read_cpu_temp

    tg.time = fake_time
    tg.read_cpu_usage = mock_usage
    tg.read_cpu_temp = mock_temp

    try:
        slept = tg.check_and_cool(threshold_usage=85.0, threshold_temp=92.0, check_interval=0.1, max_wait=1.0, verbose=False)
        assert slept == 0.0
    finally:
        tg.time = orig_time
        tg.read_cpu_usage = orig_usage
        tg.read_cpu_temp = orig_temp
