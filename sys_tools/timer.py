# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


import time
import datetime
import pytest


class Timer(object):
    def __init__(self):
        self.reset()

    @property
    def average_time(self):
        return self.total_time / self.calls if self.calls > 0 else 0.0

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.add(time.time() - self.start_time)
        if average:
            return self.average_time
        else:
            return self.diff

    def add(self, time_diff):
        self.diff = time_diff
        self.total_time += self.diff
        self.calls += 1

    def reset(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0

    def avg_time_str(self):
        time_str = str(datetime.timedelta(seconds=self.average_time))
        return time_str


def get_time_str(time_diff):
    time_str = str(datetime.timedelta(seconds=time_diff))
    return time_str


class cvTimerError(Exception):

    def __init__(self, message):
        self.message = message
        super(cvTimerError, self).__init__(message)


class cvTimer(object):
    """A flexible Timer class.

    :Example:

    >>> import time
    >>> import mmcv
    >>> with mmcv.Timer():
    >>>     # simulate a code block that will run for 1s
    >>>     time.sleep(1)
    1.000
    >>> with mmcv.Timer(print_tmpl='it takes {:.1f} seconds'):
    >>>     # simulate a code block that will run for 1s
    >>>     time.sleep(1)
    it takes 1.0 seconds
    >>> timer = mmcv.Timer()
    >>> time.sleep(0.5)
    >>> print(timer.since_start())
    0.500
    >>> time.sleep(0.5)
    >>> print(timer.since_last_check())
    0.500
    >>> print(timer.since_start())
    1.000
    """

    def __init__(self, start=True, print_tmpl=None):
        self._is_running = False
        self.print_tmpl = print_tmpl if print_tmpl else '{:.3f}'
        if start:
            self.start()

    @property
    def is_running(self):
        """bool: indicate whether the timer is running"""
        return self._is_running

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        print(self.print_tmpl.format(self.since_last_check()))
        self._is_running = False

    def start(self):
        """Start the timer."""
        if not self._is_running:
            self._t_start = time.time()
            self._is_running = True
        self._t_last = time.time()

    def since_start(self):
        """Total time since the timer is started.

        Returns (float): Time in seconds.
        """
        if not self._is_running:
            raise cvTimerError('timer is not running')
        self._t_last = time.time()
        return self._t_last - self._t_start

    def since_last_check(self):
        """Time since the last checking.

        Either :func:`since_start` or :func:`since_last_check` is a checking
        operation.

        Returns (float): Time in seconds.
        """
        if not self._is_running:
            raise cvTimerError('timer is not running')
        dur = time.time() - self._t_last
        self._t_last = time.time()
        return dur


_g_timers = {}  # global timers


def check_cvtime(timer_id):
    """Add check points in a single line.

    This method is suitable for running a task on a list of items. A timer will
    be registered when the method is called for the first time.

    :Example:

    >>> import time
    >>> import mmcv
    >>> for i in range(1, 6):
    >>>     # simulate a code block
    >>>     time.sleep(i)
    >>>     mmcv.check_time('task1')
    2.000
    3.000
    4.000
    5.000

    Args:
        timer_id (str): Timer identifier.
    """
    if timer_id not in _g_timers:
        _g_timers[timer_id] = cvTimer()
        return 0
    else:
        return _g_timers[timer_id].since_last_check()


def test_timer_init():
    timer = cvTimer(start=False)
    assert not timer.is_running
    timer.start()
    assert timer.is_running
    timer = cvTimer()
    assert timer.is_running


def test_timer_run():
    timer = cvTimer()
    time.sleep(1)
    assert abs(timer.since_start() - 1) < 1e-2
    time.sleep(1)
    assert abs(timer.since_last_check() - 1) < 1e-2
    assert abs(timer.since_start() - 2) < 1e-2
    timer = cvTimer(False)
    with pytest.raises(cvTimerError):
        timer.since_start()
    with pytest.raises(cvTimerError):
        timer.since_last_check()


def test_timer_context(capsys):
    with cvTimer():
        time.sleep(1)
    out, _ = capsys.readouterr()
    assert abs(float(out) - 1) < 1e-2
    with cvTimer(print_tmpl='time: {:.1f}s'):
        time.sleep(1)
    out, _ = capsys.readouterr()
    assert out == 'time: 1.0s\n'