import sys
from multiprocessing import Pool
import time
from sys_tools import misc, timer


class ProgressBar(object):
    """A progress bar which can print the progress"""

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width
                          if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        if sys.version_info > (3, 3):
            from shutil import get_terminal_size
        else:
            from backports.shutil_get_terminal_size import get_terminal_size
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal width is too small ({}), please consider '
                  'widen the terminal for better progressbar '
                  'visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write('[{}] 0/{}, elapsed: 0s, ETA:'.format(
                ' ' * self.bar_width, self.task_num))
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.timer = timer.cvTimer()

    def update(self):
        self.completed += 1
        elapsed = self.timer.since_start()
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '>' * mark_width + ' ' * (self.bar_width - mark_width)
            sys.stdout.write(
                '\r[{}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s'.format(
                    bar_chars, self.completed, self.task_num, fps,
                    int(elapsed + 0.5), eta))
        else:
            sys.stdout.write(
                'completed: {}, elapsed: {}s, {:.1f} tasks/s'.format(
                    self.completed, int(elapsed + 0.5), fps))
        sys.stdout.flush()


def track_progress(func, tasks, bar_width=50, **kwargs):
    """Track the progress of tasks execution with a progress bar.

    Tasks are done with a simple for-loop.

    Args:
        func (callable): The function to be applied to each task.
        tasks (list or tuple[Iterable, int]): A list of tasks or
            (tasks, total num).
        bar_width (int): Width of progress bar.

    Returns:
        list: The task results.
    """
    if isinstance(tasks, tuple):
        assert len(tasks) == 2
        assert isinstance(tasks[0], misc.collections_abc.Iterable)
        assert isinstance(tasks[1], int)
        task_num = tasks[1]
        tasks = tasks[0]
    elif isinstance(tasks, misc.collections_abc.Iterable):
        task_num = len(tasks)
    else:
        raise TypeError(
            '"tasks" must be an iterable object or a (iterator, int) tuple')
    prog_bar = ProgressBar(task_num, bar_width)
    results = []
    for task in tasks:
        results.append(func(task, **kwargs))
        prog_bar.update()
    sys.stdout.write('\n')
    return results


def init_pool(process_num, initializer=None, initargs=None):
    if initializer is None:
        return Pool(process_num)
    elif initargs is None:
        return Pool(process_num, initializer)
    else:
        if not isinstance(initargs, tuple):
            raise TypeError('"initargs" must be a tuple')
        return Pool(process_num, initializer, initargs)


def track_parallel_progress(func,
                            tasks,
                            nproc,
                            initializer=None,
                            initargs=None,
                            bar_width=50,
                            chunksize=1,
                            skip_first=False,
                            keep_order=True):
    """Track the progress of parallel task execution with a progress bar.

    The built-in :mod:`multiprocessing` module is used for process pools and
    tasks are done with :func:`Pool.map` or :func:`Pool.imap_unordered`.

    Args:
        func (callable): The function to be applied to each task.
        tasks (list or tuple[Iterable, int]): A list of tasks or
            (tasks, total num).
        nproc (int): Process (worker) number.
        initializer (None or callable): Refer to :class:`multiprocessing.Pool`
            for details.
        initargs (None or tuple): Refer to :class:`multiprocessing.Pool` for
            details.
        chunksize (int): Refer to :class:`multiprocessing.Pool` for details.
        bar_width (int): Width of progress bar.
        skip_first (bool): Whether to skip the first sample for each worker
            when estimating fps, since the initialization step may takes
            longer.
        keep_order (bool): If True, :func:`Pool.imap` is used, otherwise
            :func:`Pool.imap_unordered` is used.

    Returns:
        list: The task results.
    """
    if isinstance(tasks, tuple):
        assert len(tasks) == 2
        assert isinstance(tasks[0], misc.collections_abc.Iterable)
        assert isinstance(tasks[1], int)
        task_num = tasks[1]
        tasks = tasks[0]
    elif isinstance(tasks, misc.collections_abc.Iterable):
        task_num = len(tasks)
    else:
        raise TypeError(
            '"tasks" must be an iterable object or a (iterator, int) tuple')
    pool = init_pool(nproc, initializer, initargs)
    start = not skip_first
    task_num -= nproc * chunksize * int(skip_first)
    prog_bar = ProgressBar(task_num, bar_width, start)
    results = []
    if keep_order:
        gen = pool.imap(func, tasks, chunksize)
    else:
        gen = pool.imap_unordered(func, tasks, chunksize)
    for result in gen:
        results.append(result)
        if skip_first:
            if len(results) < nproc * chunksize:
                continue
            elif len(results) == nproc * chunksize:
                prog_bar.start()
                continue
        prog_bar.update()
    sys.stdout.write('\n')
    pool.close()
    pool.join()
    return results



class TestProgressBar(object):

    def test_start(self, capsys):
        bar_width = 20
        # without total task num
        prog_bar = ProgressBar(bar_width=bar_width)
        out, _ = capsys.readouterr()
        assert out == 'completed: 0, elapsed: 0s'
        prog_bar = ProgressBar(bar_width=bar_width, start=False)
        out, _ = capsys.readouterr()
        assert out == ''
        prog_bar.start()
        out, _ = capsys.readouterr()
        assert out == 'completed: 0, elapsed: 0s'
        # with total task num
        prog_bar = ProgressBar(10, bar_width=bar_width)
        out, _ = capsys.readouterr()
        assert out == '[{}] 0/10, elapsed: 0s, ETA:'.format(' ' * bar_width)
        prog_bar = ProgressBar(10, bar_width=bar_width, start=False)
        out, _ = capsys.readouterr()
        assert out == ''
        prog_bar.start()
        out, _ = capsys.readouterr()
        assert out == '[{}] 0/10, elapsed: 0s, ETA:'.format(' ' * bar_width)

    def test_update(self, capsys):
        bar_width = 20
        # without total task num
        prog_bar = ProgressBar(bar_width=bar_width)
        capsys.readouterr()
        time.sleep(1)
        prog_bar.update()
        out, _ = capsys.readouterr()
        assert out == 'completed: 1, elapsed: 1s, 1.0 tasks/s'
        # with total task num
        prog_bar = ProgressBar(10, bar_width=bar_width)
        capsys.readouterr()
        time.sleep(1)
        prog_bar.update()
        out, _ = capsys.readouterr()
        assert out == ('\r[{}] 1/10, 1.0 task/s, elapsed: 1s, ETA:     9s'.
                       format('>' * 2 + ' ' * 18))


def sleep_1s(num):
    time.sleep(1)
    return num


def test_track_progress_list(capsys):

    ret = track_progress(sleep_1s, [1, 2, 3], bar_width=3)
    out, _ = capsys.readouterr()
    assert out == ('[   ] 0/3, elapsed: 0s, ETA:'
                   '\r[>  ] 1/3, 1.0 task/s, elapsed: 1s, ETA:     2s'
                   '\r[>> ] 2/3, 1.0 task/s, elapsed: 2s, ETA:     1s'
                   '\r[>>>] 3/3, 1.0 task/s, elapsed: 3s, ETA:     0s\n')
    assert ret == [1, 2, 3]


def test_track_progress_iterator(capsys):

    ret = track_progress(
        sleep_1s, ((i for i in [1, 2, 3]), 3), bar_width=3)
    out, _ = capsys.readouterr()
    assert out == ('[   ] 0/3, elapsed: 0s, ETA:'
                   '\r[>  ] 1/3, 1.0 task/s, elapsed: 1s, ETA:     2s'
                   '\r[>> ] 2/3, 1.0 task/s, elapsed: 2s, ETA:     1s'
                   '\r[>>>] 3/3, 1.0 task/s, elapsed: 3s, ETA:     0s\n')
    assert ret == [1, 2, 3]


def test_track_parallel_progress_list(capsys):

    results = track_parallel_progress(
        sleep_1s, [1, 2, 3, 4], 2, bar_width=4)
    out, _ = capsys.readouterr()
    assert out == ('[    ] 0/4, elapsed: 0s, ETA:'
                   '\r[>   ] 1/4, 1.0 task/s, elapsed: 1s, ETA:     3s'
                   '\r[>>  ] 2/4, 2.0 task/s, elapsed: 1s, ETA:     1s'
                   '\r[>>> ] 3/4, 1.5 task/s, elapsed: 2s, ETA:     1s'
                   '\r[>>>>] 4/4, 2.0 task/s, elapsed: 2s, ETA:     0s\n')
    assert results == [1, 2, 3, 4]


def test_track_parallel_progress_iterator(capsys):

    results = track_parallel_progress(
        sleep_1s, ((i for i in [1, 2, 3, 4]), 4), 2, bar_width=4)
    out, _ = capsys.readouterr()
    assert out == ('[    ] 0/4, elapsed: 0s, ETA:'
                   '\r[>   ] 1/4, 1.0 task/s, elapsed: 1s, ETA:     3s'
                   '\r[>>  ] 2/4, 2.0 task/s, elapsed: 1s, ETA:     1s'
                   '\r[>>> ] 3/4, 1.5 task/s, elapsed: 2s, ETA:     1s'
                   '\r[>>>>] 4/4, 2.0 task/s, elapsed: 2s, ETA:     0s\n')
    assert results == [1, 2, 3, 4]
