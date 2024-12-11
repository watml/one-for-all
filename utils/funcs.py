import numpy as np
import torch
from tqdm import tqdm
import os
import fcntl
import platform
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class set_numpy_seed:
    def __init__(self, seed):
        self.seed = seed
        self.state = None

    def __enter__(self):
        self.state = np.random.get_state()
        np.random.seed(self.seed)

    def __exit__(self, exc_type, exc_val, exc_tb):
        np.random.set_state(self.state)


class set_torch_seed:
    def __init__(self, seed):
        self.seed = seed
        self.state = None

    def __enter__(self):
        self.state = torch.get_rng_state()
        torch.manual_seed(self.seed)

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.set_rng_state(self.state)


class os_lock:
    def __init__(self, file, *, log=None):
        path_chips = file.split(os.sep)
        path = os.sep.join(path_chips[:-1])
        os.makedirs(path, exist_ok=True)
        self.lockfile = file + ".lock"
        self.lock_state = False
        self.file = file
        self.log = log

    def acquire(self):
        try:
            fd = os.open(self.lockfile, os.O_CREAT | os.O_EXCL)
        except OSError:
            return False
        else:
            os.close(fd)
            return True

    def release(self):
        os.remove(self.lockfile)

    def __enter__(self):
        if not os.path.exists(self.file):
            self.lock_state = self.acquire()
        if self.lock_state and self.log is not None:
            with open(self.log, "a") as f:
                f.write(f"The node {platform.node()} is running for {self.file}\n")
        return self.lock_state

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.lock_state:
            self.release()

class fcntl_lock:
    def __init__(self, path):
        os.makedirs(path, exist_ok=True)
        self.lock = open(os.path.join(path, ".locker"), "w+")

    def acquire(self):
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError:
            return False
        else:
            return True

    def release(self):
        fcntl.flock(self.lock, fcntl.LOCK_UN)
        self.lock.close()


class vd_tqdm(tqdm):
    def __init__(self, *args, file_prog=None, **kwargs):
        tqdm.__init__(self, *args, **kwargs)
        self.file_prog = file_prog
        if file_prog is not None:
            path_chips = file_prog.split(os.sep)
            self.path = os.sep.join(path_chips[:-1])
        else:
            self.path = None

    def update(self, n=1):
        displayed = super(vd_tqdm, self).update(n)
        if self.file_prog is not None and displayed:
            rate = self.format_dict["rate"]
            if rate:
                remaining = (self.format_dict["total"] - self.format_dict["n"]) / rate
                lock = fcntl_lock(self.path)
                while True:
                    is_lock = lock.acquire()
                    if is_lock:
                        break
                with open(self.file_prog, "a") as f:
                    f.write(f"the remaining time on node {platform.node()} " +
                            f"is {self.format_interval(remaining)} {datetime.now()}\n")
                lock.release()
        return displayed


class vd_DotMap:
    def __init__(self, data=None):
        if not isinstance(data, dict):
            raise TypeError("The input must be a dict")
        if data is None:
            data = {}
        self._data = data

    def __getattr__(self, attr):
        if attr in self._data:
            return self._data[attr]
        else:
            raise AttributeError(f"'vd_DotMap' object has no attribute '{attr}'")

    def __setattr__(self, attr, value):
        if attr == "_data":
            super().__setattr__(attr, value)
        else:
            raise TypeError("'vd_DotMap' object does not support attribute assignment")

    def __repr__(self):
        return repr(self._data)


def export_legend(legend, fig_saved):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(fig_saved, dpi="figure", bbox_inches=bbox)



def plot_curves(x, ys, fig_saved, labels=None, xlabel=None, ylabel=None, title=None, plot_std=True, yscale=None,
                clrs=None):
    fig, ax = plt.subplots(figsize=(32, 24))
    plt.grid()
    if labels is None:
        enumerator = zip(ys)
    else:
        enumerator = zip(ys, labels)
    if clrs is None:
        clrs = sns.color_palette("tab20", len(ys))
    for i, take in enumerate(enumerator):
        if len(take) == 2:
            y, label = take
        else:
            y, label = take, None
        curve_mean = np.mean(y, axis=0)
        ax.plot(x, curve_mean, label=label, linewidth=10, c=clrs[i])
        if plot_std and len(y.shape) == 2:
            curve_std = np.std(y, axis=0)
            ax.fill_between(x, curve_mean - curve_std, curve_mean + curve_std, alpha=0.2, facecolor=clrs[i])

    if yscale is not None:
        ax.set_yscale(yscale)
    ax.tick_params(axis='x', labelsize=80)
    ax.tick_params(axis='y', labelsize=80)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=100)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=100)
    if title is not None:
        ax.set_title(title, fontsize=100)
    if labels is not None:
        plt.legend(fontsize=75, framealpha=0.5)

    plt.savefig(fig_saved, bbox_inches='tight')
    plt.close(fig)