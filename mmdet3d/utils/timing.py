import time
import numpy as np
import torch

_time_samples = {}

def get_time():
    torch.cuda.synchronize()
    return time.time()

def log_time(module_name, secs):
    if module_name not in _time_samples.keys():
        _time_samples[module_name] = []
    _time_samples[module_name].append(secs)

def print_time():
    print()
    for k, v in _time_samples.items():
        v = np.array(v)
        time_mean_msec = np.mean(v) * 1000
        print(f'Mean time for module {k} : {time_mean_msec:.2f} msec')
    print()
