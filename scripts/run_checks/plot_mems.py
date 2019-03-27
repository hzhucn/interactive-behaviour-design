#!/usr/bin/env python3

import argparse
import os
import subprocess

import tempfile

from pylab import *

plt.rcParams.update({'figure.max_open_warning': 0})

parser = argparse.ArgumentParser()
parser.add_argument('mem_log', nargs='*')
args = parser.parse_args()

with tempfile.TemporaryDirectory() as d:
    for i, log in enumerate(args.mem_log):
        log_name = os.path.basename(log)
        print(log_name)
        figure()
        with open(log) as f:
            lines = f.read().rstrip().split('\n')
        mems = [float(l.split()[1]) for l in lines]
        times = [float(l.split()[2]) for l in lines]
        rtimes = [t - times[0] for t in times]
        title(log_name)
        plot(rtimes, mems)
        savefig(os.path.join(d, f'{i}.png'))
    subprocess.call(f'montage {d}/*.png -tile 6x -mode concatenate result.png', shell=True)
