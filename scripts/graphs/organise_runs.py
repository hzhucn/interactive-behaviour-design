#!/usr/bin/env python3

"""
Pass me a directory which looks like

  seaquest-0-drlhp_1551196735_3994512
  seaquest-0-sdrlhp-bc_1551196683_3994512
  seaquest-0-sdrlhp_1551196722_3994512
  seaquest-1-drlhp_1551196733_3994512
  seaquest-1-sdrlhp-bc_1551196761_3994512
  seaquest-1-sdrlhp_1551196757_3994512
  seaquest-2-drlhp_1551196717_3994512
  seaquest-2-sdrlhp-bc_1551196726_3994512
  seaquest-2-sdrlhp_1551196730_3994512

and I'll turn it into

  Seaquest
    DRLHP
      0
      1
      2
    SDRLHP
      0
      1
      2
    SDRLHP-BC
      0
      1
      2
"""

import argparse

import os
import re

parser = argparse.ArgumentParser()
parser.add_argument('runs_dir')
args = parser.parse_args()

name_dict = {'seaquest': 'Seaquest',
             'lunarlander': 'Lunar Lander',
             'fetch': 'Fetch',
             'fetchpp': 'Fetch'}

types = ['sdrlhp-bc', 'sdrlhp', 'drlhp']

for f in os.scandir(args.runs_dir):
    m = re.search(r'([^-]*)-(\d)-([^_]*)_', f.name)  # e.g. (seaquest)-(0)-(drlhp-bc)_
    if m is None:
        raise Exception("Couldn't detect name format for run", f)
    env_shortname = m.group(1)
    try:
        env_name = name_dict[env_shortname]
    except KeyError:
        env_name = env_shortname
    seed = m.group(2)
    type_name = m.group(3)
    link_path = os.path.join(env_name, type_name.upper(), seed)
    os.makedirs(os.path.dirname(link_path), exist_ok=True)
    os.symlink(os.path.abspath(f.path), link_path)
