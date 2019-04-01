# Interactive behaviour design

This repository contains the code for
[*Towards an IDE for agent design*](https://drive.google.com/file/d/1lYdp5ym5OeL0WpzVBLL1rZYpga_1h6WC/view)
(Matthew Rahtz, James Fang, Anca D. Dragan and Dylan Hadfield-Menell),
and for Matthew Rahtz's master's thesis, *Designing agents with user-defined concepts*.

## Cloning

Note that this repository contains submodules.
To make sure your clone includes those submodules, do:

`$ git clone --recurse-submodules https://github.com/HumanCompatibleAI/interactive-behaviour-design`.

## Dependencies

For binary dependencies, see the [Dockerfile](Dockerfile).

Package dependencies are listed in the [Pipfile](Pipfile), but need to be installed
in a particular order. Again, check the [Dockerfile](Dockerfile).

## Training

[`run.py`](run.py) is the main program script.

To train using demonstrations from an oracle instead of a human, use
[`scripts/train/auto_train_rl.py`](scripts/train/auto_train_rl.py)
(or
[`scripts/train/auto_train_prefs.py`](scripts/train/auto_train_prefs.py)
to train using
[Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741)
-style comparisons). This will automatically set up a tmux session
running both `run.py` and the [oracle](oracle.py), and coordinate
the various stages of training.

[`scripts/train/train_all.py`](scripts/train/train_all.py) will print out the commands
to run to produce the full set of results.

## Graphs

[`scripts/graphs/plots.py`](scripts/graphs/plots.py) generates the graphs, but expects
a particular directory setup, generated by
[`scripts/graphs/organise_runs.py`](scripts/graphs/organise_runs.py).

## Tests

To run tests:

`$ python -u -m unittest discover -v -p '*_test.py'`

These will take about 40 minutes to complete.

## Bugs and TODOs

* In Fetch tasks, the oracle gives demonstrations based on rewards based on the *changes* in position.
  I'm not sure this can actually be learned by the reward predictor -
  we see much lower reward predictor accuracy when training with these
  rewards than with rewards based on the *absolute* distances.
  We should probably switch back to that latter reward function.
* Implement monitoring for what proportion of segments from each episode are
  labelling when training in DRLHP mode.
* Success rate for the demonstration environment (`env_demo/success_rate`) is
  broken.
