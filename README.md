## Known bugs

* In Fetch tasks, the oracle gives demonstrations based on rewards based on the /changes/ in position. I'm not sure this is a function that can actually be learned by the reward predictor - we see much lower reward predictor accuracy when training with these rewards than with rewards based on the /absolute/ distances. We should probably switch back to that latter reward function.

## TODOs

* Implement monitoring for what proportion of segments from each episode are labelling when training in DRLHP mode
* The repository and the Docker images contain BAIR's MuJoCo key. Strip the key out before releasing anything publicly.
