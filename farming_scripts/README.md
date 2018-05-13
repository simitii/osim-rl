# Code for Parallel Training(Farming)

Codes are taken from an open-source code base https://github.com/ctmakro/stanford-osrl and modified according to the requirements of our project. 

## Dependencies
  - Python 2.7 (For running osim-rl. There were some compatiblity issues with Python 3) 
  - osim-rl (the simulation interface)
  - opensim (the simulation software)
  - Pyro4 (RPC. For communication)

## Parallelism / Farming
* Since the envirenment is very slow or computationally expensive, farming(running multiple environments parallel) is crutial. 
* Note: Please read <https://github.com/stanfordnmbl/osim-rl/issues/58> for more information about the slowness of the environment.

## How to Run
Assuming all dependencies are installed. Following command should be run on the training machine(a lot of cores needed):
```
python2 farm.py
```

## What Does "farm.py" Do
It starts a Pyro4 server which listens **20099 port** for command to run environments in parallel threads. When commanded over the port, it will run the environment with given parameters(action vs) and report the result(observations,rewars vs) to the commander.


## How to Command "farm.py"
Commanding is done with [rlkit](https://github.com/simitii/rlkit) since we use [rlkit](https://github.com/simitii/rlkit) as the reinforcement learning framework.