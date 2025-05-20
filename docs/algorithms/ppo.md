# PPO

Proximal Policy Optimization

This package integrate [rsl_rl package](https://github.com/leggedrobotics/rsl_rl.git) and use PPO algorithm inside this library.

<div class="iframely-embed"><div class="iframely-responsive" style="height: 140px; padding-bottom: 0;"><a href="https://github.com/leggedrobotics/rsl_rl" data-iframely-url="//iframely.net/CTTcMoF0?card=small"></a></div></div><script async src="//iframely.net/embed.js"></script>

## How to make your experiment.

Example source code is in [this directory.](https://github.com/team-re-boot/genesis_ros/tree/master/genesis_ros/ppo/config/go2_walking)

<div class="iframely-embed"><div class="iframely-responsive" style="height: 140px; padding-bottom: 0;"><a href="https://github.com/team-re-boot/genesis_ros/tree/master/genesis_ros/ppo/config/go2_walking" data-iframely-url="//iframely.net/xh712z3b?card=small"></a></div></div><script async src="//iframely.net/embed.js"></script>

```bash
.
└── go2_walking # Name of the experiment
    ├── command_config.yaml # Configuration for the robot command
    ├── entities.py # Python script for listing up entities inside experiment
    ├── environment_config.yaml # Configuration for the environment
    ├── observation_config.yaml # Configuration for the observation
    ├── reward_functions.py # Python script for define reward functions
    ├── simulation_config.yaml # Configuration for the simulation
    └── train_config.yaml # Configuration for the training

1 directory, 7 files
```

### command_config.yaml

```yaml
num_commands: 3 # number of commands
lin_vel_x_range: [0.5, 0.5] # range of linear velocity in x direction
lin_vel_y_range: [0.0, 0.0] # range of linear velocity in y direction
ang_vel_range: [0.0, 0.0] # range of angular velocity
```

This file describes the configuration for the robot command.
In this experiment, the robot can be given a speed command. The speed commands are given in the forward and backward directions and in the direction of rotation.

### entities.py

```python
import genesis as gs
from typing import List


def get_entities() -> List[gs.morphs.Morph]:
    return [gs.morphs.Plane()]
```

This python script describes the configuration for the entities inside simulation.
This script must contain function named `get_entities()` with `List[gs.morphs.Morph]` return type.
