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

!!! note
    This yaml file is optional for the experiment.

```yaml
num_commands: 3 # number of commands
lin_vel_x_range: [0.5, 0.5] # range of linear velocity in x direction
lin_vel_y_range: [0.0, 0.0] # range of linear velocity in y direction
ang_vel_range: [0.0, 0.0] # range of angular velocity
```

This file describes the configuration for the robot command.

In this experiment, the robot can be given a speed command.

The speed commands are given in the forward and backward directions and in the direction of rotation.

### entities.py

!!! warning
    This python script is required for the experiment.

```python
import genesis as gs
from typing import List


def get_entities() -> List[gs.morphs.Morph]:
    return [gs.morphs.Plane()]
```

This python script describes the configuration for the entities inside simulation.

This script must contain function named `get_entities()` with `List[gs.morphs.Morph]` return type.

### environment_config.yaml

!!! note
    This yaml file is optional for the experiment.

```yaml
default_joint_angles: # The default joint angles for the robot
  FL_hip_joint: 0.0 # Front left hip joint, this name comes from the robot URDF
  FR_hip_joint: 0.0
  RL_hip_joint: 0.0
  RR_hip_joint: 0.0
  FL_thigh_joint: 0.8
  FR_thigh_joint: 0.8
  RL_thigh_joint: 1.0
  RR_thigh_joint: 1.0
  FL_calf_joint: -1.5
  FR_calf_joint: -1.5
  RL_calf_joint: -1.5
  RR_calf_joint: -1.5
kp: 20.0 # Proportional gain for the PD controller
kd: 0.5 # Derivative gain for the PD controller
base_init_pos: [0.0, 0.0, 0.42] # Initial position of the robot base
base_init_quat: [1.0, 0.0, 0.0, 0.0] # Initial orientation of the robot base
episode_length_seconds: 20.0 # Length of the episode in seconds
resampling_time_seconds: 4.0 # Time between resampling the action
action_scale: 0.25 # Scale for the action space
simulate_action_latency: true # Whether to simulate action latency
clip_action: 100.0 # Clip the action to a certain range
```

This file describes the configuration for the simulation environment.

The initial posture of the robot, hyperparameter of the PPO algorithm, etc. can be set.

### observation_config.yaml

!!! note
    This yaml file is optional for the experiment.

```yaml
obs_scales: # Scale for each observation
  lin_vel: 2.0 # Scale for linear velocity
  ang_vel: 0.25 # Scale for angular velocity
  dof_pos: 1.0 # Scale for joint position
  dof_vel: 0.05 # Scale for joint velocity
```

This file describes the configuration for the observation.

Currently, only the Observation scale can be set.

### reward_functions.py

!!! warning
    This python script is required for the experiment.

```python
import torch


def get_reward_functions():
    reward_functions = []

    # ------------ reward functions----------------
    def reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / 0.25)

    reward_functions.append((reward_tracking_lin_vel, 1.0))

    def reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / 0.25)

    reward_functions.append((reward_tracking_ang_vel, 0.2))

    def reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    reward_functions.append((reward_lin_vel_z, -1.0))

    def reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    reward_functions.append((reward_action_rate, -0.005))

    def reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    reward_functions.append((reward_similar_to_default, -0.1))

    def reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - 0.3)

    reward_functions.append((reward_base_height, -50.0))

    return reward_functions

```

This python script defines the reward functions in this experiment.

This python script must contain `get_reward_functions()` function with the return value is a function object that takes self as its first argument and changes torch.Tensor to a list of tuples of type float.

1st element of the tuple means the each reward function, 2nd element of the tuple means the scale of the each reward function.

<div class="iframely-embed"><div class="iframely-responsive" style="height: 140px; padding-bottom: 0;"><a href="https://github.com/team-re-boot/genesis_ros/blob/master/genesis_ros/ppo/ppo_env.py" data-iframely-url="//iframely.net/9xjFkyjT?card=small"></a></div></div><script async src="//iframely.net/embed.js"></script>

The functions defined in this script are added as member functions of the PPOEnv class defined in ppo_env.py and executed at each simulation frame.

### simulation_config.yaml

!!! note
    This yaml file is optional for the experiment.

```yaml
simulate_action_latency: True # Whether to simulate action latency
dt: 0.02  # Time step for the simulation
```

This file describes the configuration for the simulation latency and time step.

### train_config.yaml

!!! note
    This yaml file is optional for the experiment.

```yaml
runner:
  experiment_name: go2_walking # Name of the experiment
```

This file describes the configuration for the training.

This file only needs while training.
