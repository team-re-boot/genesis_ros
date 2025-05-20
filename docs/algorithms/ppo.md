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
algorithm: ppo # Algorithm to use

policy: # Settings for the policy. See also, https://github.com/leggedrobotics/rsl_rl
  activation: elu # Activation function for the policy network
  actor_hidden_dims: [512, 256, 128] # Hidden dimensions for the actor network
  critic_hidden_dims: [512, 256, 128] # Hidden dimensions for the critic network
  init_noise_std: 1.0 # Initial noise standard deviation
  class_name: ActorCritic # Loading the ActorCritic class

runner:
  experiment_name: go2_walking # Name of the experiment
  checkpoint: -1 # Checkpoint to load, -1 means the latest checkpoint
  load_run: -1 # Load run number, -1 means the latest run
  log_interval: 1 # Interval for logging
  max_iterations: 101 # Maximum number of iterations

runner_class_name: OnPolicyRunner # Class name for the runner. See also, https://github.com/leggedrobotics/rsl_rl

```

This file describes the configuration for the training.

This file only needs while training.

## Train

```bash
uv run ppo_train --config genesis_ros/ppo/config/go2_walking/ --device gpu
```

If the training script succeed, show dialogs like below.

Command usage is below.

```bash
uv run ppo_train --help

usage: ppo_train [-h] -c CONFIG -d {cpu,gpu} [--num_environments NUM_ENVIRONMENTS] [--urdf_path URDF_PATH]

options:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Path to the config directory (default: /home/masaya/workspace/genesis_ros/genesis_ros/ppo/ppo_train.py)
  -d {cpu,gpu}, --device {cpu,gpu}
                        Specify device which you want to run PPO and simulation. (default: None)
  --num_environments NUM_ENVIRONMENTS
                        Number of environments (default: 4096)
  --urdf_path URDF_PATH
                        Path to the URDF file (default: urdf/go2/urdf/go2.urdf)
```

??? note
    uv run ppo_train --config genesis_ros/ppo/config/go2_walking/ --device gpu

    Number of joints:  12

    Joints :  ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']

    Number of actions:  12

    Adding reward function:  reward_tracking_lin_vel
    Reward_scale =  1.0
    Reward scale considering time delta =  0.02

    Adding reward function:  reward_tracking_ang_vel
    Reward_scale =  0.2
    Reward scale considering time delta =  0.004

    Adding reward function:  reward_lin_vel_z
    Reward_scale =  -1.0
    Reward scale considering time delta =  -0.02

    Adding reward function:  reward_action_rate
    Reward_scale =  -0.005
    Reward scale considering time delta =  -0.0001

    Adding reward function:  reward_similar_to_default
    Reward_scale =  -0.1
    Reward scale considering time delta =  -0.002

    Adding reward function:  reward_base_height
    Reward_scale =  -50.0
    Reward scale considering time delta =  -1.0

    Reward functions setup finished.

    Actor MLP: Sequential(

        (0): Linear(in_features=45, out_features=512, bias=True)

        (1): ELU(alpha=1.0)

        (2): Linear(in_features=512, out_features=256, bias=True)

        (3): ELU(alpha=1.0)

        (4): Linear(in_features=256, out_features=128, bias=True)

        (5): ELU(alpha=1.0)

        (6): Linear(in_features=128, out_features=12, bias=True)

    )
    Critic MLP: Sequential(

        (0): Linear(in_features=45, out_features=512, bias=True)

        (1): ELU(alpha=1.0)

        (2): Linear(in_features=512, out_features=256, bias=True)

        (3): ELU(alpha=1.0)

        (4): Linear(in_features=256, out_features=128, bias=True)

        (5): ELU(alpha=1.0)

        (6): Linear(in_features=128, out_features=1, bias=True)

    )

    ################################################################################
                            Learning iteration 0/101

                        Computation: 83209 steps/s (collection: 0.861s, learning 0.320s)
                Value function loss: 0.0125
                        Surrogate loss: -0.0004
                Mean action noise std: 1.00
                    Mean total reward: 0.17
                Mean episode length: 22.88
    Mean episode rew_reward_tracking_lin_vel: 0.0107
    Mean episode rew_reward_tracking_ang_vel: 0.0020
    Mean episode rew_reward_lin_vel_z: -0.0040
    Mean episode rew_reward_action_rate: -0.0014
    Mean episode rew_reward_similar_to_default: -0.0012
    Mean episode rew_reward_base_height: -0.0025
    --------------------------------------------------------------------------------
                    Total timesteps: 98304
                        Iteration time: 1.18s
                            Total time: 1.18s
                                ETA: 119.3s

    Storing git diff for 'genesis_ros' in: logs/go2_walking/git/genesis_ros.diff
    ################################################################################
                            Learning iteration 1/101

                        Computation: 189431 steps/s (collection: 0.381s, learning 0.138s)
                Value function loss: 0.0056
                        Surrogate loss: -0.0047
                Mean action noise std: 1.00
                    Mean total reward: 0.39
                Mean episode length: 44.82
    Mean episode rew_reward_tracking_lin_vel: 0.0284
    Mean episode rew_reward_tracking_ang_vel: 0.0050
    Mean episode rew_reward_lin_vel_z: -0.0053
    Mean episode rew_reward_action_rate: -0.0040
    Mean episode rew_reward_similar_to_default: -0.0053
    Mean episode rew_reward_base_height: -0.0034
    --------------------------------------------------------------------------------
                    Total timesteps: 196608
                        Iteration time: 0.52s
                            Total time: 1.70s
                                ETA: 85.0s

    ################################################################################
                            Learning iteration 2/101

                        Computation: 191996 steps/s (collection: 0.368s, learning 0.144s)
                Value function loss: 0.0041
                        Surrogate loss: -0.0037
                Mean action noise std: 1.00
                    Mean total reward: 0.40
                Mean episode length: 57.54
    Mean episode rew_reward_tracking_lin_vel: 0.0404
    Mean episode rew_reward_tracking_ang_vel: 0.0075
    Mean episode rew_reward_lin_vel_z: -0.0053
    Mean episode rew_reward_action_rate: -0.0063
    Mean episode rew_reward_similar_to_default: -0.0088
    Mean episode rew_reward_base_height: -0.0042
    --------------------------------------------------------------------------------
                    Total timesteps: 294912
                        Iteration time: 0.51s
                            Total time: 2.21s
                                ETA: 73.0s

    ################################################################################
                            Learning iteration 3/101

                        Computation: 179816 steps/s (collection: 0.401s, learning 0.146s)
                Value function loss: 0.0041
                        Surrogate loss: -0.0025
                Mean action noise std: 1.00
                    Mean total reward: 0.70
                Mean episode length: 79.31
    Mean episode rew_reward_tracking_lin_vel: 0.0540
    Mean episode rew_reward_tracking_ang_vel: 0.0104
    Mean episode rew_reward_lin_vel_z: -0.0054
    Mean episode rew_reward_action_rate: -0.0087
    Mean episode rew_reward_similar_to_default: -0.0125
    Mean episode rew_reward_base_height: -0.0050
    --------------------------------------------------------------------------------
                    Total timesteps: 393216
                        Iteration time: 0.55s
                            Total time: 2.76s
                                ETA: 67.6s

    ################################################################################
                            Learning iteration 4/101

                        Computation: 181743 steps/s (collection: 0.396s, learning 0.145s)
                Value function loss: 0.0035
                        Surrogate loss: -0.0083
                Mean action noise std: 1.00
                    Mean total reward: 0.86
                Mean episode length: 94.18
    Mean episode rew_reward_tracking_lin_vel: 0.0591
    Mean episode rew_reward_tracking_ang_vel: 0.0119
    Mean episode rew_reward_lin_vel_z: -0.0056
    Mean episode rew_reward_action_rate: -0.0101
    Mean episode rew_reward_similar_to_default: -0.0142
    Mean episode rew_reward_base_height: -0.0052
    --------------------------------------------------------------------------------
                    Total timesteps: 491520
                        Iteration time: 0.54s
                            Total time: 3.30s
                                ETA: 64.0s

    ################################################################################
                            Learning iteration 5/101

                        Computation: 190068 steps/s (collection: 0.373s, learning 0.144s)
                Value function loss: 0.0030
                        Surrogate loss: -0.0081
                Mean action noise std: 1.00
                    Mean total reward: 1.20
                Mean episode length: 118.28
    Mean episode rew_reward_tracking_lin_vel: 0.0850
    Mean episode rew_reward_tracking_ang_vel: 0.0154
    Mean episode rew_reward_lin_vel_z: -0.0058
    Mean episode rew_reward_action_rate: -0.0133
    Mean episode rew_reward_similar_to_default: -0.0187
    Mean episode rew_reward_base_height: -0.0058
    --------------------------------------------------------------------------------
                    Total timesteps: 589824
                        Iteration time: 0.52s
                            Total time: 3.82s
                                ETA: 61.1s

    ################################################################################
                            Learning iteration 6/101

                        Computation: 195093 steps/s (collection: 0.369s, learning 0.135s)
                Value function loss: 0.0030
                        Surrogate loss: -0.0071
                Mean action noise std: 1.00
                    Mean total reward: 1.28
                Mean episode length: 139.45
    Mean episode rew_reward_tracking_lin_vel: 0.0935
    Mean episode rew_reward_tracking_ang_vel: 0.0189
    Mean episode rew_reward_lin_vel_z: -0.0059
    Mean episode rew_reward_action_rate: -0.0159
    Mean episode rew_reward_similar_to_default: -0.0225
    Mean episode rew_reward_base_height: -0.0060
    --------------------------------------------------------------------------------
                    Total timesteps: 688128
                        Iteration time: 0.50s
                            Total time: 4.32s
                                ETA: 58.6s

    ################################################################################
                            Learning iteration 7/101

                        Computation: 194238 steps/s (collection: 0.371s, learning 0.135s)
                Value function loss: 0.0026
                        Surrogate loss: -0.0078
                Mean action noise std: 1.00
                    Mean total reward: 1.15
                Mean episode length: 155.41
    Mean episode rew_reward_tracking_lin_vel: 0.0951
    Mean episode rew_reward_tracking_ang_vel: 0.0212
    Mean episode rew_reward_lin_vel_z: -0.0059
    Mean episode rew_reward_action_rate: -0.0180
    Mean episode rew_reward_similar_to_default: -0.0252
    Mean episode rew_reward_base_height: -0.0061
    --------------------------------------------------------------------------------
                    Total timesteps: 786432
                        Iteration time: 0.51s
                            Total time: 4.83s
                                ETA: 56.7s

    ################################################################################
                            Learning iteration 8/101

                        Computation: 190354 steps/s (collection: 0.380s, learning 0.137s)
                Value function loss: 0.0020
                        Surrogate loss: -0.0086
                Mean action noise std: 1.00
                    Mean total reward: 1.12
                Mean episode length: 171.41
    Mean episode rew_reward_tracking_lin_vel: 0.0909
    Mean episode rew_reward_tracking_ang_vel: 0.0232
    Mean episode rew_reward_lin_vel_z: -0.0060
    Mean episode rew_reward_action_rate: -0.0195
    Mean episode rew_reward_similar_to_default: -0.0275
    Mean episode rew_reward_base_height: -0.0064
    --------------------------------------------------------------------------------
                    Total timesteps: 884736
                        Iteration time: 0.52s
                            Total time: 5.34s
                                ETA: 55.2s

    ################################################################################
                            Learning iteration 9/101

                        Computation: 195596 steps/s (collection: 0.367s, learning 0.135s)
                Value function loss: 0.0021
                        Surrogate loss: -0.0067
                Mean action noise std: 1.00
                    Mean total reward: 1.36
                Mean episode length: 185.04
    Mean episode rew_reward_tracking_lin_vel: 0.1008
    Mean episode rew_reward_tracking_ang_vel: 0.0254
    Mean episode rew_reward_lin_vel_z: -0.0060
    Mean episode rew_reward_action_rate: -0.0213
    Mean episode rew_reward_similar_to_default: -0.0303
    Mean episode rew_reward_base_height: -0.0067
    --------------------------------------------------------------------------------
                    Total timesteps: 983040
                        Iteration time: 0.50s
                            Total time: 5.85s
                                ETA: 53.8s

    ################################################################################
                        Learning iteration 10/101

                        Computation: 197626 steps/s (collection: 0.362s, learning 0.135s)
                Value function loss: 0.0017
                        Surrogate loss: -0.0024
                Mean action noise std: 0.99
                    Mean total reward: 1.32
                Mean episode length: 190.39
    Mean episode rew_reward_tracking_lin_vel: 0.1045
    Mean episode rew_reward_tracking_ang_vel: 0.0264
    Mean episode rew_reward_lin_vel_z: -0.0060
    Mean episode rew_reward_action_rate: -0.0219
    Mean episode rew_reward_similar_to_default: -0.0313
    Mean episode rew_reward_base_height: -0.0068
    --------------------------------------------------------------------------------
                    Total timesteps: 1081344
                        Iteration time: 0.50s
                            Total time: 6.34s
                                ETA: 52.5s

    ################################################################################
                        Learning iteration 11/101

                        Computation: 197344 steps/s (collection: 0.362s, learning 0.136s)
                Value function loss: 0.0014
                        Surrogate loss: -0.0022
                Mean action noise std: 0.99
                    Mean total reward: 0.80
                Mean episode length: 141.65
    Mean episode rew_reward_tracking_lin_vel: 0.0909
    Mean episode rew_reward_tracking_ang_vel: 0.0238
    Mean episode rew_reward_lin_vel_z: -0.0058
    Mean episode rew_reward_action_rate: -0.0195
    Mean episode rew_reward_similar_to_default: -0.0280
    Mean episode rew_reward_base_height: -0.0064
    --------------------------------------------------------------------------------
                    Total timesteps: 1179648
                        Iteration time: 0.50s
                            Total time: 6.84s
                                ETA: 51.3s

    ################################################################################
                        Learning iteration 12/101

                        Computation: 197284 steps/s (collection: 0.363s, learning 0.136s)
                Value function loss: 0.0051
                        Surrogate loss: -0.0041
                Mean action noise std: 0.99
                    Mean total reward: 0.83
                Mean episode length: 134.77
    Mean episode rew_reward_tracking_lin_vel: 0.0716
    Mean episode rew_reward_tracking_ang_vel: 0.0202
    Mean episode rew_reward_lin_vel_z: -0.0055
    Mean episode rew_reward_action_rate: -0.0162
    Mean episode rew_reward_similar_to_default: -0.0233
    Mean episode rew_reward_base_height: -0.0057
    --------------------------------------------------------------------------------
                    Total timesteps: 1277952
                        Iteration time: 0.50s
                            Total time: 7.34s
                                ETA: 50.3s

    ################################################################################
                        Learning iteration 13/101

                        Computation: 195517 steps/s (collection: 0.367s, learning 0.136s)
                Value function loss: 0.0015
                        Surrogate loss: 0.0001
                Mean action noise std: 0.98
                    Mean total reward: 0.53
                Mean episode length: 88.52
    Mean episode rew_reward_tracking_lin_vel: 0.0587
    Mean episode rew_reward_tracking_ang_vel: 0.0165
    Mean episode rew_reward_lin_vel_z: -0.0053
    Mean episode rew_reward_action_rate: -0.0128
    Mean episode rew_reward_similar_to_default: -0.0187
    Mean episode rew_reward_base_height: -0.0052
    --------------------------------------------------------------------------------
                    Total timesteps: 1376256
                        Iteration time: 0.50s
                            Total time: 7.84s
                                ETA: 49.3s

    ################################################################################
                        Learning iteration 14/101

                        Computation: 197134 steps/s (collection: 0.360s, learning 0.138s)
                Value function loss: 0.0009
                        Surrogate loss: -0.0005
                Mean action noise std: 0.98
                    Mean total reward: 0.44
                Mean episode length: 73.75
    Mean episode rew_reward_tracking_lin_vel: 0.0484
    Mean episode rew_reward_tracking_ang_vel: 0.0139
    Mean episode rew_reward_lin_vel_z: -0.0052
    Mean episode rew_reward_action_rate: -0.0105
    Mean episode rew_reward_similar_to_default: -0.0157
    Mean episode rew_reward_base_height: -0.0049
    --------------------------------------------------------------------------------
                    Total timesteps: 1474560
                        Iteration time: 0.50s
                            Total time: 8.34s
                                ETA: 48.4s

    ################################################################################
                        Learning iteration 15/101

                        Computation: 193403 steps/s (collection: 0.374s, learning 0.135s)
                Value function loss: 0.0007
                        Surrogate loss: -0.0055
                Mean action noise std: 0.98
                    Mean total reward: 0.43
                Mean episode length: 72.96
    Mean episode rew_reward_tracking_lin_vel: 0.0425
    Mean episode rew_reward_tracking_ang_vel: 0.0120
    Mean episode rew_reward_lin_vel_z: -0.0051
    Mean episode rew_reward_action_rate: -0.0088
    Mean episode rew_reward_similar_to_default: -0.0135
    Mean episode rew_reward_base_height: -0.0048
    --------------------------------------------------------------------------------
                    Total timesteps: 1572864
                        Iteration time: 0.51s
                            Total time: 8.85s
                                ETA: 47.6s

    ################################################################################
                        Learning iteration 16/101

                        Computation: 187829 steps/s (collection: 0.377s, learning 0.146s)
                Value function loss: 0.0006
                        Surrogate loss: -0.0080
                Mean action noise std: 0.97
                    Mean total reward: 0.47
                Mean episode length: 75.62
    Mean episode rew_reward_tracking_lin_vel: 0.0413
    Mean episode rew_reward_tracking_ang_vel: 0.0116
    Mean episode rew_reward_lin_vel_z: -0.0050
    Mean episode rew_reward_action_rate: -0.0083
    Mean episode rew_reward_similar_to_default: -0.0130
    Mean episode rew_reward_base_height: -0.0048
    --------------------------------------------------------------------------------
                    Total timesteps: 1671168
                        Iteration time: 0.52s
                            Total time: 9.37s
                                ETA: 46.9s

    ################################################################################
                        Learning iteration 17/101

                        Computation: 184820 steps/s (collection: 0.387s, learning 0.145s)
                Value function loss: 0.0005
                        Surrogate loss: -0.0081
                Mean action noise std: 0.96
                    Mean total reward: 0.46
                Mean episode length: 74.18
    Mean episode rew_reward_tracking_lin_vel: 0.0431
    Mean episode rew_reward_tracking_ang_vel: 0.0122
    Mean episode rew_reward_lin_vel_z: -0.0050
    Mean episode rew_reward_action_rate: -0.0087
    Mean episode rew_reward_similar_to_default: -0.0137
    Mean episode rew_reward_base_height: -0.0050
    --------------------------------------------------------------------------------
                    Total timesteps: 1769472
                        Iteration time: 0.53s
                            Total time: 9.90s
                                ETA: 46.2s

    ################################################################################
                        Learning iteration 18/101

                        Computation: 200302 steps/s (collection: 0.356s, learning 0.135s)
                Value function loss: 0.0005
                        Surrogate loss: -0.0078
                Mean action noise std: 0.95
                    Mean total reward: 0.49
                Mean episode length: 80.15
    Mean episode rew_reward_tracking_lin_vel: 0.0442
    Mean episode rew_reward_tracking_ang_vel: 0.0126
    Mean episode rew_reward_lin_vel_z: -0.0051
    Mean episode rew_reward_action_rate: -0.0088
    Mean episode rew_reward_similar_to_default: -0.0141
    Mean episode rew_reward_base_height: -0.0050
    --------------------------------------------------------------------------------
                    Total timesteps: 1867776
                        Iteration time: 0.49s
                            Total time: 10.40s
                                ETA: 45.4s

    ################################################################################
                        Learning iteration 19/101

                        Computation: 185845 steps/s (collection: 0.392s, learning 0.137s)
                Value function loss: 0.0004
                        Surrogate loss: -0.0084
                Mean action noise std: 0.94
                    Mean total reward: 0.53
                Mean episode length: 84.54
    Mean episode rew_reward_tracking_lin_vel: 0.0480
    Mean episode rew_reward_tracking_ang_vel: 0.0139
    Mean episode rew_reward_lin_vel_z: -0.0051
    Mean episode rew_reward_action_rate: -0.0096
    Mean episode rew_reward_similar_to_default: -0.0158
    Mean episode rew_reward_base_height: -0.0051
    --------------------------------------------------------------------------------
                    Total timesteps: 1966080
                        Iteration time: 0.53s
                            Total time: 10.92s
                                ETA: 44.8s

    ################################################################################
                        Learning iteration 20/101

                        Computation: 191768 steps/s (collection: 0.377s, learning 0.136s)
                Value function loss: 0.0004
                        Surrogate loss: -0.0105
                Mean action noise std: 0.93
                    Mean total reward: 0.60
                Mean episode length: 94.24
    Mean episode rew_reward_tracking_lin_vel: 0.0510
    Mean episode rew_reward_tracking_ang_vel: 0.0152
    Mean episode rew_reward_lin_vel_z: -0.0050
    Mean episode rew_reward_action_rate: -0.0102
    Mean episode rew_reward_similar_to_default: -0.0172
    Mean episode rew_reward_base_height: -0.0051
    --------------------------------------------------------------------------------
                    Total timesteps: 2064384
                        Iteration time: 0.51s
                            Total time: 11.44s
                                ETA: 44.1s

    ################################################################################
                        Learning iteration 21/101

                        Computation: 197235 steps/s (collection: 0.364s, learning 0.135s)
                Value function loss: 0.0003
                        Surrogate loss: -0.0100
                Mean action noise std: 0.92
                    Mean total reward: 0.63
                Mean episode length: 99.88
    Mean episode rew_reward_tracking_lin_vel: 0.0542
    Mean episode rew_reward_tracking_ang_vel: 0.0165
    Mean episode rew_reward_lin_vel_z: -0.0050
    Mean episode rew_reward_action_rate: -0.0109
    Mean episode rew_reward_similar_to_default: -0.0186
    Mean episode rew_reward_base_height: -0.0052
    --------------------------------------------------------------------------------
                    Total timesteps: 2162688
                        Iteration time: 0.50s
                            Total time: 11.94s
                                ETA: 43.4s

    ################################################################################
                        Learning iteration 22/101

                        Computation: 193296 steps/s (collection: 0.365s, learning 0.144s)
                Value function loss: 0.0003
                        Surrogate loss: -0.0077
                Mean action noise std: 0.91
                    Mean total reward: 0.71
                Mean episode length: 113.95
    Mean episode rew_reward_tracking_lin_vel: 0.0575
    Mean episode rew_reward_tracking_ang_vel: 0.0179
    Mean episode rew_reward_lin_vel_z: -0.0051
    Mean episode rew_reward_action_rate: -0.0115
    Mean episode rew_reward_similar_to_default: -0.0200
    Mean episode rew_reward_base_height: -0.0052
    --------------------------------------------------------------------------------
                    Total timesteps: 2260992
                        Iteration time: 0.51s
                            Total time: 12.44s
                                ETA: 42.7s

    ################################################################################
                        Learning iteration 23/101

                        Computation: 173925 steps/s (collection: 0.416s, learning 0.150s)
                Value function loss: 0.0003
                        Surrogate loss: -0.0078
                Mean action noise std: 0.89
                    Mean total reward: 0.77
                Mean episode length: 122.36
    Mean episode rew_reward_tracking_lin_vel: 0.0613
    Mean episode rew_reward_tracking_ang_vel: 0.0196
    Mean episode rew_reward_lin_vel_z: -0.0050
    Mean episode rew_reward_action_rate: -0.0122
    Mean episode rew_reward_similar_to_default: -0.0216
    Mean episode rew_reward_base_height: -0.0050
    --------------------------------------------------------------------------------
                    Total timesteps: 2359296
                        Iteration time: 0.57s
                            Total time: 13.01s
                                ETA: 42.3s

    ################################################################################
                        Learning iteration 24/101

                        Computation: 181843 steps/s (collection: 0.391s, learning 0.149s)
                Value function loss: 0.0002
                        Surrogate loss: -0.0070
                Mean action noise std: 0.88
                    Mean total reward: 0.80
                Mean episode length: 121.07
    Mean episode rew_reward_tracking_lin_vel: 0.0633
    Mean episode rew_reward_tracking_ang_vel: 0.0201
    Mean episode rew_reward_lin_vel_z: -0.0050
    Mean episode rew_reward_action_rate: -0.0124
    Mean episode rew_reward_similar_to_default: -0.0220
    Mean episode rew_reward_base_height: -0.0050
    --------------------------------------------------------------------------------
                    Total timesteps: 2457600
                        Iteration time: 0.54s
                            Total time: 13.55s
                                ETA: 41.7s

    ################################################################################
                        Learning iteration 25/101

                        Computation: 185099 steps/s (collection: 0.384s, learning 0.147s)
                Value function loss: 0.0002
                        Surrogate loss: -0.0051
                Mean action noise std: 0.87
                    Mean total reward: 0.86
                Mean episode length: 133.35
    Mean episode rew_reward_tracking_lin_vel: 0.0687
    Mean episode rew_reward_tracking_ang_vel: 0.0228
    Mean episode rew_reward_lin_vel_z: -0.0050
    Mean episode rew_reward_action_rate: -0.0137
    Mean episode rew_reward_similar_to_default: -0.0247
    Mean episode rew_reward_base_height: -0.0049
    --------------------------------------------------------------------------------
                    Total timesteps: 2555904
                        Iteration time: 0.53s
                            Total time: 14.08s
                                ETA: 41.2s

    ################################################################################
                        Learning iteration 26/101

                        Computation: 186376 steps/s (collection: 0.381s, learning 0.147s)
                Value function loss: 0.0008
                        Surrogate loss: -0.0054
                Mean action noise std: 0.86
                    Mean total reward: 0.92
                Mean episode length: 144.85
    Mean episode rew_reward_tracking_lin_vel: 0.0697
    Mean episode rew_reward_tracking_ang_vel: 0.0233
    Mean episode rew_reward_lin_vel_z: -0.0049
    Mean episode rew_reward_action_rate: -0.0138
    Mean episode rew_reward_similar_to_default: -0.0250
    Mean episode rew_reward_base_height: -0.0047
    --------------------------------------------------------------------------------
                    Total timesteps: 2654208
                        Iteration time: 0.53s
                            Total time: 14.61s
                                ETA: 40.6s

    ################################################################################
                        Learning iteration 27/101

                        Computation: 195102 steps/s (collection: 0.366s, learning 0.138s)
                Value function loss: 0.0002
                        Surrogate loss: -0.0081
                Mean action noise std: 0.85
                    Mean total reward: 0.94
                Mean episode length: 145.95
    Mean episode rew_reward_tracking_lin_vel: 0.0774
    Mean episode rew_reward_tracking_ang_vel: 0.0265
    Mean episode rew_reward_lin_vel_z: -0.0051
    Mean episode rew_reward_action_rate: -0.0153
    Mean episode rew_reward_similar_to_default: -0.0282
    Mean episode rew_reward_base_height: -0.0052
    --------------------------------------------------------------------------------
                    Total timesteps: 2752512
                        Iteration time: 0.50s
                            Total time: 15.11s
                                ETA: 39.9s

    ################################################################################
                        Learning iteration 28/101

                        Computation: 214375 steps/s (collection: 0.316s, learning 0.143s)
                Value function loss: 0.0001
                        Surrogate loss: -0.0087
                Mean action noise std: 0.84
                    Mean total reward: 0.95
                Mean episode length: 147.90
    Mean episode rew_reward_tracking_lin_vel: 0.0834
    Mean episode rew_reward_tracking_ang_vel: 0.0304
    Mean episode rew_reward_lin_vel_z: -0.0048
    Mean episode rew_reward_action_rate: -0.0174
    Mean episode rew_reward_similar_to_default: -0.0303
    Mean episode rew_reward_base_height: -0.0045
    --------------------------------------------------------------------------------
                    Total timesteps: 2850816
                        Iteration time: 0.46s
                            Total time: 15.57s
                                ETA: 39.2s

    ################################################################################
                        Learning iteration 29/101

                        Computation: 209287 steps/s (collection: 0.329s, learning 0.141s)
                Value function loss: 0.0002
                        Surrogate loss: -0.0075
                Mean action noise std: 0.83
                    Mean total reward: 0.98
                Mean episode length: 149.87
    Mean episode rew_reward_tracking_lin_vel: 0.0836
    Mean episode rew_reward_tracking_ang_vel: 0.0294
    Mean episode rew_reward_lin_vel_z: -0.0050
    Mean episode rew_reward_action_rate: -0.0165
    Mean episode rew_reward_similar_to_default: -0.0297
    Mean episode rew_reward_base_height: -0.0043
    --------------------------------------------------------------------------------
                    Total timesteps: 2949120
                        Iteration time: 0.47s
                            Total time: 16.04s
                                ETA: 38.5s

    ################################################################################
                        Learning iteration 30/101

                        Computation: 214500 steps/s (collection: 0.313s, learning 0.145s)
                Value function loss: 0.0002
                        Surrogate loss: -0.0057
                Mean action noise std: 0.81
                    Mean total reward: 1.03
                Mean episode length: 159.16
    Mean episode rew_reward_tracking_lin_vel: 0.0916
    Mean episode rew_reward_tracking_ang_vel: 0.0324
    Mean episode rew_reward_lin_vel_z: -0.0051
    Mean episode rew_reward_action_rate: -0.0182
    Mean episode rew_reward_similar_to_default: -0.0343
    Mean episode rew_reward_base_height: -0.0051
    --------------------------------------------------------------------------------
                    Total timesteps: 3047424
                        Iteration time: 0.46s
                            Total time: 16.50s
                                ETA: 37.8s

    ################################################################################
                        Learning iteration 31/101

                        Computation: 219545 steps/s (collection: 0.305s, learning 0.143s)
                Value function loss: 0.0003
                        Surrogate loss: -0.0053
                Mean action noise std: 0.80
                    Mean total reward: 1.07
                Mean episode length: 164.88
    Mean episode rew_reward_tracking_lin_vel: 0.1222
    Mean episode rew_reward_tracking_ang_vel: 0.0474
    Mean episode rew_reward_lin_vel_z: -0.0059
    Mean episode rew_reward_action_rate: -0.0263
    Mean episode rew_reward_similar_to_default: -0.0484
    Mean episode rew_reward_base_height: -0.0068
    --------------------------------------------------------------------------------
                    Total timesteps: 3145728
                        Iteration time: 0.45s
                            Total time: 16.95s
                                ETA: 37.1s

    ################################################################################
                        Learning iteration 32/101

                        Computation: 233311 steps/s (collection: 0.287s, learning 0.134s)
                Value function loss: 0.0002
                        Surrogate loss: -0.0071
                Mean action noise std: 0.80
                    Mean total reward: 1.09
                Mean episode length: 166.66
    Mean episode rew_reward_tracking_lin_vel: 0.1014
    Mean episode rew_reward_tracking_ang_vel: 0.0389
    Mean episode rew_reward_lin_vel_z: -0.0052
    Mean episode rew_reward_action_rate: -0.0198
    Mean episode rew_reward_similar_to_default: -0.0386
    Mean episode rew_reward_base_height: -0.0046
    --------------------------------------------------------------------------------
                    Total timesteps: 3244032
                        Iteration time: 0.42s
                            Total time: 17.37s
                                ETA: 36.3s

    ################################################################################
                        Learning iteration 33/101

                        Computation: 230559 steps/s (collection: 0.292s, learning 0.134s)
                Value function loss: 0.0002
                        Surrogate loss: -0.0078
                Mean action noise std: 0.79
                    Mean total reward: 1.13
                Mean episode length: 173.06
    Mean episode rew_reward_tracking_lin_vel: 0.1142
    Mean episode rew_reward_tracking_ang_vel: 0.0423
    Mean episode rew_reward_lin_vel_z: -0.0053
    Mean episode rew_reward_action_rate: -0.0227
    Mean episode rew_reward_similar_to_default: -0.0435
    Mean episode rew_reward_base_height: -0.0050
    --------------------------------------------------------------------------------
                    Total timesteps: 3342336
                        Iteration time: 0.43s
                            Total time: 17.79s
                                ETA: 35.6s

    ################################################################################
                        Learning iteration 34/101

                        Computation: 212580 steps/s (collection: 0.318s, learning 0.144s)
                Value function loss: 0.0003
                        Surrogate loss: -0.0062
                Mean action noise std: 0.78
                    Mean total reward: 1.24
                Mean episode length: 193.03
    Mean episode rew_reward_tracking_lin_vel: 0.1305
    Mean episode rew_reward_tracking_ang_vel: 0.0516
    Mean episode rew_reward_lin_vel_z: -0.0056
    Mean episode rew_reward_action_rate: -0.0268
    Mean episode rew_reward_similar_to_default: -0.0511
    Mean episode rew_reward_base_height: -0.0056
    --------------------------------------------------------------------------------
                    Total timesteps: 3440640
                        Iteration time: 0.46s
                            Total time: 18.26s
                                ETA: 34.9s

    ################################################################################
                        Learning iteration 35/101

                        Computation: 211898 steps/s (collection: 0.320s, learning 0.144s)
                Value function loss: 0.0002
                        Surrogate loss: -0.0072
                Mean action noise std: 0.78
                    Mean total reward: 1.34
                Mean episode length: 208.77
    Mean episode rew_reward_tracking_lin_vel: 0.1313
    Mean episode rew_reward_tracking_ang_vel: 0.0523
    Mean episode rew_reward_lin_vel_z: -0.0056
    Mean episode rew_reward_action_rate: -0.0265
    Mean episode rew_reward_similar_to_default: -0.0516
    Mean episode rew_reward_base_height: -0.0057
    --------------------------------------------------------------------------------
                    Total timesteps: 3538944
                        Iteration time: 0.46s
                            Total time: 18.72s
                                ETA: 34.3s

    ################################################################################
                        Learning iteration 36/101

                        Computation: 195494 steps/s (collection: 0.358s, learning 0.145s)
                Value function loss: 0.0004
                        Surrogate loss: -0.0071
                Mean action noise std: 0.77
                    Mean total reward: 1.56
                Mean episode length: 241.08
    Mean episode rew_reward_tracking_lin_vel: 0.1463
    Mean episode rew_reward_tracking_ang_vel: 0.0582
    Mean episode rew_reward_lin_vel_z: -0.0057
    Mean episode rew_reward_action_rate: -0.0283
    Mean episode rew_reward_similar_to_default: -0.0568
    Mean episode rew_reward_base_height: -0.0054
    --------------------------------------------------------------------------------
                    Total timesteps: 3637248
                        Iteration time: 0.50s
                            Total time: 19.22s
                                ETA: 33.8s

    ################################################################################
                        Learning iteration 37/101

                        Computation: 201471 steps/s (collection: 0.352s, learning 0.136s)
                Value function loss: 0.0006
                        Surrogate loss: -0.0083
                Mean action noise std: 0.77
                    Mean total reward: 2.02
                Mean episode length: 312.09
    Mean episode rew_reward_tracking_lin_vel: 0.1517
    Mean episode rew_reward_tracking_ang_vel: 0.0606
    Mean episode rew_reward_lin_vel_z: -0.0056
    Mean episode rew_reward_action_rate: -0.0290
    Mean episode rew_reward_similar_to_default: -0.0581
    Mean episode rew_reward_base_height: -0.0056
    --------------------------------------------------------------------------------
                    Total timesteps: 3735552
                        Iteration time: 0.49s
                            Total time: 19.71s
                                ETA: 33.2s

    ################################################################################
                        Learning iteration 38/101

                        Computation: 199155 steps/s (collection: 0.358s, learning 0.136s)
                Value function loss: 0.0007
                        Surrogate loss: -0.0092
                Mean action noise std: 0.77
                    Mean total reward: 2.34
                Mean episode length: 354.41
    Mean episode rew_reward_tracking_lin_vel: 0.1631
    Mean episode rew_reward_tracking_ang_vel: 0.0658
    Mean episode rew_reward_lin_vel_z: -0.0056
    Mean episode rew_reward_action_rate: -0.0311
    Mean episode rew_reward_similar_to_default: -0.0620
    Mean episode rew_reward_base_height: -0.0054
    --------------------------------------------------------------------------------
                    Total timesteps: 3833856
                        Iteration time: 0.49s
                            Total time: 20.21s
                                ETA: 32.6s

    ################################################################################
                        Learning iteration 39/101

                        Computation: 204695 steps/s (collection: 0.345s, learning 0.135s)
                Value function loss: 0.0011
                        Surrogate loss: -0.0069
                Mean action noise std: 0.78
                    Mean total reward: 2.54
                Mean episode length: 384.71
    Mean episode rew_reward_tracking_lin_vel: 0.1722
    Mean episode rew_reward_tracking_ang_vel: 0.0694
    Mean episode rew_reward_lin_vel_z: -0.0060
    Mean episode rew_reward_action_rate: -0.0332
    Mean episode rew_reward_similar_to_default: -0.0658
    Mean episode rew_reward_base_height: -0.0058
    --------------------------------------------------------------------------------
                    Total timesteps: 3932160
                        Iteration time: 0.48s
                            Total time: 20.69s
                                ETA: 32.1s

    ################################################################################
                        Learning iteration 40/101

                        Computation: 199028 steps/s (collection: 0.355s, learning 0.139s)
                Value function loss: 0.0021
                        Surrogate loss: -0.0055
                Mean action noise std: 0.78
                    Mean total reward: 2.49
                Mean episode length: 375.56
    Mean episode rew_reward_tracking_lin_vel: 0.1630
    Mean episode rew_reward_tracking_ang_vel: 0.0650
    Mean episode rew_reward_lin_vel_z: -0.0060
    Mean episode rew_reward_action_rate: -0.0307
    Mean episode rew_reward_similar_to_default: -0.0618
    Mean episode rew_reward_base_height: -0.0057
    --------------------------------------------------------------------------------
                    Total timesteps: 4030464
                        Iteration time: 0.49s
                            Total time: 21.18s
                                ETA: 31.5s

    ################################################################################
                        Learning iteration 41/101

                        Computation: 193496 steps/s (collection: 0.372s, learning 0.136s)
                Value function loss: 0.0037
                        Surrogate loss: -0.0040
                Mean action noise std: 0.79
                    Mean total reward: 2.86
                Mean episode length: 430.23
    Mean episode rew_reward_tracking_lin_vel: 0.1769
    Mean episode rew_reward_tracking_ang_vel: 0.0702
    Mean episode rew_reward_lin_vel_z: -0.0060
    Mean episode rew_reward_action_rate: -0.0331
    Mean episode rew_reward_similar_to_default: -0.0680
    Mean episode rew_reward_base_height: -0.0059
    --------------------------------------------------------------------------------
                    Total timesteps: 4128768
                        Iteration time: 0.51s
                            Total time: 21.69s
                                ETA: 31.0s

    ################################################################################
                        Learning iteration 42/101

                        Computation: 197367 steps/s (collection: 0.360s, learning 0.138s)
                Value function loss: 0.0064
                        Surrogate loss: -0.0028
                Mean action noise std: 0.79
                    Mean total reward: 2.83
                Mean episode length: 418.23
    Mean episode rew_reward_tracking_lin_vel: 0.1863
    Mean episode rew_reward_tracking_ang_vel: 0.0734
    Mean episode rew_reward_lin_vel_z: -0.0062
    Mean episode rew_reward_action_rate: -0.0346
    Mean episode rew_reward_similar_to_default: -0.0711
    Mean episode rew_reward_base_height: -0.0060
    --------------------------------------------------------------------------------
                    Total timesteps: 4227072
                        Iteration time: 0.50s
                            Total time: 22.19s
                                ETA: 30.4s

    ################################################################################
                        Learning iteration 43/101

                        Computation: 191008 steps/s (collection: 0.371s, learning 0.143s)
                Value function loss: 0.0081
                        Surrogate loss: -0.0035
                Mean action noise std: 0.80
                    Mean total reward: 3.22
                Mean episode length: 471.42
    Mean episode rew_reward_tracking_lin_vel: 0.1955
    Mean episode rew_reward_tracking_ang_vel: 0.0758
    Mean episode rew_reward_lin_vel_z: -0.0063
    Mean episode rew_reward_action_rate: -0.0355
    Mean episode rew_reward_similar_to_default: -0.0741
    Mean episode rew_reward_base_height: -0.0060
    --------------------------------------------------------------------------------
                    Total timesteps: 4325376
                        Iteration time: 0.51s
                            Total time: 22.70s
                                ETA: 29.9s

    ################################################################################
                        Learning iteration 44/101

                        Computation: 194568 steps/s (collection: 0.360s, learning 0.146s)
                Value function loss: 0.0078
                        Surrogate loss: -0.0036
                Mean action noise std: 0.80
                    Mean total reward: 3.10
                Mean episode length: 443.47
    Mean episode rew_reward_tracking_lin_vel: 0.2088
    Mean episode rew_reward_tracking_ang_vel: 0.0794
    Mean episode rew_reward_lin_vel_z: -0.0065
    Mean episode rew_reward_action_rate: -0.0374
    Mean episode rew_reward_similar_to_default: -0.0783
    Mean episode rew_reward_base_height: -0.0062
    --------------------------------------------------------------------------------
                    Total timesteps: 4423680
                        Iteration time: 0.51s
                            Total time: 23.21s
                                ETA: 29.4s

    ################################################################################
                        Learning iteration 45/101

                        Computation: 187721 steps/s (collection: 0.379s, learning 0.145s)
                Value function loss: 0.0069
                        Surrogate loss: -0.0038
                Mean action noise std: 0.80
                    Mean total reward: 3.06
                Mean episode length: 430.89
    Mean episode rew_reward_tracking_lin_vel: 0.2025
    Mean episode rew_reward_tracking_ang_vel: 0.0742
    Mean episode rew_reward_lin_vel_z: -0.0065
    Mean episode rew_reward_action_rate: -0.0350
    Mean episode rew_reward_similar_to_default: -0.0753
    Mean episode rew_reward_base_height: -0.0062
    --------------------------------------------------------------------------------
                    Total timesteps: 4521984
                        Iteration time: 0.52s
                            Total time: 23.73s
                                ETA: 28.9s

    ################################################################################
                        Learning iteration 46/101

                        Computation: 187437 steps/s (collection: 0.380s, learning 0.145s)
                Value function loss: 0.0057
                        Surrogate loss: -0.0041
                Mean action noise std: 0.81
                    Mean total reward: 3.05
                Mean episode length: 410.10
    Mean episode rew_reward_tracking_lin_vel: 0.1952
    Mean episode rew_reward_tracking_ang_vel: 0.0692
    Mean episode rew_reward_lin_vel_z: -0.0064
    Mean episode rew_reward_action_rate: -0.0324
    Mean episode rew_reward_similar_to_default: -0.0707
    Mean episode rew_reward_base_height: -0.0058
    --------------------------------------------------------------------------------
                    Total timesteps: 4620288
                        Iteration time: 0.52s
                            Total time: 24.25s
                                ETA: 28.4s

    ################################################################################
                        Learning iteration 47/101

                        Computation: 185164 steps/s (collection: 0.388s, learning 0.143s)
                Value function loss: 0.0057
                        Surrogate loss: -0.0042
                Mean action noise std: 0.81
                    Mean total reward: 3.19
                Mean episode length: 430.97
    Mean episode rew_reward_tracking_lin_vel: 0.1908
    Mean episode rew_reward_tracking_ang_vel: 0.0648
    Mean episode rew_reward_lin_vel_z: -0.0064
    Mean episode rew_reward_action_rate: -0.0306
    Mean episode rew_reward_similar_to_default: -0.0697
    Mean episode rew_reward_base_height: -0.0060
    --------------------------------------------------------------------------------
                    Total timesteps: 4718592
                        Iteration time: 0.53s
                            Total time: 24.78s
                                ETA: 27.9s

    ################################################################################
                        Learning iteration 48/101

                        Computation: 197451 steps/s (collection: 0.361s, learning 0.137s)
                Value function loss: 0.0049
                        Surrogate loss: -0.0054
                Mean action noise std: 0.81
                    Mean total reward: 3.68
                Mean episode length: 490.70
    Mean episode rew_reward_tracking_lin_vel: 0.2481
    Mean episode rew_reward_tracking_ang_vel: 0.0849
    Mean episode rew_reward_lin_vel_z: -0.0072
    Mean episode rew_reward_action_rate: -0.0404
    Mean episode rew_reward_similar_to_default: -0.0907
    Mean episode rew_reward_base_height: -0.0069
    --------------------------------------------------------------------------------
                    Total timesteps: 4816896
                        Iteration time: 0.50s
                            Total time: 25.28s
                                ETA: 27.3s

    ################################################################################
                        Learning iteration 49/101

                        Computation: 181522 steps/s (collection: 0.391s, learning 0.150s)
                Value function loss: 0.0046
                        Surrogate loss: -0.0067
                Mean action noise std: 0.82
                    Mean total reward: 3.16
                Mean episode length: 410.28
    Mean episode rew_reward_tracking_lin_vel: 0.2208
    Mean episode rew_reward_tracking_ang_vel: 0.0730
    Mean episode rew_reward_lin_vel_z: -0.0068
    Mean episode rew_reward_action_rate: -0.0346
    Mean episode rew_reward_similar_to_default: -0.0792
    Mean episode rew_reward_base_height: -0.0064
    --------------------------------------------------------------------------------
                    Total timesteps: 4915200
                        Iteration time: 0.54s
                            Total time: 25.82s
                                ETA: 26.9s

    ################################################################################
                        Learning iteration 50/101

                        Computation: 195306 steps/s (collection: 0.367s, learning 0.136s)
                Value function loss: 0.0061
                        Surrogate loss: -0.0033
                Mean action noise std: 0.82
                    Mean total reward: 3.78
                Mean episode length: 483.20
    Mean episode rew_reward_tracking_lin_vel: 0.2518
    Mean episode rew_reward_tracking_ang_vel: 0.0818
    Mean episode rew_reward_lin_vel_z: -0.0073
    Mean episode rew_reward_action_rate: -0.0393
    Mean episode rew_reward_similar_to_default: -0.0898
    Mean episode rew_reward_base_height: -0.0069
    --------------------------------------------------------------------------------
                    Total timesteps: 5013504
                        Iteration time: 0.50s
                            Total time: 26.33s
                                ETA: 26.3s

    ################################################################################
                        Learning iteration 51/101

                        Computation: 192340 steps/s (collection: 0.362s, learning 0.149s)
                Value function loss: 0.0040
                        Surrogate loss: -0.0037
                Mean action noise std: 0.82
                    Mean total reward: 3.85
                Mean episode length: 479.40
    Mean episode rew_reward_tracking_lin_vel: 0.2520
    Mean episode rew_reward_tracking_ang_vel: 0.0800
    Mean episode rew_reward_lin_vel_z: -0.0072
    Mean episode rew_reward_action_rate: -0.0384
    Mean episode rew_reward_similar_to_default: -0.0884
    Mean episode rew_reward_base_height: -0.0068
    --------------------------------------------------------------------------------
                    Total timesteps: 5111808
                        Iteration time: 0.51s
                            Total time: 26.84s
                                ETA: 25.8s

    ################################################################################
                        Learning iteration 52/101

                        Computation: 198717 steps/s (collection: 0.359s, learning 0.136s)
                Value function loss: 0.0035
                        Surrogate loss: -0.0062
                Mean action noise std: 0.83
                    Mean total reward: 3.71
                Mean episode length: 446.07
    Mean episode rew_reward_tracking_lin_vel: 0.2407
    Mean episode rew_reward_tracking_ang_vel: 0.0737
    Mean episode rew_reward_lin_vel_z: -0.0071
    Mean episode rew_reward_action_rate: -0.0350
    Mean episode rew_reward_similar_to_default: -0.0821
    Mean episode rew_reward_base_height: -0.0066
    --------------------------------------------------------------------------------
                    Total timesteps: 5210112
                        Iteration time: 0.49s
                            Total time: 27.33s
                                ETA: 25.3s

    ################################################################################
                        Learning iteration 53/101

                        Computation: 200679 steps/s (collection: 0.353s, learning 0.137s)
                Value function loss: 0.0026
                        Surrogate loss: -0.0055
                Mean action noise std: 0.83
                    Mean total reward: 4.05
                Mean episode length: 485.83
    Mean episode rew_reward_tracking_lin_vel: 0.2641
    Mean episode rew_reward_tracking_ang_vel: 0.0812
    Mean episode rew_reward_lin_vel_z: -0.0072
    Mean episode rew_reward_action_rate: -0.0393
    Mean episode rew_reward_similar_to_default: -0.0893
    Mean episode rew_reward_base_height: -0.0068
    --------------------------------------------------------------------------------
                    Total timesteps: 5308416
                        Iteration time: 0.49s
                            Total time: 27.82s
                                ETA: 24.7s

    ################################################################################
                        Learning iteration 54/101

                        Computation: 207934 steps/s (collection: 0.335s, learning 0.137s)
                Value function loss: 0.0026
                        Surrogate loss: -0.0061
                Mean action noise std: 0.83
                    Mean total reward: 4.07
                Mean episode length: 478.61
    Mean episode rew_reward_tracking_lin_vel: 0.2807
    Mean episode rew_reward_tracking_ang_vel: 0.0845
    Mean episode rew_reward_lin_vel_z: -0.0077
    Mean episode rew_reward_action_rate: -0.0411
    Mean episode rew_reward_similar_to_default: -0.0930
    Mean episode rew_reward_base_height: -0.0074
    --------------------------------------------------------------------------------
                    Total timesteps: 5406720
                        Iteration time: 0.47s
                            Total time: 28.30s
                                ETA: 24.2s

    ################################################################################
                        Learning iteration 55/101

                        Computation: 215529 steps/s (collection: 0.321s, learning 0.135s)
                Value function loss: 0.0024
                        Surrogate loss: -0.0051
                Mean action noise std: 0.82
                    Mean total reward: 3.81
                Mean episode length: 430.91
    Mean episode rew_reward_tracking_lin_vel: 0.2335
    Mean episode rew_reward_tracking_ang_vel: 0.0661
    Mean episode rew_reward_lin_vel_z: -0.0069
    Mean episode rew_reward_action_rate: -0.0319
    Mean episode rew_reward_similar_to_default: -0.0715
    Mean episode rew_reward_base_height: -0.0063
    --------------------------------------------------------------------------------
                    Total timesteps: 5505024
                        Iteration time: 0.46s
                            Total time: 28.75s
                                ETA: 23.6s

    ################################################################################
                        Learning iteration 56/101

                        Computation: 207124 steps/s (collection: 0.337s, learning 0.137s)
                Value function loss: 0.0018
                        Surrogate loss: -0.0040
                Mean action noise std: 0.82
                    Mean total reward: 4.61
                Mean episode length: 531.48
    Mean episode rew_reward_tracking_lin_vel: 0.4075
    Mean episode rew_reward_tracking_ang_vel: 0.1287
    Mean episode rew_reward_lin_vel_z: -0.0091
    Mean episode rew_reward_action_rate: -0.0635
    Mean episode rew_reward_similar_to_default: -0.1409
    Mean episode rew_reward_base_height: -0.0099
    --------------------------------------------------------------------------------
                    Total timesteps: 5603328
                        Iteration time: 0.47s
                            Total time: 29.23s
                                ETA: 23.1s

    ################################################################################
                        Learning iteration 57/101

                        Computation: 199003 steps/s (collection: 0.356s, learning 0.138s)
                Value function loss: 0.0018
                        Surrogate loss: -0.0038
                Mean action noise std: 0.82
                    Mean total reward: 6.82
                Mean episode length: 825.78
    Mean episode rew_reward_tracking_lin_vel: 0.4725
    Mean episode rew_reward_tracking_ang_vel: 0.1513
    Mean episode rew_reward_lin_vel_z: -0.0098
    Mean episode rew_reward_action_rate: -0.0746
    Mean episode rew_reward_similar_to_default: -0.1614
    Mean episode rew_reward_base_height: -0.0110
    --------------------------------------------------------------------------------
                    Total timesteps: 5701632
                        Iteration time: 0.49s
                            Total time: 29.72s
                                ETA: 22.5s

    ################################################################################
                        Learning iteration 58/101

                        Computation: 199370 steps/s (collection: 0.356s, learning 0.137s)
                Value function loss: 0.0024
                        Surrogate loss: -0.0030
                Mean action noise std: 0.82
                    Mean total reward: 7.77
                Mean episode length: 921.06
    Mean episode rew_reward_tracking_lin_vel: 0.4943
    Mean episode rew_reward_tracking_ang_vel: 0.1560
    Mean episode rew_reward_lin_vel_z: -0.0100
    Mean episode rew_reward_action_rate: -0.0768
    Mean episode rew_reward_similar_to_default: -0.1659
    Mean episode rew_reward_base_height: -0.0106
    --------------------------------------------------------------------------------
                    Total timesteps: 5799936
                        Iteration time: 0.49s
                            Total time: 30.21s
                                ETA: 22.0s

    ################################################################################
                        Learning iteration 59/101

                        Computation: 192994 steps/s (collection: 0.365s, learning 0.145s)
                Value function loss: 0.0023
                        Surrogate loss: -0.0041
                Mean action noise std: 0.82
                    Mean total reward: 8.40
                Mean episode length: 974.26
    Mean episode rew_reward_tracking_lin_vel: 0.5194
    Mean episode rew_reward_tracking_ang_vel: 0.1615
    Mean episode rew_reward_lin_vel_z: -0.0102
    Mean episode rew_reward_action_rate: -0.0789
    Mean episode rew_reward_similar_to_default: -0.1717
    Mean episode rew_reward_base_height: -0.0104
    --------------------------------------------------------------------------------
                    Total timesteps: 5898240
                        Iteration time: 0.51s
                            Total time: 30.72s
                                ETA: 21.5s

    ################################################################################
                        Learning iteration 60/101

                        Computation: 200636 steps/s (collection: 0.353s, learning 0.137s)
                Value function loss: 0.0026
                        Surrogate loss: -0.0040
                Mean action noise std: 0.82
                    Mean total reward: 8.51
                Mean episode length: 946.06
    Mean episode rew_reward_tracking_lin_vel: 0.5304
    Mean episode rew_reward_tracking_ang_vel: 0.1607
    Mean episode rew_reward_lin_vel_z: -0.0104
    Mean episode rew_reward_action_rate: -0.0781
    Mean episode rew_reward_similar_to_default: -0.1704
    Mean episode rew_reward_base_height: -0.0100
    --------------------------------------------------------------------------------
                    Total timesteps: 5996544
                        Iteration time: 0.49s
                            Total time: 31.21s
                                ETA: 21.0s

    ################################################################################
                        Learning iteration 61/101

                        Computation: 202769 steps/s (collection: 0.350s, learning 0.135s)
                Value function loss: 0.0024
                        Surrogate loss: -0.0023
                Mean action noise std: 0.83
                    Mean total reward: 8.97
                Mean episode length: 974.23
    Mean episode rew_reward_tracking_lin_vel: 0.5470
    Mean episode rew_reward_tracking_ang_vel: 0.1630
    Mean episode rew_reward_lin_vel_z: -0.0106
    Mean episode rew_reward_action_rate: -0.0787
    Mean episode rew_reward_similar_to_default: -0.1720
    Mean episode rew_reward_base_height: -0.0096
    --------------------------------------------------------------------------------
                    Total timesteps: 6094848
                        Iteration time: 0.48s
                            Total time: 31.70s
                                ETA: 20.5s

    ################################################################################
                        Learning iteration 62/101

                        Computation: 203921 steps/s (collection: 0.346s, learning 0.136s)
                Value function loss: 0.0017
                        Surrogate loss: -0.0021
                Mean action noise std: 0.83
                    Mean total reward: 9.26
                Mean episode length: 984.19
    Mean episode rew_reward_tracking_lin_vel: 0.5664
    Mean episode rew_reward_tracking_ang_vel: 0.1649
    Mean episode rew_reward_lin_vel_z: -0.0109
    Mean episode rew_reward_action_rate: -0.0792
    Mean episode rew_reward_similar_to_default: -0.1749
    Mean episode rew_reward_base_height: -0.0095
    --------------------------------------------------------------------------------
                    Total timesteps: 6193152
                        Iteration time: 0.48s
                            Total time: 32.18s
                                ETA: 19.9s

    ################################################################################
                        Learning iteration 63/101

                        Computation: 205205 steps/s (collection: 0.345s, learning 0.134s)
                Value function loss: 0.0016
                        Surrogate loss: -0.0015
                Mean action noise std: 0.83
                    Mean total reward: 9.58
                Mean episode length: 984.90
    Mean episode rew_reward_tracking_lin_vel: 0.5834
    Mean episode rew_reward_tracking_ang_vel: 0.1654
    Mean episode rew_reward_lin_vel_z: -0.0111
    Mean episode rew_reward_action_rate: -0.0791
    Mean episode rew_reward_similar_to_default: -0.1763
    Mean episode rew_reward_base_height: -0.0094
    --------------------------------------------------------------------------------
                    Total timesteps: 6291456
                        Iteration time: 0.48s
                            Total time: 32.66s
                                ETA: 19.4s

    ################################################################################
                        Learning iteration 64/101

                        Computation: 204657 steps/s (collection: 0.346s, learning 0.135s)
                Value function loss: 0.0014
                        Surrogate loss: -0.0037
                Mean action noise std: 0.82
                    Mean total reward: 9.60
                Mean episode length: 964.22
    Mean episode rew_reward_tracking_lin_vel: 0.5954
    Mean episode rew_reward_tracking_ang_vel: 0.1651
    Mean episode rew_reward_lin_vel_z: -0.0113
    Mean episode rew_reward_action_rate: -0.0788
    Mean episode rew_reward_similar_to_default: -0.1773
    Mean episode rew_reward_base_height: -0.0094
    --------------------------------------------------------------------------------
                    Total timesteps: 6389760
                        Iteration time: 0.48s
                            Total time: 33.14s
                                ETA: 18.9s

    ################################################################################
                        Learning iteration 65/101

                        Computation: 203366 steps/s (collection: 0.347s, learning 0.136s)
                Value function loss: 0.0008
                        Surrogate loss: -0.0031
                Mean action noise std: 0.82
                    Mean total reward: 10.14
                Mean episode length: 994.18
    Mean episode rew_reward_tracking_lin_vel: 0.6167
    Mean episode rew_reward_tracking_ang_vel: 0.1672
    Mean episode rew_reward_lin_vel_z: -0.0115
    Mean episode rew_reward_action_rate: -0.0795
    Mean episode rew_reward_similar_to_default: -0.1786
    Mean episode rew_reward_base_height: -0.0091
    --------------------------------------------------------------------------------
                    Total timesteps: 6488064
                        Iteration time: 0.48s
                            Total time: 33.62s
                                ETA: 18.3s

    ################################################################################
                        Learning iteration 66/101

                        Computation: 204939 steps/s (collection: 0.344s, learning 0.136s)
                Value function loss: 0.0008
                        Surrogate loss: -0.0001
                Mean action noise std: 0.81
                    Mean total reward: 10.09
                Mean episode length: 969.85
    Mean episode rew_reward_tracking_lin_vel: 0.6143
    Mean episode rew_reward_tracking_ang_vel: 0.1629
    Mean episode rew_reward_lin_vel_z: -0.0115
    Mean episode rew_reward_action_rate: -0.0777
    Mean episode rew_reward_similar_to_default: -0.1726
    Mean episode rew_reward_base_height: -0.0089
    --------------------------------------------------------------------------------
                    Total timesteps: 6586368
                        Iteration time: 0.48s
                            Total time: 34.10s
                                ETA: 17.8s

    ################################################################################
                        Learning iteration 67/101

                        Computation: 214299 steps/s (collection: 0.324s, learning 0.135s)
                Value function loss: 0.0009
                        Surrogate loss: -0.0040
                Mean action noise std: 0.81
                    Mean total reward: 10.08
                Mean episode length: 950.35
    Mean episode rew_reward_tracking_lin_vel: 0.6210
    Mean episode rew_reward_tracking_ang_vel: 0.1587
    Mean episode rew_reward_lin_vel_z: -0.0119
    Mean episode rew_reward_action_rate: -0.0760
    Mean episode rew_reward_similar_to_default: -0.1721
    Mean episode rew_reward_base_height: -0.0089
    --------------------------------------------------------------------------------
                    Total timesteps: 6684672
                        Iteration time: 0.46s
                            Total time: 34.56s
                                ETA: 17.3s

    ################################################################################
                        Learning iteration 68/101

                        Computation: 224797 steps/s (collection: 0.303s, learning 0.135s)
                Value function loss: 0.0008
                        Surrogate loss: -0.0024
                Mean action noise std: 0.80
                    Mean total reward: 9.96
                Mean episode length: 925.98
    Mean episode rew_reward_tracking_lin_vel: 0.6012
    Mean episode rew_reward_tracking_ang_vel: 0.1488
    Mean episode rew_reward_lin_vel_z: -0.0114
    Mean episode rew_reward_action_rate: -0.0712
    Mean episode rew_reward_similar_to_default: -0.1621
    Mean episode rew_reward_base_height: -0.0091
    --------------------------------------------------------------------------------
                    Total timesteps: 6782976
                        Iteration time: 0.44s
                            Total time: 35.00s
                                ETA: 16.7s

    ################################################################################
                        Learning iteration 69/101

                        Computation: 219185 steps/s (collection: 0.312s, learning 0.136s)
                Value function loss: 0.0011
                        Surrogate loss: -0.0038
                Mean action noise std: 0.80
                    Mean total reward: 9.80
                Mean episode length: 895.61
    Mean episode rew_reward_tracking_lin_vel: 0.4100
    Mean episode rew_reward_tracking_ang_vel: 0.0953
    Mean episode rew_reward_lin_vel_z: -0.0097
    Mean episode rew_reward_action_rate: -0.0463
    Mean episode rew_reward_similar_to_default: -0.1025
    Mean episode rew_reward_base_height: -0.0065
    --------------------------------------------------------------------------------
                    Total timesteps: 6881280
                        Iteration time: 0.45s
                            Total time: 35.45s
                                ETA: 16.2s

    ################################################################################
                        Learning iteration 70/101

                        Computation: 243687 steps/s (collection: 0.269s, learning 0.135s)
                Value function loss: 0.0007
                        Surrogate loss: -0.0013
                Mean action noise std: 0.79
                    Mean total reward: 9.67
                Mean episode length: 879.32
    Mean episode rew_reward_tracking_lin_vel: 0.4832
    Mean episode rew_reward_tracking_ang_vel: 0.0989
    Mean episode rew_reward_lin_vel_z: -0.0109
    Mean episode rew_reward_action_rate: -0.0509
    Mean episode rew_reward_similar_to_default: -0.1158
    Mean episode rew_reward_base_height: -0.0076
    --------------------------------------------------------------------------------
                    Total timesteps: 6979584
                        Iteration time: 0.40s
                            Total time: 35.85s
                                ETA: 15.7s

    ################################################################################
                        Learning iteration 71/101

                        Computation: 239126 steps/s (collection: 0.276s, learning 0.135s)
                Value function loss: 0.0007
                        Surrogate loss: -0.0017
                Mean action noise std: 0.79
                    Mean total reward: 9.51
                Mean episode length: 857.63
    Mean episode rew_reward_tracking_lin_vel: 0.4622
    Mean episode rew_reward_tracking_ang_vel: 0.0963
    Mean episode rew_reward_lin_vel_z: -0.0106
    Mean episode rew_reward_action_rate: -0.0487
    Mean episode rew_reward_similar_to_default: -0.1112
    Mean episode rew_reward_base_height: -0.0068
    --------------------------------------------------------------------------------
                    Total timesteps: 7077888
                        Iteration time: 0.41s
                            Total time: 36.26s
                                ETA: 15.1s

    ################################################################################
                        Learning iteration 72/101

                        Computation: 239294 steps/s (collection: 0.276s, learning 0.135s)
                Value function loss: 0.0006
                        Surrogate loss: -0.0023
                Mean action noise std: 0.78
                    Mean total reward: 9.50
                Mean episode length: 843.82
    Mean episode rew_reward_tracking_lin_vel: 0.6450
    Mean episode rew_reward_tracking_ang_vel: 0.1419
    Mean episode rew_reward_lin_vel_z: -0.0124
    Mean episode rew_reward_action_rate: -0.0690
    Mean episode rew_reward_similar_to_default: -0.1550
    Mean episode rew_reward_base_height: -0.0086
    --------------------------------------------------------------------------------
                    Total timesteps: 7176192
                        Iteration time: 0.41s
                            Total time: 36.67s
                                ETA: 14.6s

    ################################################################################
                        Learning iteration 73/101

                        Computation: 240887 steps/s (collection: 0.269s, learning 0.139s)
                Value function loss: 0.0006
                        Surrogate loss: -0.0023
                Mean action noise std: 0.78
                    Mean total reward: 9.36
                Mean episode length: 825.05
    Mean episode rew_reward_tracking_lin_vel: 0.4021
    Mean episode rew_reward_tracking_ang_vel: 0.0817
    Mean episode rew_reward_lin_vel_z: -0.0104
    Mean episode rew_reward_action_rate: -0.0413
    Mean episode rew_reward_similar_to_default: -0.0856
    Mean episode rew_reward_base_height: -0.0067
    --------------------------------------------------------------------------------
                    Total timesteps: 7274496
                        Iteration time: 0.41s
                            Total time: 37.08s
                                ETA: 14.0s

    ################################################################################
                        Learning iteration 74/101

                        Computation: 237971 steps/s (collection: 0.276s, learning 0.137s)
                Value function loss: 0.0007
                        Surrogate loss: -0.0022
                Mean action noise std: 0.77
                    Mean total reward: 9.21
                Mean episode length: 798.90
    Mean episode rew_reward_tracking_lin_vel: 0.4147
    Mean episode rew_reward_tracking_ang_vel: 0.0832
    Mean episode rew_reward_lin_vel_z: -0.0101
    Mean episode rew_reward_action_rate: -0.0418
    Mean episode rew_reward_similar_to_default: -0.0898
    Mean episode rew_reward_base_height: -0.0068
    --------------------------------------------------------------------------------
                    Total timesteps: 7372800
                        Iteration time: 0.41s
                            Total time: 37.49s
                                ETA: 13.5s

    ################################################################################
                        Learning iteration 75/101

                        Computation: 230043 steps/s (collection: 0.292s, learning 0.135s)
                Value function loss: 0.0007
                        Surrogate loss: -0.0016
                Mean action noise std: 0.77
                    Mean total reward: 9.33
                Mean episode length: 797.71
    Mean episode rew_reward_tracking_lin_vel: 0.6300
    Mean episode rew_reward_tracking_ang_vel: 0.1242
    Mean episode rew_reward_lin_vel_z: -0.0123
    Mean episode rew_reward_action_rate: -0.0641
    Mean episode rew_reward_similar_to_default: -0.1361
    Mean episode rew_reward_base_height: -0.0079
    --------------------------------------------------------------------------------
                    Total timesteps: 7471104
                        Iteration time: 0.43s
                            Total time: 37.92s
                                ETA: 13.0s

    ################################################################################
                        Learning iteration 76/101

                        Computation: 241081 steps/s (collection: 0.270s, learning 0.138s)
                Value function loss: 0.0003
                        Surrogate loss: -0.0046
                Mean action noise std: 0.76
                    Mean total reward: 9.43
                Mean episode length: 797.71
    Mean episode rew_reward_tracking_lin_vel: 0.7506
    Mean episode rew_reward_tracking_ang_vel: 0.1539
    Mean episode rew_reward_lin_vel_z: -0.0131
    Mean episode rew_reward_action_rate: -0.0755
    Mean episode rew_reward_similar_to_default: -0.1715
    Mean episode rew_reward_base_height: -0.0092
    --------------------------------------------------------------------------------
                    Total timesteps: 7569408
                        Iteration time: 0.41s
                            Total time: 38.33s
                                ETA: 12.4s

    ################################################################################
                        Learning iteration 77/101

                        Computation: 226469 steps/s (collection: 0.299s, learning 0.135s)
                Value function loss: 0.0009
                        Surrogate loss: -0.0020
                Mean action noise std: 0.76
                    Mean total reward: 9.50
                Mean episode length: 778.51
    Mean episode rew_reward_tracking_lin_vel: 0.6206
    Mean episode rew_reward_tracking_ang_vel: 0.1310
    Mean episode rew_reward_lin_vel_z: -0.0124
    Mean episode rew_reward_action_rate: -0.0629
    Mean episode rew_reward_similar_to_default: -0.1396
    Mean episode rew_reward_base_height: -0.0083
    --------------------------------------------------------------------------------
                    Total timesteps: 7667712
                        Iteration time: 0.43s
                            Total time: 38.76s
                                ETA: 11.9s

    ################################################################################
                        Learning iteration 78/101

                        Computation: 223195 steps/s (collection: 0.303s, learning 0.138s)
                Value function loss: 0.0002
                        Surrogate loss: -0.0030
                Mean action noise std: 0.75
                    Mean total reward: 9.91
                Mean episode length: 780.35
    Mean episode rew_reward_tracking_lin_vel: 0.7656
    Mean episode rew_reward_tracking_ang_vel: 0.1590
    Mean episode rew_reward_lin_vel_z: -0.0137
    Mean episode rew_reward_action_rate: -0.0773
    Mean episode rew_reward_similar_to_default: -0.1779
    Mean episode rew_reward_base_height: -0.0101
    --------------------------------------------------------------------------------
                    Total timesteps: 7766016
                        Iteration time: 0.44s
                            Total time: 39.20s
                                ETA: 11.4s

    ################################################################################
                        Learning iteration 79/101

                        Computation: 223833 steps/s (collection: 0.304s, learning 0.136s)
                Value function loss: 0.0005
                        Surrogate loss: -0.0012
                Mean action noise std: 0.74
                    Mean total reward: 10.86
                Mean episode length: 825.29
    Mean episode rew_reward_tracking_lin_vel: 0.7890
    Mean episode rew_reward_tracking_ang_vel: 0.1618
    Mean episode rew_reward_lin_vel_z: -0.0147
    Mean episode rew_reward_action_rate: -0.0792
    Mean episode rew_reward_similar_to_default: -0.1846
    Mean episode rew_reward_base_height: -0.0102
    --------------------------------------------------------------------------------
                    Total timesteps: 7864320
                        Iteration time: 0.44s
                            Total time: 39.64s
                                ETA: 10.9s

    ################################################################################
                        Learning iteration 80/101

                        Computation: 211715 steps/s (collection: 0.323s, learning 0.141s)
                Value function loss: 0.0004
                        Surrogate loss: -0.0045
                Mean action noise std: 0.73
                    Mean total reward: 11.86
                Mean episode length: 874.71
    Mean episode rew_reward_tracking_lin_vel: 0.8019
    Mean episode rew_reward_tracking_ang_vel: 0.1623
    Mean episode rew_reward_lin_vel_z: -0.0151
    Mean episode rew_reward_action_rate: -0.0792
    Mean episode rew_reward_similar_to_default: -0.1857
    Mean episode rew_reward_base_height: -0.0107
    --------------------------------------------------------------------------------
                    Total timesteps: 7962624
                        Iteration time: 0.46s
                            Total time: 40.11s
                                ETA: 10.4s

    ################################################################################
                        Learning iteration 81/101

                        Computation: 215523 steps/s (collection: 0.321s, learning 0.135s)
                Value function loss: 0.0003
                        Surrogate loss: -0.0042
                Mean action noise std: 0.72
                    Mean total reward: 13.11
                Mean episode length: 953.91
    Mean episode rew_reward_tracking_lin_vel: 0.8301
    Mean episode rew_reward_tracking_ang_vel: 0.1668
    Mean episode rew_reward_lin_vel_z: -0.0153
    Mean episode rew_reward_action_rate: -0.0820
    Mean episode rew_reward_similar_to_default: -0.1897
    Mean episode rew_reward_base_height: -0.0105
    --------------------------------------------------------------------------------
                    Total timesteps: 8060928
                        Iteration time: 0.46s
                            Total time: 40.56s
                                ETA: 9.9s

    ################################################################################
                        Learning iteration 82/101

                        Computation: 210662 steps/s (collection: 0.331s, learning 0.136s)
                Value function loss: 0.0004
                        Surrogate loss: 0.0002
                Mean action noise std: 0.72
                    Mean total reward: 13.86
                Mean episode length: 987.83
    Mean episode rew_reward_tracking_lin_vel: 0.8249
    Mean episode rew_reward_tracking_ang_vel: 0.1637
    Mean episode rew_reward_lin_vel_z: -0.0154
    Mean episode rew_reward_action_rate: -0.0808
    Mean episode rew_reward_similar_to_default: -0.1847
    Mean episode rew_reward_base_height: -0.0104
    --------------------------------------------------------------------------------
                    Total timesteps: 8159232
                        Iteration time: 0.47s
                            Total time: 41.03s
                                ETA: 9.4s

    ################################################################################
                        Learning iteration 83/101

                        Computation: 204276 steps/s (collection: 0.345s, learning 0.136s)
                Value function loss: 0.0002
                        Surrogate loss: -0.0027
                Mean action noise std: 0.71
                    Mean total reward: 14.29
                Mean episode length: 996.27
    Mean episode rew_reward_tracking_lin_vel: 0.8457
    Mean episode rew_reward_tracking_ang_vel: 0.1674
    Mean episode rew_reward_lin_vel_z: -0.0153
    Mean episode rew_reward_action_rate: -0.0823
    Mean episode rew_reward_similar_to_default: -0.1863
    Mean episode rew_reward_base_height: -0.0100
    --------------------------------------------------------------------------------
                    Total timesteps: 8257536
                        Iteration time: 0.48s
                            Total time: 41.51s
                                ETA: 8.9s

    ################################################################################
                        Learning iteration 84/101

                        Computation: 206207 steps/s (collection: 0.341s, learning 0.135s)
                Value function loss: 0.0003
                        Surrogate loss: -0.0001
                Mean action noise std: 0.70
                    Mean total reward: 14.74
                Mean episode length: 1001.00
    Mean episode rew_reward_tracking_lin_vel: 0.8590
    Mean episode rew_reward_tracking_ang_vel: 0.1683
    Mean episode rew_reward_lin_vel_z: -0.0155
    Mean episode rew_reward_action_rate: -0.0820
    Mean episode rew_reward_similar_to_default: -0.1844
    Mean episode rew_reward_base_height: -0.0095
    --------------------------------------------------------------------------------
                    Total timesteps: 8355840
                        Iteration time: 0.48s
                            Total time: 41.99s
                                ETA: 8.4s

    ################################################################################
                        Learning iteration 85/101

                        Computation: 203840 steps/s (collection: 0.346s, learning 0.136s)
                Value function loss: 0.0004
                        Surrogate loss: -0.0005
                Mean action noise std: 0.70
                    Mean total reward: 14.95
                Mean episode length: 1001.00
    Mean episode rew_reward_tracking_lin_vel: 0.8575
    Mean episode rew_reward_tracking_ang_vel: 0.1679
    Mean episode rew_reward_lin_vel_z: -0.0152
    Mean episode rew_reward_action_rate: -0.0814
    Mean episode rew_reward_similar_to_default: -0.1792
    Mean episode rew_reward_base_height: -0.0090
    --------------------------------------------------------------------------------
                    Total timesteps: 8454144
                        Iteration time: 0.48s
                            Total time: 42.47s
                                ETA: 7.9s

    ################################################################################
                        Learning iteration 86/101

                        Computation: 200489 steps/s (collection: 0.355s, learning 0.135s)
                Value function loss: 0.0005
                        Surrogate loss: -0.0019
                Mean action noise std: 0.69
                    Mean total reward: 15.14
                Mean episode length: 997.35
    Mean episode rew_reward_tracking_lin_vel: 0.8701
    Mean episode rew_reward_tracking_ang_vel: 0.1698
    Mean episode rew_reward_lin_vel_z: -0.0154
    Mean episode rew_reward_action_rate: -0.0816
    Mean episode rew_reward_similar_to_default: -0.1777
    Mean episode rew_reward_base_height: -0.0088
    --------------------------------------------------------------------------------
                    Total timesteps: 8552448
                        Iteration time: 0.49s
                            Total time: 42.96s
                                ETA: 7.4s

    ################################################################################
                        Learning iteration 87/101

                        Computation: 204132 steps/s (collection: 0.345s, learning 0.137s)
                Value function loss: 0.0004
                        Surrogate loss: -0.0025
                Mean action noise std: 0.69
                    Mean total reward: 15.23
                Mean episode length: 991.11
    Mean episode rew_reward_tracking_lin_vel: 0.8746
    Mean episode rew_reward_tracking_ang_vel: 0.1698
    Mean episode rew_reward_lin_vel_z: -0.0152
    Mean episode rew_reward_action_rate: -0.0814
    Mean episode rew_reward_similar_to_default: -0.1753
    Mean episode rew_reward_base_height: -0.0085
    --------------------------------------------------------------------------------
                    Total timesteps: 8650752
                        Iteration time: 0.48s
                            Total time: 43.44s
                                ETA: 6.9s

    ################################################################################
                        Learning iteration 88/101

                        Computation: 204420 steps/s (collection: 0.344s, learning 0.137s)
                Value function loss: 0.0002
                        Surrogate loss: -0.0058
                Mean action noise std: 0.68
                    Mean total reward: 15.58
                Mean episode length: 1001.00
    Mean episode rew_reward_tracking_lin_vel: 0.8863
    Mean episode rew_reward_tracking_ang_vel: 0.1710
    Mean episode rew_reward_lin_vel_z: -0.0153
    Mean episode rew_reward_action_rate: -0.0814
    Mean episode rew_reward_similar_to_default: -0.1743
    Mean episode rew_reward_base_height: -0.0084
    --------------------------------------------------------------------------------
                    Total timesteps: 8749056
                        Iteration time: 0.48s
                            Total time: 43.92s
                                ETA: 6.4s

    ################################################################################
                        Learning iteration 89/101

                        Computation: 198382 steps/s (collection: 0.360s, learning 0.136s)
                Value function loss: 0.0002
                        Surrogate loss: -0.0035
                Mean action noise std: 0.67
                    Mean total reward: 15.78
                Mean episode length: 1001.00
    Mean episode rew_reward_tracking_lin_vel: 0.8921
    Mean episode rew_reward_tracking_ang_vel: 0.1718
    Mean episode rew_reward_lin_vel_z: -0.0153
    Mean episode rew_reward_action_rate: -0.0814
    Mean episode rew_reward_similar_to_default: -0.1730
    Mean episode rew_reward_base_height: -0.0083
    --------------------------------------------------------------------------------
                    Total timesteps: 8847360
                        Iteration time: 0.50s
                            Total time: 44.42s
                                ETA: 5.9s

    ################################################################################
                        Learning iteration 90/101

                        Computation: 189722 steps/s (collection: 0.374s, learning 0.144s)
                Value function loss: 0.0005
                        Surrogate loss: -0.0032
                Mean action noise std: 0.67
                    Mean total reward: 15.89
                Mean episode length: 1000.61
    Mean episode rew_reward_tracking_lin_vel: 0.8928
    Mean episode rew_reward_tracking_ang_vel: 0.1715
    Mean episode rew_reward_lin_vel_z: -0.0154
    Mean episode rew_reward_action_rate: -0.0804
    Mean episode rew_reward_similar_to_default: -0.1712
    Mean episode rew_reward_base_height: -0.0082
    --------------------------------------------------------------------------------
                    Total timesteps: 8945664
                        Iteration time: 0.52s
                            Total time: 44.94s
                                ETA: 5.4s

    ################################################################################
                        Learning iteration 91/101

                        Computation: 183332 steps/s (collection: 0.384s, learning 0.153s)
                Value function loss: 0.0002
                        Surrogate loss: -0.0017
                Mean action noise std: 0.66
                    Mean total reward: 16.06
                Mean episode length: 1001.00
    Mean episode rew_reward_tracking_lin_vel: 0.9030
    Mean episode rew_reward_tracking_ang_vel: 0.1729
    Mean episode rew_reward_lin_vel_z: -0.0155
    Mean episode rew_reward_action_rate: -0.0807
    Mean episode rew_reward_similar_to_default: -0.1706
    Mean episode rew_reward_base_height: -0.0080
    --------------------------------------------------------------------------------
                    Total timesteps: 9043968
                        Iteration time: 0.54s
                            Total time: 45.47s
                                ETA: 4.9s

    ################################################################################
                        Learning iteration 92/101

                        Computation: 189728 steps/s (collection: 0.373s, learning 0.145s)
                Value function loss: 0.0001
                        Surrogate loss: -0.0016
                Mean action noise std: 0.65
                    Mean total reward: 16.24
                Mean episode length: 1001.00
    Mean episode rew_reward_tracking_lin_vel: 0.9113
    Mean episode rew_reward_tracking_ang_vel: 0.1731
    Mean episode rew_reward_lin_vel_z: -0.0154
    Mean episode rew_reward_action_rate: -0.0801
    Mean episode rew_reward_similar_to_default: -0.1702
    Mean episode rew_reward_base_height: -0.0080
    --------------------------------------------------------------------------------
                    Total timesteps: 9142272
                        Iteration time: 0.52s
                            Total time: 45.99s
                                ETA: 4.5s

    ################################################################################
                        Learning iteration 93/101

                        Computation: 187639 steps/s (collection: 0.375s, learning 0.148s)
                Value function loss: 0.0001
                        Surrogate loss: -0.0028
                Mean action noise std: 0.65
                    Mean total reward: 16.31
                Mean episode length: 1001.00
    Mean episode rew_reward_tracking_lin_vel: 0.9152
    Mean episode rew_reward_tracking_ang_vel: 0.1735
    Mean episode rew_reward_lin_vel_z: -0.0155
    Mean episode rew_reward_action_rate: -0.0798
    Mean episode rew_reward_similar_to_default: -0.1696
    Mean episode rew_reward_base_height: -0.0080
    --------------------------------------------------------------------------------
                    Total timesteps: 9240576
                        Iteration time: 0.52s
                            Total time: 46.51s
                                ETA: 4.0s

    ################################################################################
                        Learning iteration 94/101

                        Computation: 179795 steps/s (collection: 0.376s, learning 0.171s)
                Value function loss: 0.0005
                        Surrogate loss: 0.0006
                Mean action noise std: 0.65
                    Mean total reward: 16.27
                Mean episode length: 989.51
    Mean episode rew_reward_tracking_lin_vel: 0.9138
    Mean episode rew_reward_tracking_ang_vel: 0.1732
    Mean episode rew_reward_lin_vel_z: -0.0152
    Mean episode rew_reward_action_rate: -0.0785
    Mean episode rew_reward_similar_to_default: -0.1683
    Mean episode rew_reward_base_height: -0.0079
    --------------------------------------------------------------------------------
                    Total timesteps: 9338880
                        Iteration time: 0.55s
                            Total time: 47.06s
                                ETA: 3.5s

    ################################################################################
                        Learning iteration 95/101

                        Computation: 191495 steps/s (collection: 0.368s, learning 0.146s)
                Value function loss: 0.0001
                        Surrogate loss: -0.0060
                Mean action noise std: 0.64
                    Mean total reward: 16.53
                Mean episode length: 999.08
    Mean episode rew_reward_tracking_lin_vel: 0.9265
    Mean episode rew_reward_tracking_ang_vel: 0.1744
    Mean episode rew_reward_lin_vel_z: -0.0157
    Mean episode rew_reward_action_rate: -0.0787
    Mean episode rew_reward_similar_to_default: -0.1690
    Mean episode rew_reward_base_height: -0.0078
    --------------------------------------------------------------------------------
                    Total timesteps: 9437184
                        Iteration time: 0.51s
                            Total time: 47.57s
                                ETA: 3.0s

    ################################################################################
                        Learning iteration 96/101

                        Computation: 191643 steps/s (collection: 0.370s, learning 0.143s)
                Value function loss: 0.0003
                        Surrogate loss: -0.0027
                Mean action noise std: 0.63
                    Mean total reward: 16.60
                Mean episode length: 999.26
    Mean episode rew_reward_tracking_lin_vel: 0.9246
    Mean episode rew_reward_tracking_ang_vel: 0.1736
    Mean episode rew_reward_lin_vel_z: -0.0154
    Mean episode rew_reward_action_rate: -0.0776
    Mean episode rew_reward_similar_to_default: -0.1686
    Mean episode rew_reward_base_height: -0.0081
    --------------------------------------------------------------------------------
                    Total timesteps: 9535488
                        Iteration time: 0.51s
                            Total time: 48.09s
                                ETA: 2.5s

    ################################################################################
                        Learning iteration 97/101

                        Computation: 215778 steps/s (collection: 0.315s, learning 0.140s)
                Value function loss: 0.0001
                        Surrogate loss: -0.0035
                Mean action noise std: 0.62
                    Mean total reward: 16.67
                Mean episode length: 999.26
    Mean episode rew_reward_tracking_lin_vel: 0.9347
    Mean episode rew_reward_tracking_ang_vel: 0.1755
    Mean episode rew_reward_lin_vel_z: -0.0156
    Mean episode rew_reward_action_rate: -0.0779
    Mean episode rew_reward_similar_to_default: -0.1678
    Mean episode rew_reward_base_height: -0.0079
    --------------------------------------------------------------------------------
                    Total timesteps: 9633792
                        Iteration time: 0.46s
                            Total time: 48.54s
                                ETA: 2.0s

    ################################################################################
                        Learning iteration 98/101

                        Computation: 205445 steps/s (collection: 0.332s, learning 0.146s)
                Value function loss: 0.0001
                        Surrogate loss: -0.0061
                Mean action noise std: 0.61
                    Mean total reward: 16.79
                Mean episode length: 999.26
    Mean episode rew_reward_tracking_lin_vel: 0.9359
    Mean episode rew_reward_tracking_ang_vel: 0.1762
    Mean episode rew_reward_lin_vel_z: -0.0153
    Mean episode rew_reward_action_rate: -0.0761
    Mean episode rew_reward_similar_to_default: -0.1674
    Mean episode rew_reward_base_height: -0.0077
    --------------------------------------------------------------------------------
                    Total timesteps: 9732096
                        Iteration time: 0.48s
                            Total time: 49.02s
                                ETA: 1.5s

    ################################################################################
                        Learning iteration 99/101

                        Computation: 195821 steps/s (collection: 0.359s, learning 0.143s)
                Value function loss: 0.0001
                        Surrogate loss: -0.0013
                Mean action noise std: 0.60
                    Mean total reward: 16.97
                Mean episode length: 1001.00
    Mean episode rew_reward_tracking_lin_vel: 0.9404
    Mean episode rew_reward_tracking_ang_vel: 0.1767
    Mean episode rew_reward_lin_vel_z: -0.0153
    Mean episode rew_reward_action_rate: -0.0764
    Mean episode rew_reward_similar_to_default: -0.1678
    Mean episode rew_reward_base_height: -0.0080
    --------------------------------------------------------------------------------
                    Total timesteps: 9830400
                        Iteration time: 0.50s
                            Total time: 49.52s
                                ETA: 1.0s

    ################################################################################
                        Learning iteration 100/101

                        Computation: 196723 steps/s (collection: 0.359s, learning 0.141s)
                Value function loss: 0.0003
                        Surrogate loss: -0.0006
                Mean action noise std: 0.60
                    Mean total reward: 17.02
                Mean episode length: 998.39
    Mean episode rew_reward_tracking_lin_vel: 0.9393
    Mean episode rew_reward_tracking_ang_vel: 0.1766
    Mean episode rew_reward_lin_vel_z: -0.0152
    Mean episode rew_reward_action_rate: -0.0753
    Mean episode rew_reward_similar_to_default: -0.1669
    Mean episode rew_reward_base_height: -0.0079
    --------------------------------------------------------------------------------
                    Total timesteps: 9928704
                        Iteration time: 0.50s
                            Total time: 50.02s
                                ETA: 0.5s
