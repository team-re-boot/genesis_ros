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
