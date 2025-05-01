from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from dataclass_wizard import YAMLWizard
from pathlib import Path


@dataclass
class Algorithm(YAMLWizard):
    class_name: str = "PPO"
    clip_param: float = 0.2
    desired_kl: float = 0.01
    entropy_coef: float = 0.01
    gamma: float = 0.99
    lam: float = 0.95
    learning_rate: float = 0.001
    max_grad_norm: float = 1.0
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    schedule: str = "adaptive"
    use_clipped_value_loss: bool = True
    value_loss_coef: float = 1.0

    @classmethod
    def safe_load(cls, path: Path) -> Any:
        if path.exists():
            return Algorithm().from_yaml_file(path)
        else:
            return Algorithm()


@dataclass
class Policy(YAMLWizard):
    activation: str = "elu"
    actor_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    init_noise_std: float = 1.0
    class_name: str = "ActorCritic"

    @classmethod
    def safe_load(cls, path: Path) -> Any:
        if path.exists():
            return Policy().from_yaml_file(path)
        else:
            return Policy()


@dataclass
class Runner(YAMLWizard):
    checkpoint: int = -1
    experiment_name: str = "genesis_ros_ppo"
    load_run: int = -1
    log_interval: int = 1
    max_iterations: int = 101
    record_interval: int = -1
    resume: bool = False
    resume_path: Optional[str] = None
    run_name: str = ""

    @classmethod
    def safe_load(cls, path: Path) -> Any:
        if path.exists():
            return Runner().from_yaml_file(path)
        else:
            return Runner()


@dataclass
class TrainConfig(YAMLWizard):
    algorithm: Algorithm = Algorithm()
    init_member_classes: Dict = field(default_factory=dict)
    policy: Policy = Policy()
    runner: Runner = Runner()
    runner_class_name: str = "OnPolicyRunner"
    num_steps_per_env: int = 24
    save_interval: int = 100
    empirical_normalization: Optional[str] = None
    seed: int = 1

    @classmethod
    def safe_load(cls, path: Path) -> Any:
        if path.exists():
            return TrainConfig().from_yaml_file(path)
        else:
            return TrainConfig()
