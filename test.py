import os
import torch
import hydra
from hydra.utils import instantiate
from envs import SirEnvironment
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO, DQN, A2C, SAC, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    EveryNTimesteps,
    StopTrainingOnMaxEpisodes,
    StopTrainingOnNoModelImprovement,
    StopTrainingOnRewardThreshold,
    ProgressBarCallback
)

@hydra.main(version_base=None, config_path="conf", config_name="ppo")
def main(conf: DictConfig):
    print(conf.sir)
    train_env = SirEnvironment(**conf.sir)
    log_dir = "./sir_ppo_log"
    os.makedirs(log_dir, exist_ok=True)
    train_env = Monitor(train_env, log_dir)
    policy_kwargs = dict(
                            activation_fn=torch.nn.ReLU,
                            net_arch=[16, 32, 64, 16]
                        )
    model = PPO("MlpPolicy", train_env, verbose=0,
                policy_kwargs=policy_kwargs)

    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./checkpoints/',
                                             name_prefix='rl_model')
    callback = CallbackList([checkpoint_callback, ProgressBarCallback()])

    model.learn(total_timesteps=conf.n_steps, callback=callback)

if __name__ == '__main__':
    main()
