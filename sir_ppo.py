import os
import hydra
from hydra.utils import instantiate
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

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

sns.set_theme(style="whitegrid")

@hydra.main(version_base=None, config_path="conf", config_name="ppo")
def main(conf: DictConfig):
    train_env = instantiate(conf.sir)
    check_env(train_env)
    log_dir = "./sir_ppo_log"
    os.makedirs(log_dir, exist_ok=True)
    train_env = Monitor(train_env, log_dir)
    policy_kwargs = dict(
                            # activation_fn=torch.nn.ReLU,
                            # net_arch=[16, 32, 64, 16]
                        )
    model = PPO("MlpPolicy", train_env, verbose=0,
                policy_kwargs=policy_kwargs)

    eval_env = instantiate(conf.sir)
    eval_callback = EvalCallback(
            eval_env,
            eval_freq=300*500,
            verbose=0,
            warn=False,
            log_path='eval_log',
            best_model_save_path='best_model'
        )
    checkpoint_callback = CheckpointCallback(save_freq=300*500, save_path='./checkpoints/',
                                             name_prefix='rl_model')
    callback = CallbackList([checkpoint_callback, eval_callback, ProgressBarCallback()])

    model.learn(total_timesteps=conf.n_steps, callback=callback)

    os.makedirs('figures', exist_ok=True)
    df = pd.read_csv(f"{log_dir}/monitor.csv", skiprows=1)
    sns.lineplot(data=df.r)
    plt.xlabel('episodes')
    plt.ylabel('The cummulative return')
    plt.savefig(f"figures/reward.png")
    plt.close()

    # Visualize Controlled SIR Dynamics
    model = PPO.load(f'best_model/best_model.zip')
    state, _ = eval_env.reset()
    done = False
    while not done:
        action, _ = model.predict(state, deterministic=True)
        state, _, done, _, _ = eval_env.step(action)

    df = eval_env.dynamics
    best_reward = df.rewards.sum()
    # plt.figure(figsize=(8,8))
    # plt.subplot(3, 1, 1)
    # plt.title(f"R = {df.rewards.sum():,.4f}")
    # sns.lineplot(data=df, x='days', y='infected', color='r')
    # plt.xticks(color='w')
    # plt.subplot(3, 1, 2)
    # sns.lineplot(data=df, x='days', y='vaccines', color='k', drawstyle='steps-pre')
    # plt.ylim([-0.001, max(conf.sir.v_max * 1.2, 0.01)])
    # plt.xticks(color='w')
    # plt.subplot(3, 1, 3)
    # sns.lineplot(data=df, x='days', y='rewards', color='g')
    # plt.savefig(f"figures/best.png")
    # plt.close()
    
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(8, 8), sharex=True)
    plt.suptitle(f"R = {df.rewards.sum():,.4f}")
    sns.lineplot(data=df, x='days', y='susceptible', color='r', ax=axes[0])

    sns.lineplot(data=df, x='days', y='infected', color='r', ax=axes[1])

    sns.lineplot(data=df, x='days', y='vaccines', color='k', drawstyle='steps-pre', ax=axes[2])
    # axes[2].set_ylim([-conf.sir.nu_daily_max*0.1, max(conf.sir.nu_daily_max * 1.2, 0.01)])
    axes[2].set_ylim([-0.1, 1.1])

    sns.lineplot(data=df, x='days', y='rewards', color='g', ax=axes[3])

    plt.subplots_adjust(hspace=0.25)

    plt.savefig(f"figures/best.png")
    plt.close()

    best_checkpoint = ""
    max_val = -float('inf')
    for path in tqdm(os.listdir('checkpoints')):
        model = PPO.load(f'checkpoints/{path}')
        state, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, _, done, _, _ = eval_env.step(action)
        df = eval_env.dynamics

        cum_reward = df.rewards.sum()
        if cum_reward > max_val:
            max_val = cum_reward
            best_checkpoint = path

        # plt.figure(figsize=(8,8))
        # plt.subplot(3, 1, 1)
        # plt.title(f"R = {df.rewards.sum():,.4f}")
        # sns.lineplot(data=df, x='days', y='infected', color='r')
        # plt.xticks(color='w')
        # plt.subplot(3, 1, 2)
        # sns.lineplot(data=df, x='days', y='vaccines', color='k', drawstyle='steps-pre')
        # plt.ylim([-0.001, max(conf.sir.v_max * 1.1, 0.01)])
        # plt.xticks(color='w')
        # plt.subplot(3, 1, 3)
        # sns.lineplot(data=df, x='days', y='rewards', color='g')
        # plt.savefig(f"figures/{path.replace('.zip', '.png')}")
        # plt.close()

        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(8, 8), sharex=True)
        plt.suptitle(f"R = {df.rewards.sum():,.4f}")
        sns.lineplot(data=df, x='days', y='susceptible', color='r', ax=axes[0])

        sns.lineplot(data=df, x='days', y='infected', color='r', ax=axes[1])

        sns.lineplot(data=df, x='days', y='vaccines', color='k', drawstyle='steps-pre', ax=axes[2])
        # axes[2].set_ylim([-conf.sir.nu_daily_max*0.1, max(conf.sir.nu_daily_max * 1.2, 0.01)])
        axes[2].set_ylim([-0.1, 1.1])

        sns.lineplot(data=df, x='days', y='rewards', color='g', ax=axes[3])

        plt.subplots_adjust(hspace=0.25)

        plt.savefig(f"figures/{path.replace('.zip', '.png')}")
        plt.close()


    # # Visualize Controlled SIR Dynamics
    # if best_reward < max_val:
    #     best_reward = max_val
    #     model = PPO.load(f'checkpoints/{best_checkpoint}')
    #     state, _ = eval_env.reset()
    #     done = False
    #     while not done:
    #         action, _ = model.predict(state, deterministic=True)
    #         state, _, done, _, _ = eval_env.step(action)
    #     df = eval_env.dynamics
    #     # sns.lineplot(data=df, x='days', y='susceptible')
    #     plt.figure(figsize=(8,8))
    #     plt.subplot(3, 1, 1)
    #     plt.title(f"R = {df.rewards.sum():,.4f}")
    #     sns.lineplot(data=df, x='days', y='infected', color='r')
    #     plt.xticks(color='w')
    #     plt.subplot(3, 1, 2)
    #     sns.lineplot(data=df, x='days', y='vaccines', color='k', drawstyle='steps-pre')
    #     plt.ylim([-0.001, max(conf.sir.v_max * 1.1, 0.01)])
    #     plt.xticks(color='w')
    #     plt.subplot(3, 1, 3)
    #     sns.lineplot(data=df, x='days', y='rewards', color='g')
    #     plt.savefig(f"figures/best.png")
    #     plt.close()

if __name__ == '__main__':
    main()