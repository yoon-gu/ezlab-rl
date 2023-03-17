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
    mean_reward, std_reward = evaluate_policy(model, train_env, n_eval_episodes=100)
    print("Before:")
    print(f"\tmean_reward:{mean_reward:,.2f} +/- {std_reward:.2f}")

    eval_env = instantiate(conf.sir)
    eval_callback = EvalCallback(
            eval_env,
            eval_freq=1000,
            verbose=0,
            warn=False,
            log_path='eval_log',
            best_model_save_path='best_model'
        )
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./checkpoints/',
                                             name_prefix='rl_model')
    callback = CallbackList([checkpoint_callback, eval_callback, ProgressBarCallback()])

    model.learn(total_timesteps=conf.n_steps, callback=callback)
    mean_reward, std_reward = evaluate_policy(model, train_env, n_eval_episodes=100)
    print("After:")
    print(f"\tmean_reward:{mean_reward:,.2f} +/- {std_reward:.2f}")

    os.makedirs('figures', exist_ok=True)
    df = pd.read_csv(f"{log_dir}/monitor.csv", skiprows=1)
    sns.lineplot(data=df.r)
    plt.xlabel('episodes')
    plt.ylabel('The cummulative return')
    plt.savefig(f"figures/reward.png")
    plt.close()

    # Visualize Controlled SIR Dynamics
    state = eval_env.reset()
    done = False
    while not done:
        action, _ = model.predict(state)
        state, _, done, _ = eval_env.step(action)

    df = eval_env.dynamics
    sns.lineplot(data=df, x='days', y='susceptible', marker=".")
    sns.lineplot(data=df, x='days', y='infected', marker=".")
    ax2 = plt.twinx()
    sns.lineplot(data=df, x='days', y='vaccines', ax=ax2, marker="o", color="g")
    plt.grid()
    plt.title(f"R = {df.rewards.sum():4.2f}")
    plt.savefig(f"figures/best.png")
    plt.close()

    for path in tqdm(os.listdir('checkpoints')):
        model = PPO.load(f'checkpoints/{path}')
        state = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, _, done, _ = eval_env.step(action)
        df = eval_env.dynamics
        fig, (a1, a2) = plt.subplots(2, 1)
        sns.lineplot(ax=a1, data=df, x='days', y='susceptible', marker=".")
        sns.lineplot(ax=a1, data=df, x='days', y='infected', marker=".")
        ax2 = a1.twinx()
        sns.lineplot(data=df, x='days', y='vaccines', ax=ax2, marker="o", color="g")
        plt.grid()
        sns.lineplot(ax=a2, data=df, x='days', y='rewards', marker="^")
        a2.set_ylim([-80, 10])
        plt.title(f"R = {df.rewards.sum():4.2f} ({int(path.split('_')[2]):,})")
        plt.savefig(f"figures/{path.replace('.zip', '.png')}")
        plt.close()
if __name__ == '__main__':
    main()