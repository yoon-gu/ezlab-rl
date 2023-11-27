# Verificition of EpidRLearn, exteme cases and sensitivity analysis of the reward
import os
import numpy as np
import env_
import model.plot_reward as plot_reward

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy
import torch as th



# Create log dir
log_dir = "temp/"
os.makedirs(log_dir, exist_ok=True)
SAVE_MODEL_PATH = "temp/models/"
os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

# Custom network init
# Custom actor (pi) and value function (vf) networks
# of one layer of size 128 each with Relu activation function
policy_kwargs = dict(activation_fn=th.nn.Tanh,
net_arch=[dict(pi=[128], vf=[128])])

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))
        return True




def train(problem_id, reward_id,  model_name, time_steps = 250_000):
    # run the model for the extreme cases and plot,save the results

    #problem_id = 0: 
    #problem_od = 1 
    problem_id = [problem_id]
    for problem in problem_id:
        print(problem)
        # train
        period = 'spring'
        envr = env_.Epidemic(problem_id = problem, reward_id=reward_id, period = period)
        env = Monitor(envr, log_dir) 

        # Logs will be saved in log_dir/monitor.csv
        env = Monitor(env, SAVE_MODEL_PATH)
        callback = SaveOnBestTrainingRewardCallback(check_freq=50_000, log_dir=log_dir)
        model = PPO('MlpPolicy', env, gamma=0.99,  verbose=1, policy_kwargs=policy_kwargs)
        model.learn(total_timesteps=int(time_steps), callback=callback)

        #model.save(SAVE_MODEL_PATH+"epidRLearn_Paper")
        model.save(SAVE_MODEL_PATH+model_name)

        plot_reward.plot_results(problem, [log_dir], time_steps, results_plotter.X_TIMESTEPS, "")
