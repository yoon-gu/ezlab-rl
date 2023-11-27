import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
#net_arch=[dict(pi=[32, 32], vf=[32, 32])])
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



def plot_reward_cases(reward_id, list_of_models):

    """
    reward_id: int, to add on the label
    list_of_models: list of dfs, that has the exposed age groups for each of the different models. to be used for plotting and comparison
    
    """

    N = 566_994.0 + 1528112.0 + 279444.0


    # autumn period fitted parameters
    unreported_0_19_p2 = 0.9755026399435202
    unreported_20_69_p2 = 0.4763668186715981
    unreported_70_p2 = 0.01
    exposed_rate = 5.1

    colors = ['blue', 'green', 'red']
    names = ['EpidRlean', 'Libin', 'Ohi']
    fig = plt.figure()
    for j, c, n in zip(range(len(colors)), range(len(list_of_models)), range(len(names))):
        print(colors[c])
        exposed_0_19 = list_of_models[j].exposed_0_19
        exposed_19_69 = list_of_models[j].exposed_19_69
        exposed_70 = list_of_models[j].exposed_70
        total = [None] * len(exposed_0_19)

        # Plotting the deterministcs vs EpideRlearn + the rewards for each

        for i in range(len(exposed_19_69)):
            
            total[i] = np.array(exposed_0_19[i])/N *  1/exposed_rate * (1- unreported_0_19_p2) + np.array(exposed_0_19[i])/N *  1/exposed_rate * (unreported_0_19_p2)\
            + np.array(exposed_19_69[i])/N *  1/exposed_rate * (1- unreported_20_69_p2) + np.array(exposed_19_69[i])/N *  1/exposed_rate * (unreported_20_69_p2)\
                + np.array(exposed_70[i])/N *  1/exposed_rate * (1- unreported_70_p2) + np.array(exposed_70[i])/N *  1/exposed_rate * (unreported_70_p2)

        
        
        plt.plot(total, label=names[n], color = colors[c])
    plt.gcf().set_size_inches(8, 7)
    plt.legend(loc=1)
    plt.ylabel("Population %")
    plt.xlabel("Weeks")
    plt.title("Reward Comparison")
    plt.savefig("Figures/Comparison_for_various_papers_rewardid_{}.svg".format(reward_id), format="svg")
    plt.savefig("Figures/Comparison_for_various_papers_rewardid_{}.png".format(reward_id))
    plt.show()



def evaluate_rewards(model_id, num_episodes, reward_id, problem_id, period):
    """
    Evaluate the RL agent

    model:
    num_episodes: int
    reward_id: int
    period: string
    print: boolean

    """
    # This function will only work for a single Environment


    for m_id, r_id in zip(model_id, reward_id):
        
        print(m_id, r_id)

        if m_id == 0: 
            
            model = PPO.load("temp/models/epidRLearn_NEW")
        else:
            print(m_id)
            model = PPO.load("temp/models/reward_id_{}.zip".format(m_id))
        
        envr = env_.Epidemic(problem_id, r_id, period=period)

        for i in range(num_episodes):
            exposed_0_19 = []
            exposed_19_69 = []
            exposed_70 = []
                
            episode_rewards = []
            actions = []


            done = False
            obs = envr.reset()

            while not done:
                # _states are only useful when using LSTM policies

                action, _states = model.predict(obs)
                actions.append(action)
                # here, action,a rewards and dones are arrays
                # because we are using vectorized env
                obs, reward, done, info = envr.step(action)
                #print(reward)

                
                exposed_0_19.append(obs[3])
                exposed_19_69.append(obs[4])
                exposed_70.append(obs[5])

                episode_rewards.append(reward)

            log_dir = "Analysis_Files/"
            os.makedirs(log_dir, exist_ok=True)

            exposed_dictionary = {'exposed_0_19':exposed_0_19, 'exposed_19_69':exposed_19_69, 'exposed_70':exposed_70 }
            exposed = pd.DataFrame(exposed_dictionary)
            exposed.to_csv("Analysis_Files/exposed_model{}.csv".format(m_id))

                
        #envr.render()

    
    return None


def train(problem_id, reward_id,  time_steps = 250_000):
    # run the model for the extreme cases and plot,save the results

    #problem_id = 0: 
    #problem_od = 1 
    
    for reward in reward_id:
        print(reward)
        # train
        period = 'spring'
        envr = env_.Epidemic(problem_id = problem_id, reward_id=reward, period = period)
        env = Monitor(envr, log_dir) 

        # Logs will be saved in log_dir/monitor.csv
        env = Monitor(env, SAVE_MODEL_PATH)
        callback = SaveOnBestTrainingRewardCallback(check_freq=50_000, log_dir=log_dir)
        model = PPO('MlpPolicy', env, gamma=0.99,  verbose=1, policy_kwargs=policy_kwargs)
        model.learn(total_timesteps=int(time_steps), callback=callback)

        model.save(SAVE_MODEL_PATH+"reward_id_{}".format(reward))

        plot_reward.plot_results(reward, [log_dir], time_steps, results_plotter.X_TIMESTEPS, "")
