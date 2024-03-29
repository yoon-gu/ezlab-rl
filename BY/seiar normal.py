# python sliar_rl.py n_episodes=10000 eps_start=0.0
# hydra list 호출
# python sliar_rl.py nu_actions="[0,1]" tau_actions="[0,1]"

import random
import hydra
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from dqn_agent import Agent
from omegaconf import DictConfig, OmegaConf
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from itertools import product
import pandas as pd
scale = 1

# nu: vaccine, tau: treatment, sigma: social distancing
@hydra.main(version_base=None, config_path="conf", config_name="seiar")
def main(conf : DictConfig) -> None:
    # # Control
    # nu_actions = conf.nu_actions
    # tau_actions = conf.tau_actions
    # sigma_actions = conf.sigma_actions

    # # Check the actions list
    # ACTIONS = list(product(nu_actions, tau_actions, sigma_actions))
    # #print(ACTIONS)
    # action_size = len(ACTIONS)
    # #print(f"actions size = {len(ACTIONS)}")

    def seiar(y, t, beta, psi, nu, kappa, alpha, tau, p, eta, f, epsilon, q, delta):
        S, E, I, A = y
        Lambda = epsilon * E + (1 - q) * I + delta * A
        # nu*S : the number of vaccination ==> nu * S : nu
        dSdt = -beta * S * Lambda - psi * nu * S
        dEdt = beta * S * Lambda - kappa * E
        dIdt = p * kappa * E - alpha * I - tau * I
        dAdt = (1 - p) * kappa * E - eta * A
        return [dSdt, dEdt, dIdt, dAdt]

    class SeiarEnvironment:
        def __init__(self):
            self.N = 5e7+1
            self.psi = conf.psi
            self.nu_daily_max = conf.nu_daily_max/self.N
            self.nu_total_max = conf.nu_total_max/self.N
            self.nu_min = 0.0
            self.kappa = conf.kappa
            self.alpha = conf.alpha
            self.tau = conf.tau
            self.p = conf.p
            self.eta = conf.eta
            self.f = conf.f
            self.epsilon = conf.epsilon
            self.q = conf.q
            self.delta = conf.delta
            self.S0 = conf.S0/self.N
            self.E0 = conf.E0/self.N
            self.I0 = conf.I0/self.N
            self.A0 = conf.A0/self.N
            self.R0 = 1.9847
            self.beta = 1.32339796*1e-8 * self.N      # R0 = 1.9847
            #self.beta = 1.60032*1e-8        # R0 = 2.4
            #self.beta = 9.33520*1e-9        # R0 = 1.3
            #self.beta = self.R0 / (self.S0 * ((self.epsilon / self.kappa) + ((1 - self.q) * self.p / self.alpha) + (self.delta * (1 - self.p) / self.eta)))
            self.tf = conf.tf
            self.dt = conf.dt
            self.scale = scale
            self.time = 0
            self.days = [self.time]
            self.history = [[conf.S0/self.N, conf.E0/self.N, conf.I0/self.N, conf.A0/self.N]]
            self.nus = []
            self.rewards = []

        def reset(self):
            self.state = np.array([self.S0, self.E0, self.I0, self.A0])
            self.kappa = conf.kappa
            self.alpha = conf.alpha
            self.p = conf.p
            self.eta = conf.eta
            self.epsilon = conf.epsilon
            self.q = conf.q
            self.delta = conf.delta
            self.R0 = 1.9847
            self.beta = 1.32339796*1e-8 * self.N
            #self.beta = 1.60032*1e-8        # R0 = 2.4
            #self.beta = 9.33520*1e-9        # R0 = 1.3
            #self.beta = self.R0 / (self.S0 * ((self.epsilon / self.kappa) + ((1 - self.q) * self.p / self.alpha) + (self.delta * (1 - self.p) / self.eta)))
            self.dt = conf.dt
            self.scale = scale
            self.nus = []
            self.rewards = []
            self.histroy = [self.state]
            return self.state
        

        def step(self, action):
            nu = self.nu_min if action == 0 else self.nu_daily_max
            self.nus.append(nu)
            S0, E0, I0, A0 = self.state
            sol = odeint(seiar, [max(0, S0-nu), E0, I0, A0], np.linspace(0, conf.dt, 101),
                        args=(self.beta, self.psi, 0, self.kappa, self.alpha, self.tau, self.p, self.eta, self.f, self.epsilon, self.q, self.delta))
            new_state = sol[-1, :]
            S, E, I, A = new_state
            self.state = new_state

            # check reward by initialization
            penalty = abs(max((0, sum(self.nus)-self.nu_total_max)))**2
            reward = - I - penalty
            reward = reward/self.scale
            reward *= self.dt

            self.rewards.append(reward)
            self.days.append(self.time)
            self.history.append([S, E, I, A])

            done = True if self.time >= self.tf else False
            return (new_state, reward, done, {})


    plt.rcParams['figure.figsize'] = (8, 4.5)

    # 1. Without Control
    env = SeiarEnvironment()
    state = env.reset()
    max_t = conf.tf
    states = state
    actions = []
    time_stamp = np.arange(0, max_t, conf.dt)
    for t in time_stamp:
        action = 0
        actions.append(action)
        next_state, reward, done, _ = env.step(action)
        states = np.vstack((states, next_state))
        state = next_state
    time_stamp = np.append(time_stamp, max_t)

    # Plot w/o control
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(time_stamp, states[:,0]*(5e7+1), '.-b', label = 'S')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(time_stamp, states[:,1]*(5e7+1), '.-y', label = 'L')
    ax2.plot(time_stamp, states[:,2]*(5e7+1), '.-r', label = 'I')
    ax2.plot(time_stamp, states[:,3]*(5e7+1), '.-g', label = 'A')
    ax2.legend(loc='lower right')
    plt.grid()
    plt.legend()
    plt.title('SLIAR model without control')
    plt.xlabel('day')
    plt.savefig('SLIAR_wo_control'+str(conf.dt)+'.png', dpi=300)
    plt.show(block=False)


    # 2. Train DQN Agent
    env = SeiarEnvironment()
    action_size = 2
    # seed = 0 : 고정
    agent = Agent(state_size=4, action_size=action_size, seed=0, scale=scale)
    # ## Parameters
    # n_episodes=conf.n_episodes
    # max_t=conf.tf
    # eps_start=conf.eps_start
    # eps_end=conf.eps_end
    # eps_decay=conf.eps_decay

    # ## Loop to learn
    # losss = []
    # max_scores=[]
    # scores = []                        # list containing scores from each episode
    # scores_window = deque(maxlen=1000)  # last 100 scores
    # ACTIONS_window = []
    # eps = eps_start                    # initialize epsilon
    # eps_window = []
    # for i_episode in range(1, n_episodes+1):
    #     state = env.reset()
    #     states = state
    #     score = 0
    #     actions = []
    #     time_stamp = np.arange(0, max_t, conf.dt)
    #     for t in time_stamp:
    #         action = agent.act(state, eps)
    #         actions.append(action)
    #         next_state, reward, done, _ = env.step(action)
    #         agent.step(state, action, reward, next_state, done)
    #         states = np.vstack((states, next_state))
    #         state = next_state
    #         score += reward
    #         if done:
    #             break
    #     scores_window.append(score)       # save most recent score
    #     scores.append(score)              # save most recent score
    #     ACTIONS_window.append(actions)   # save actions
    #     eps = max(eps_end, eps_decay*eps) # decrease epsilon
    #     eps_window.append(eps)
    #     max_sc = max(scores)

    #     print('\rEpisode {}\tAverage Score: {:,.2f}'.format(i_episode, np.mean(scores_window)), end="")
    #     print('\rEpisode {}\tNow Score: {:,.2f}'.format(i_episode, score), end="")

    #     if i_episode % 1000 == 0:
    #         print('\rEpisode {}\tAverage Score: {:,.2f}'.format(i_episode, np.mean(scores_window)))
    #         print(np.array(actions)[:5], eps)
    #         torch.save(agent.qnetwork_local.state_dict(), 'checkpoint'+str(i_episode)+'.pth')
    

    #     if i_episode % 1000 == 0:
    #         plt.clf()
    #         plt.plot(scores)
    #         plt.grid()
    #         plt.ylabel('cumulative future reward')
    #         plt.xlabel('episode')
    #         plt.savefig(f'SLIAR_score_{i_episode}.png', dpi=300)
    #         plt.show(block=False)

    #         plt.clf()
    #         plt.plot(eps_window)
    #         plt.grid()
    #         plt.title('The change of epsilon'+str(eps_start))
    #         plt.ylabel('epsilon')
    #         plt.xlabel('episode')
    #         plt.savefig(f'epsilon_{i_episode}.png', dpi=300)
    #         plt.show(block=False)        


    # torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')

    # actions_window = np.array(ACTIONS_window)
    # df1 = pd.DataFrame(actions_window)
    # df1.to_csv('ACTIONS.csv')   
    
    # scoress = np.array(scores)
    # df2 = pd.DataFrame(scoress)
    # df2.to_csv('scores.csv')
    
    # plt.clf()
    # plt.plot(scores)
    # plt.grid()
    # plt.ylabel('cumulative future reward')
    # plt.xlabel('episode')
    # plt.savefig('SLIAR_score.png', dpi=300)
    # plt.show(block=False)

    # plt.clf()
    # plt.plot(eps_window)
    # plt.grid()
    # plt.title('The change of epsilon'+str(eps_start))
    # plt.ylabel('epsilon')
    # plt.xlabel('episode')
    # plt.savefig('epsilon.png', dpi=300)
    # plt.show(block=False)


    # 3. Visualize Controlled SIR Dynamics
    # for homemac : boyeonkim
    os.chdir("/Users/boyeonkim/research/ezlab-rl/BY")
    # 현재 작업 경로 불러오기
    npath = os.getcwd()
    # 작업 경로 추가
    PATH = npath + f'/outputs/2023-11-09/04-03-12'
    # 추가한 경로로 변경
    os.chdir(PATH)
    # 폴더만들기
    os.makedirs(f'{PATH}/figures', exist_ok=True) 
    agent.qnetwork_local.load_state_dict(torch.load(f'checkpoint.pth'))

    env = SeiarEnvironment()
    state = env.reset()
    max_t = 300
    states = state
    actions = []
    score = 0
    time_stamp = np.arange(0, max_t, conf.dt)
    for t in time_stamp:
        action = agent.act(state, eps=0.0)
        actions = np.append(action)
        next_state, reward, done, _ = env.step(action)
        states = np.vstack((states, next_state))
        state = next_state
        score += reward
    actions_ = np.array(actions)
    time_stamp = np.append(time_stamp, max_t)

    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(time_stamp, states[:,0]*(5e7+1), '.-b', label = 'S')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(time_stamp, states[:,1]*(5e7+1), '.-y', label = 'L')
    ax2.plot(time_stamp, states[:,2]*(5e7+1), '.-r', label = 'I')
    ax2.plot(time_stamp, states[:,3]*(5e7+1), '.-g', label = 'A')
    ax2.legend(loc='lower right')
    plt.grid()
    plt.legend()
    plt.title('SLIAR model with control')
    plt.xlabel('day')
    plt.savefig(f"SLIAR_w_control.png", dpi=300)

    # For action plot
    plt.clf()
    plt.plot(actions_*750000, 'k+--', label = 'Vaccine')
    plt.grid()
    plt.legend()
    plt.title(f'Control:{score}')
    plt.savefig(f"control.png", dpi=300)

    plt.figure(figsize=(8,8))
    plt.subplot(3, 1, 1)
    plt.plot(time_stamp, states[:,2]*(5e7+1), '.-r', label = 'I')
    plt.ylabel('Infected')
    plt.title(f'Control:{score}')
    plt.xticks(color='w')
    
    plt.subplot(3, 1, 2)
    plt.plot(actions_*750000, '-k', label = 'Vaccine')
    plt.ylabel('Vaccination')
    #plt.ylim([0.0, 0.01])
    plt.xticks(color='w')
    
    plt.subplot(3, 1, 3)
    plt.plot(time_stamp, states[:,0]*(5e7+1), '.-b', label = 'S')
    plt.ylabel('Susceptible')
    plt.savefig(f"all.png", dpi=300)



if __name__ == '__main__':
    main()