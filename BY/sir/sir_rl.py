import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from dqn_agent import Agent
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def sir(y, t, beta, gamma, u):

## SIR setting
## ds = -beta * S * I - u * S, (u = vaccination rate <= 1)
## dI = beta * S * I - gamma * I
## dR = gamma * I 
## u = action

    S, I = y
    dydt = np.array([-beta * S * I - u * S, beta * S * I - gamma * I])
    return dydt


class SirEnvironment:
    def __init__(self, S0=5e7/(5e7+1), I0=1/(5e7+1)):
        # S0 = 5e7
        # I0 = 1
        # R0 = 0
        # N = S0+I0
        # R_0 = 1.9847
        # beta = R_0 * N / gamma
        # gamma = 1e-1

        self.state = np.array([S0, I0])
        self.R0 = 1.9847
        self.gamma = 0.12
        self.beta = self.R0 * self.gamma / (S0+I0) 
        self.beta = self.beta * (S0+I0)
        self.nu_daily_max = 750000/(S0+I0)
        self.nu_total_max = 25000000/(S0+I0)
        self.nu_min = 0.0
        self.nus = []

    def reset(self, S0=5e7/(5e7+1), I0=1/(5e7+1)):
        self.state = np.array([S0, I0])
        self.R0 = 1.9847
        self.gamma = 0.12
        self.beta = self.R0 * self.gamma / (S0+I0) 
        self.beta = self.beta * (S0+I0)
        self.nu_daily_max = 750000/(S0+I0)
        self.nu_total_max = 25000000/(S0+I0)
        self.nu_min = 0.0
        self.nus = []
        return self.state

    def step(self, action):
        nu = self.nu_min if action == 0 else self.nu_daily_max
        self.nus.append(nu)
        S0, I0 = self.state
        # sol = odeint(df, initial, dt, args = (beta, gamma, action))
        sol = odeint(sir, [max(0, S0-nu), I0], np.linspace(0, 1, 101), args=(self.beta, self.gamma, 0))
        # 계산하고 제일 끝  state가 new
        new_state = sol[-1, :]
        # new state
        S, I = new_state
        # state update
        self.state = new_state

        # reward case
        penalty = abs(max((0, sum(self.nus)-self.nu_total_max))) * 5e7
        reward = - I - penalty
        
        # new_state[1] : I < 1.0 이면 멈춤
        done = True if new_state[1] < 1.0 else False
        return (new_state, reward, done, 0)


plt.rcParams['figure.figsize'] = (8, 4.5)

# 1. Without Control
env = SirEnvironment()
state = env.reset()
# t = 300days
max_t = 300
states = state
# action은 없는 상태 why? without control 이니까
actions = []
for t in range(max_t):
    action = 0
    #print(states)
    next_state, reward, done, _ = env.step(action)
    #print(env.step(action))
    # np.vstack : 배열을 세로로 결합, 요소(열) 개수가 일치해야함, 행은 상관 없음
    # np.hstack : 배열을 가로로 결합, 행이 일치해야함, 열 상관없음.
    states = np.vstack((states, next_state))
    #print(states)
    state = next_state
    #print(state)

plt.clf()
fig, ax1 = plt.subplots()
ax1.plot(range(max_t+1), states[:,0].flatten() * (5e7+1), '.-b', label = 'S')
ax1.legend(loc = 'upper left')
ax2 = ax1.twinx()
ax2.plot(range(max_t+1), states[:,1].flatten() * (5e7+1), '.-r', label = 'I')
ax2.legend(loc = 'lower right')
plt.grid()
plt.title('SIR model without control')
plt.xlabel('day')
plt.savefig('SIR_wo_control_normal.png', dpi=300)
plt.show(block=False)

#######################################################################################################
# 2. Train DQN Agent
env = SirEnvironment()
# action | 0 : no vacc. 1 : vacc.
agent = Agent(state_size=2, action_size=2, seed=0, scale=1)
## Parameters
n_episodes=6000
eps_start=1.0 # Too large epsilon for a stable learning
eps_end=0.001
eps_decay=0.99

## Loop to learn
scores = []                        # list containing scores from each episode
scores_window = deque(maxlen=100)  # last 100 scores (replay bufferr)
eps = eps_start                    # initialize epsilon
for i_episode in range(1, n_episodes+1):
    state = env.reset()
    score = 0
    actions = []
    for t in range(max_t):
        # epsilon - greedy로 action 탐색 (policy)
        action = agent.act(state, eps)
        actions.append(action)
        # Taking action
        next_state, reward, done, _ = env.step(action)
        # Store transitions in replay memory D
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward

        if done:
            break 
    # replay buffer
    scores_window.append(score)       # save most recent score
    scores.append(score)              # save most recent score
    eps = max(eps_end, eps_decay*eps) # decrease epsilon

    
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
    if i_episode % 500 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        print(np.array(actions)[:5], eps)
    if np.mean(scores_window)>=200.0:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
        break

# 학습 다하고 저장!
torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')


plt.clf()
plt.plot(scores)
plt.grid()
plt.ylabel('cumulative future reward')
plt.xlabel('episode')
plt.savefig('SIR_score.png', dpi=300)
plt.show(block=False)

#######################################################################################################
# 3. Visualize Controlled SIR Dynamics
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
env = SirEnvironment()
max_t = 300
state = env.reset()
states = state
actions = []
for t in range(max_t):
    # from dqn_agent import Agent
    action = agent.act(state, eps=0.0)
    # optimal action
    actions = np.append(actions, action)
    next_state, reward, done, _ = env.step(action)
    agent.step(state, action, reward, next_state, done)
    states = np.vstack((states, next_state))
    state = next_state

plt.clf()
fig, ax1 = plt.subplots()
ax1.plot(range(max_t+1), states[:,0].flatten() * (5e7+1), '.-b', label = 'S')
ax1.legend(loc = 'upper left')
ax2 = ax1.twinx()
ax2.plot(range(max_t+1), states[:,1].flatten() * (5e7+1), '.-r', label = 'I')
ax2.legend(loc = 'lower right')
plt.grid()
plt.title('SIR model without control')
plt.xlabel('day')
plt.savefig('SIR_w_control.png', dpi=300)
plt.show(block=False)

plt.clf()
plt.plot(range(max_t), actions * 750000, '.-k')
plt.grid()
plt.title('Vaccine Control')
plt.xlabel('day')
plt.savefig('SIR_control_u.png', dpi=300)
plt.show(block=False)