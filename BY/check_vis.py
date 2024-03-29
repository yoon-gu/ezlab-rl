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
import yaml

def seiar(y, t, beta, psi, nu, kappa, alpha, tau, p, eta, f, epsilon, q, delta):
            S, E, I, A = y
            Lambda = epsilon * E + (1 - q) * I + delta * A
            dSdt = -beta * S * Lambda - psi * nu * S
            dEdt = beta * S * Lambda - kappa * E
            dIdt = p * kappa * E - alpha * I - tau * I
            dAdt = (1 - p) * kappa * E - eta * A
            return [dSdt, dEdt, dIdt, dAdt]
        
class SeiarEnvironment:
    def __init__(self):
        self.psi = 0.7
        self.nu_daily_max = 500000
        self.nu_total_max = 5000000
        self.nu_min = 0.0
        self.kappa = 0.7143
        self.alpha = 0.1667
        self.tau = 0.0
        self.p = 0.667
        self.eta = 0.1667
        self.f = 0.999
        self.epsilon = 0.0
        self.q = 0.5
        self.delta = 0.5
        self.S0 = 50000000
        self.E0 = 0
        self.I0 = 1
        self.A0 = 0
        self.R0 = 1.9847
        self.beta = 1.32339796*1e-8      # R0 = 1.9847
        #self.beta = self.R0 / (self.S0 * ((self.epsilon / self.kappa) + ((1 - self.q) * self.p / self.alpha) + (self.delta * (1 - self.p) / self.eta)))
        self.tf = 180
        self.dt = 1.0
        self.time = 0
        self.days = [self.time]
        self.history = [[self.S0, self.E0, self.I0, self.A0]]
        self.nus = []
        self.rewards = []

    def reset(self):
        self.state = np.array([self.S0, self.E0, self.I0, self.A0])
        self.kappa = 0.7143
        self.alpha = 0.1667
        self.p = 0.667
        self.eta = 0.1667
        self.epsilon = 0.0
        self.q = 0.5
        self.delta = 0.5
        self.R0 = 1.9847
        self.beta = 1.32339796*1e-8      # R0 = 1.9847
        #self.beta = self.R0 / (self.S0 * ((self.epsilon / self.kappa) + ((1 - self.q) * self.p / self.alpha) + (self.delta * (1 - self.p) / self.eta)))
        self.dt = 1.0
        self.nus = []
        self.rewards = []
        self.histroy = [self.state]
        return self.state
    

    def step(self, action):
        nu = self.nu_min if action == 0 else self.nu_daily_max
        self.nus.append(nu)
        S0, E0, I0, A0 = self.state
        sol = odeint(seiar, [max(0, S0-nu), E0, I0, A0], np.linspace(0, self.dt, 101),
                    args=(self.beta, self.psi, 0, self.kappa, self.alpha, self.tau, self.p, self.eta, self.f, self.epsilon, self.q, self.delta))
        new_state = sol[-1, :]
        S, E, I, A = new_state
        self.state = new_state

        # reward case
        penalty = 1e-5*abs(max((0, sum(self.nus)-self.nu_total_max)))
        reward = - I
        reward = reward/1
        reward *= self.dt

        self.rewards.append(reward)
        self.days.append(self.time)
        self.history.append([S, E, I, A])

        done = True if self.time >= self.tf else False
        return (new_state, reward, done, {})


# model 실행
#agent.qnetwork_local.load_state_dict(model)
#agent.qnetwork_local.eval()

env = SeiarEnvironment()
state = env.reset()
max_t = 300
states = state
actions = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0,]
score = 0
dt = 1
actions_ = []
time_stamp = np.arange(0, max_t, dt)
for t in time_stamp:
    action = actions[t]
    #action = agent.act(state, eps=0.0)
    actions_.append(action)
    next_state, reward, done, _ = env.step(action)
    states = np.vstack((states, next_state))
    state = next_state
    score += reward
actionss = np.array(actions_)
time_stamp = np.append(time_stamp, max_t)



# For action plot
plt.figure(figsize=(8,8))
plt.subplot(3, 1, 1)
plt.plot(time_stamp, states[:,2], '.-r', label = 'I')
plt.ylabel('Infected')
plt.title(f'Control:{score}')
plt.xticks(color='w')

plt.subplot(3, 1, 2)
plt.plot(actionss*500000, '-k', label = 'Vaccine')
plt.ylabel('Vaccination')
#plt.ylim([0.0, 0.01])
plt.xticks(color='w')

plt.subplot(3, 1, 3)
plt.plot(time_stamp, states[:,0], '.-b', label = 'S')
plt.ylabel('Susceptible')
plt.savefig(f"all_prev.png", dpi=300)