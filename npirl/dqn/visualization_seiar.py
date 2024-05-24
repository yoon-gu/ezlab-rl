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
from scipy.io import loadmat
import time
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from itertools import product
import pandas as pd
import yaml


# for homemac : boyeonkim
os.chdir("/Users/boyeonkim/research/ezlab-rl/npirl")
# 현재 작업 경로 불러오기
npath = os.getcwd()
# model 불러오기
model = torch.load(f"checkpoint.pth")
# # yaml문서 불러오기
# PATH = PATH + '/.hydra'
# os.chdir(PATH)
# with open('config.yaml') as f:
#     yaml_dir = yaml.load(f,Loader=yaml.FullLoader)
# nu_actions, tau_actions, sigma_actions, dt = yaml_dir['nu_actions'], yaml_dir['tau_actions'], yaml_dir['sigma_actions'], yaml_dir['dt']

# actions_str = list(product(nu_str, tau_str, sigma_str))
# Environment
action_size = 4
agent = Agent(state_size=126, action_size=action_size, seed=0, scale=1)
        
## Load the data
current_directory = os.getcwd()
M = loadmat(f'{current_directory}/dqn/params.mat')
MV = loadmat(f'{current_directory}/dqn/mv.mat')
BE = loadmat(f'{current_directory}/dqn/result4init.mat')
result = loadmat('dqn/result.mat')

#@jit
def update_vaccine(vac_1st, vac_2nd, ind, neg_flag_S, neg_flag_V1, neg_flag_S_hist, neg_flag_V1_hist):
    if np.any(neg_flag_S):
        # Find the vaccination number for age groups having negative states
        vac_num_neg_state = np.sum(vac_1st[ind:, neg_flag_S], axis=1)

        # Compute ratio between other age groups
        neg_flag_S_hist = neg_flag_S_hist | neg_flag_S
        ratio = vac_1st[ind:, ~neg_flag_S_hist] / np.sum(vac_1st[ind:, ~neg_flag_S_hist], axis=1, keepdims=True)

        # Compute vaccination number using the ratio
        vac = vac_num_neg_state[:, np.newaxis] * ratio

        # Deal with NaNs resulting from division by zero
        vac[np.isnan(vac)] = 0

        # Update vaccination number for 1st dose
        vac_1st[ind:, ~neg_flag_S_hist] += vac
        vac_1st[ind:, neg_flag_S_hist] = 0

    # Update 2nd vaccination number
    if np.any(neg_flag_V1):
        # Find the vaccination number for age groups having negative states
        vac_num_neg_state = np.sum(vac_2nd[ind:, neg_flag_V1], axis=1)

        # Compute ratio between other age groups
        neg_flag_V1_hist = neg_flag_V1_hist | neg_flag_V1
        ratio = vac_2nd[ind:, ~neg_flag_V1_hist] / np.sum(vac_2nd[ind:, ~neg_flag_V1_hist], axis=1, keepdims=True)

        # Compute vaccination number using the ratio
        vac = vac_num_neg_state[:, np.newaxis] * ratio

        # Deal with NaNs
        vac[np.isnan(vac)] = 0

        # Update vaccination number for 2nd dose
        vac_2nd[ind:, ~neg_flag_V1_hist] += vac
        vac_2nd[ind:, neg_flag_V1_hist] = 0
    
    return vac_1st, vac_2nd, neg_flag_S_hist, neg_flag_V1_hist

def covid1(y, dt, t_idx, mV1, mV2, e1, e2, kappa, alpha, gamma, vprevf1, vprevf2, fatality_rate, 
          vprevs1, vprevs2, severe_illness_rate, alpha_eff, delta_eff, 
          delta, beta, sd, sc, contact, neg_flag_S_hist, neg_flag_V1_hist):
    '''state size : 13 * 9 = 117
       state size : 14 * 9 (include the new inf)
       WAIFW는 action에 따라 변하므로, action에 따라 움직이게 function안에서 움직여야 함'''
    
    # initial 
    S = y[0:9]
    E = y[9:18]
    I = y[18:27]
    H = y[27:36]
    R = y[36:45]
    V1 = y[45:54]
    V2 = y[54:63]
    EV1 = y[63:72]
    EV2 = y[72:81]
    IV1 = y[81:90]
    IV2 = y[90:99]
    # 실시간 계산을 위한 것
    F = np.zeros(9)
    SI = np.zeros(9)
    new_inf = np.zeros(9)

    ## for flag
    # - Flag check
    neg_flag_S_hist = neg_flag_S_hist.reshape(-1)
    neg_flag_V1_hist = neg_flag_V1_hist.reshape(-1)
    neg_flag_S = np.full((9, 1), False, dtype=bool)
    neg_flag_S = neg_flag_S.reshape(-1)
    neg_flag_V1 = np.full((9, 1), False, dtype=bool)
    neg_flag_V1 = neg_flag_V1.reshape(-1)
    
    # WAIFW
    '''sd = 1 X 440 (440일치 social distancing)
       alpha_eff = 471 X 1 --> constant
       delta_eff = 471 X 1 --> constant
       delta = 1
       beta = 0.0505
       contact = 9X9 (contact matrix)
       변이 : 471일, 사회적 거리두기 일자 : 439일 (시작부터~439까지)
       WAIFW = (471X1)*(1X439)*(9X9)
       WAIFW = 모두 상수 * contact matrix'''
    
    # t_idx parameters --> NO need
    contact[1,1] = contact[1,1] * sc
    
    # WAIFW
    # delta + alpha effect
    mix_eff = alpha_eff + (delta * delta_eff)    
    WAIFW = mix_eff * beta * sd * contact

    # main loop

    mv1 = mV1[t_idx,:]
    mv2 = mV2[t_idx,:]
   #======================= for 1day ===================#
    for _ in range(int(1/dt)):
      # Calculate the lambda
      sumI = I + IV1 + IV2

      #'.-',labmda = 9X1
      lambdaS = np.matmul(WAIFW, sumI) * S
      WAIFWV1 = WAIFW * (1-e1)
      lambdaV1 = np.dot(WAIFWV1, sumI) * V1
      WAIFWV2 = WAIFW * (1-e2)
      lambdaV2 = np.dot(WAIFWV2, sumI) * V2

      # age flow (단기라서 없음)
   
      
      # Difference equations
      '''시간에 따른 parameter : mV1, mV2 
      연령에 따른 parmater : lambda, fatality_rate, severe_illness_rate
      e1, e2 : Vaccine eff (constant)
      alpha, delta : 변이 비율 (contant)
      kappa:
      gamma:
      mV1, mV2 : number of vaccination'''
      S_next = S + (- lambdaS - mv1) * dt
      E_next = E + (lambdaS - kappa * E) * dt
      I_next = I + (kappa * E - alpha * I) * dt
      H_next = H + (alpha * (I + IV1 + IV2) - gamma * H) * dt
      R_next = R + (gamma * H) * dt
      V1_next = V1 + (mv1 - lambdaV1 - mv2) * dt 
      V2_next = V2 + (mv2 - lambdaV2) * dt
      EV1_next = EV1 + (lambdaV1 - kappa * EV1) * dt
      EV2_next = EV2 + (lambdaV2 - kappa * EV2) * dt
      IV1_next = IV1 + (kappa * EV1 - alpha * IV1) * dt
      IV2_next = IV2 + (kappa * EV2 - alpha * IV2) * dt
      F = F + dt * (alpha * (I + (1 - vprevf1) * IV1 + (1 - vprevf2) * IV2) * fatality_rate) 
      SI = SI + dt * (alpha * (I + (1 - vprevs1)* IV1 + (1 - vprevs2) * IV2) * severe_illness_rate)
      new_inf = new_inf + dt * (alpha *((I + IV1 + IV2) + (I_next+ IV1_next + IV2_next))) / 2

      S = S_next
      E = E_next
      I = I_next
      H = H_next
      R = R_next
      V1 = V1_next
      V2 = V2_next
      EV1 = EV1_next
      EV2 = EV2_next
      IV1 = IV1_next
      IV2 = IV2_next

      # negtive flag check
      if np.any(S < 0) or np.any(V1 < 0):
            neg_flag_S = S < 0
            neg_flag_V1 = V1 < 0
            break

    if np.any(neg_flag_S) or np.any(neg_flag_V1):
      mV1, mV2, neg_flag_S_hist, neg_flag_V1_hist = update_vaccine(mV1, mV2, t_idx, neg_flag_S, neg_flag_V1, 
                                                                  neg_flag_S_hist, neg_flag_V1_hist)
      y_ = covid1(y, dt, t_idx, mV1, mV2, e1, e2, kappa, alpha, gamma, vprevf1, vprevf2, fatality_rate, 
      vprevs1, vprevs2, severe_illness_rate, alpha_eff, delta_eff, delta, beta, sd, sc, contact, neg_flag_S_hist, neg_flag_V1_hist)
   
      S = y[0:9]
      E = y[9:18]
      I = y[18:27]
      H = y[27:36]
      R = y[36:45]
      V1 = y[45:54]
      V2 = y[54:63]
      EV1 = y[63:72]
      EV2 = y[72:81]
      IV1 = y[81:90]
      IV2 = y[90:99]
      F = y[99:108]
      SI = y[108:117]
      new_inf = y[117:126]
   #======================= for 1day ===================#

    #print(lambdaS)
    dydt = np.array([S, E, I, H, R, V1, V2, EV1, EV2, IV1, IV2, F, SI, new_inf])
    dydt = dydt.reshape(1,-1)[0]
    return dydt

#@jit
def covid7(y, dt, t_idx, mV1, mV2, e1, e2, kappa, alpha, gamma, vprevf1, vprevf2, fatality_rate, 
          vprevs1, vprevs2, severe_illness_rate, alpha_eff, delta_eff, 
          delta, beta, sd, sc, contact, neg_flag_S_hist, neg_flag_V1_hist):
    '''state size : 13 * 9 = 117
       state size : 14 * 9 (include the new inf)
       WAIFW는 action에 따라 변하므로, action에 따라 움직이게 function안에서 움직여야 함'''
    
    # initial 
    S = y[0:9]
    E = y[9:18]
    I = y[18:27]
    H = y[27:36]
    R = y[36:45]
    V1 = y[45:54]
    V2 = y[54:63]
    EV1 = y[63:72]
    EV2 = y[72:81]
    IV1 = y[81:90]
    IV2 = y[90:99]
    # 실시간 계산을 위한 것
    F = y[99:108]
    SI = y[108:117]
    new_inf = y[117:126]

    ## for flag
    # - Flag check
    neg_flag_S_hist = neg_flag_S_hist.reshape(-1)
    neg_flag_V1_hist = neg_flag_V1_hist.reshape(-1)
    neg_flag_S = np.full((9, 1), False, dtype=bool)
    neg_flag_S = neg_flag_S.reshape(-1)
    neg_flag_V1 = np.full((9, 1), False, dtype=bool)
    neg_flag_V1 = neg_flag_V1.reshape(-1)
    
    # WAIFW
    '''sd = 1 X 440 (440일치 social distancing)
       alpha_eff = 471 X 1 --> constant
       delta_eff = 471 X 1 --> constant
       delta = 1
       beta = 0.0505
       contact = 9X9 (contact matrix)
       변이 : 471일, 사회적 거리두기 일자 : 439일 (시작부터~439까지)
       WAIFW = (471X1)*(1X439)*(9X9)
       WAIFW = 모두 상수 * contact matrix'''
    
    # t_idx parameters --> NO need
    contact[1,1] = contact[1,1] * sc
    
    # WAIFW
    # delta + alpha effect
    mix_eff = alpha_eff + (delta * delta_eff)    
    WAIFW = mix_eff * beta * sd * contact

    # main loop
    for i in range(7):
        t_idx_ = (t_idx*7) + i
        y1 = covid1(y, dt, t_idx_, mV1, mV2, e1, e2, kappa, alpha, gamma, vprevf1, vprevf2, fatality_rate, 
            vprevs1, vprevs2, severe_illness_rate, alpha_eff, delta_eff, delta, beta, sd, sc, contact, neg_flag_S_hist, neg_flag_V1_hist)
        y = y1

        S = y[0:9]
        E = y[9:18]
        I = y[18:27]
        H = y[27:36]
        R = y[36:45]
        V1 = y[45:54]
        V2 = y[54:63]
        EV1 = y[63:72]
        EV2 = y[72:81]
        IV1 = y[81:90]
        IV2 = y[90:99]
        F = y[99:108]
        SI = y[108:117]
        new_inf = y[117:126]
        
    #print(lambdaS)
    dydt = np.array([S, E, I, H, R, V1, V2, EV1, EV2, IV1, IV2, F, SI, new_inf])
    dydt = dydt.reshape(1,-1)[0]
    return dydt

class covidEnvironment:
    def __init__(self):
        # initial
        self.S0 = BE['S0'][:,0]
        self.E0 = BE['E0'][:,0]
        self.I0 = BE['I0'][:,0]
        self.H0 = BE['H0'][:,0]
        self.R0 = BE['R0'][:,0]
        self.V10 = BE['V10'][:,0]
        self.V20 = BE['V20'][:,0]
        self.EV10 = BE['EV10'][:,0]
        self.IV10 = BE['IV10'][:,0]
        self.EV20 = BE['EV20'][:,0]
        self.IV20 = BE['EV10'][:,0]
        self.new_inf0 = np.zeros(9)
        self.F0 = np.zeros(9)
        self.SI0 = np.zeros(9)
        # self.mV1 = BE['MV1']
        # self.mV2 = BE['MV2']
        self.mV1 = MV['mv1'][259:,:]
        self.mV2 = MV['mv2'][259:,:]
        
        # parameters
        self.beta = M['beta'][0][0]
        self.kappa = M['kappa'][0][0]
        self.alpha = M['alpha'][0][0]
        self.gamma = M['gamma'][0][0]
        self.delta = M['delta'][0][0]
        self.fatality_rate = M['fatality_rate'][0]
        self.severe_illness_rate =  M['severe_illness_rate'][0]
        self.vprevs1 = M['vprevs'][0][0]
        self.vprevs2 = M['vprevs'][0][1]
        self.vprevf1 = M['vprevf'][0][0]
        self.vprevf2 = M['vprevf'][0][1]
        self.alpha_eff = 0.03
        self.delta_eff = 0.97
        self.contact = M['contact']
        self.e1 = 0.3326
        self.e2 = 0.7770
        self.sc = 1
        self.dt = 0.01

        # action space
        self.sds = []

        # others
        self.tf = 26
        self.dt = 0.01
        self.time = 0
        self.days = [self.time]
        self.his = np.array([self.S0, self.E0, self.I0, self.H0, self.R0,
                             self.V10, self.V20, self.EV10, self.EV20, 
                             self.IV10, self.IV20, self.F0, self.SI0, self.new_inf0])
        self.hist = self.his.reshape(1,-1)[0]
        self.history = [self.hist]
        self.lambdas = []
        self.rewards = []

    def reset(self):
        self.state = np.array([self.S0, self.E0, self.I0, self.H0, self.R0,
                             self.V10, self.V20, self.EV10, self.EV20, 
                             self.IV10, self.IV20, self.F0, self.SI0, self.new_inf0])
        self.state = self.state.reshape(1,-1)[0]

        # parameters
        self.beta = M['beta'][0][0]
        self.kappa = M['kappa'][0][0]
        self.alpha = M['alpha'][0][0]
        self.gamma = M['gamma'][0][0]
        self.delta = M['delta'][0][0]
        self.fatality_rate = M['fatality_rate'][0]
        self.severe_illness_rate =  M['severe_illness_rate'][0]
        self.vprevs1 = M['vprevs'][0][0]
        self.vprevs2 = M['vprevs'][0][1]
        self.vprevf1 = M['vprevf'][0][0]
        self.vprevf2 = M['vprevf'][0][1]
        self.alpha_eff = 0.03
        self.delta_eff = 0.97
        self.contact = M['contact']
        self.e1 = 0.3326
        self.e2 = 0.7770
        self.sc = 1
        self.dt = 0.01

        # action space

        self.rewards = []
        self.history = [self.state]
        return self.state

    def step(self, action, t_idx):
        # action value change
        # (4th 3rd 2nd 1st)
        sd = [0.4402, 0.4402*1.4, 0.4402*(1.4**2), 1]
        sd = sd[action]

        # nu = self.nu_min if action == 0 else self.nu_daily_max
        # self.nus.append(nu)
        # t_idx paramter
        # -flag
        neg_flag_S_hist = np.full((9, 1), False, dtype=bool)
        neg_flag_V1_hist = np.full((9, 1), False, dtype=bool)
        # state
        y0 = self.state
        y0 = y0.reshape(1,-1)[0]
        sol = covid7(y0, self.dt, t_idx, self.mV1, self.mV2,
                    self.e1, self.e2, self.kappa, self.alpha, self.gamma, 
                    self.vprevf1, self.vprevf2, self.fatality_rate, self.vprevs1,self.vprevs2, self.severe_illness_rate, 
                    self.alpha_eff, self.delta_eff, self.delta, self.beta, sd, self.sc, self.contact, neg_flag_S_hist, neg_flag_V1_hist)
        new_state = sol

        # new state
        S = new_state[0:9]
        E = new_state[9:18]
        I = new_state[18:27]
        H = new_state[27:36]
        R = new_state[36:45]
        V1 = new_state[45:54]
        V2 = new_state[54:63]
        EV1 = new_state[63:72]
        EV2 = new_state[72:81]
        IV1 = new_state[81:90]
        IV2 = new_state[90:99]
        F = new_state[99:108]
        SI = new_state[108:117]
        new_inf = new_state[117:126]

        # state update
        self.state = new_state

        # reward case
        # 실험용 : newinf + severecase + Fatalitycase
        reward = - (np.sum(I)) / 5000

        self.rewards.append(reward)
        self.days.append(self.time)
        self.history.append(new_state)
        
        # new_state[1] : I < 1.0 이면 멈춤
        done = True if self.time >= self.tf else False
        return (new_state, reward, done, {})

        
# model 실행
agent.qnetwork_local.load_state_dict(model)
#agent.qnetwork_local.eval()

agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
env = covidEnvironment()
max_t = 26
state = env.reset()
states = state
actions = []
score = 0
for t_idx in range(max_t):
    # from dqn_agent import Agent
    action = agent.act(state, eps=0.001)
    # optimal action
    actions = np.append(actions, action)
    next_state, reward, done, _ = env.step(action,t_idx)
    agent.step(state, action, reward, next_state, done)
    states = np.vstack((states, next_state))
    state = next_state
    score += reward

# env = SeiarEnvironment()
# state = env.reset()
# max_t = 300
# states = state
# actions = []
# score = 0
# dt = 1
# time_stamp = np.arange(0, max_t, dt)
# for t in time_stamp:
#     action = agent.act(state, eps=0.0)
#     actions.append(action)
#     next_state, reward, done, _ = env.step(action)
#     states = np.vstack((states, next_state))
#     state = next_state
#     score += reward
# actions_ = np.array(actions)
# time_stamp = np.append(time_stamp, max_t)



plt.clf()
fig, ax1 = plt.subplots()
ax1.plot(time_stamp, states[:,0], '.-b', label = 'S')
ax1.legend(loc='upper left')
ax2 = ax1.twinx()
ax2.plot(time_stamp, states[:,1], '.-y', label = 'L')
ax2.plot(time_stamp, states[:,2], '.-r', label = 'I')
ax2.plot(time_stamp, states[:,3], '.-g', label = 'A')
ax2.legend(loc='lower right')
plt.grid()
plt.legend()
plt.title('SLIAR model with control')
plt.xlabel('day')
plt.savefig(f"{npath}/outputs/2023-10-30/16-31-19/figures/SLIAR_w_control_{str(epi)}.png", dpi=300)

# For action plot
plt.clf()
plt.plot(actions_*500000, 'k+--', label = 'Vaccine')
plt.grid()
plt.legend()
plt.title(f'Control:{score}')
plt.savefig(f"{npath}/outputs/2023-10-30/16-31-19/figures/control_{str(epi)}.png", dpi=300)

plt.figure(figsize=(8,8))
plt.subplot(3, 1, 1)
plt.plot(time_stamp, states[:,2], '.-r', label = 'I')
plt.ylabel('Infected')
plt.title(f'Control:{score}')
plt.xticks(color='w')

plt.subplot(3, 1, 2)
plt.plot(actions_*500000, '-k', label = 'Vaccine')
plt.ylabel('Vaccination')
#plt.ylim([0.0, 0.01])
plt.xticks(color='w')

plt.subplot(3, 1, 3)
plt.plot(time_stamp, states[:,0], '.-b', label = 'S')
plt.ylabel('Susceptible')
plt.savefig(f"{npath}/outputs/2023-10-30/16-31-19/figures/all_{str(epi)}.png", dpi=300)

