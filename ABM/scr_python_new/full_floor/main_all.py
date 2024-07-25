# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pickle

from traffic import *
from agent_all import *
from model_all import *
from place_agent import *


# %%
path = "../data/sev_hospital/survey"
sev_traffic_data = call_traffic_data(path)

sev_model = EpidemicsModel(
    beta = 0.001,
    traffic_data = sev_traffic_data,
    place = 'severance',
    incub_p_dist = stats.lognorm(s=0.547, scale=np.exp(1.857)),
    m_incub_p_dist = 24,
    presym_I_p_dist = stats.truncnorm(a=(0 - 2.3) / 0.49,
                                      b=np.inf,
                                      loc=2.3,
                                      scale=0.49),
    m_presym_I_p_dist = 24,
    I_p_dist = stats.truncnorm(a=(0 - 7.2) / 4.96,
                               b=np.inf,
                               loc = 7.2,
                               scale = 4.96),
    m_I_p_dist = 24,
    is_only_am_therapy = True,
    test_num_per_week = 1,
    )

# %% set agents

place_agent(sev_model)

#%% simulation

n_simulation = 100
T = 24 * 90
dynamics_list = []

def progress_model(model, n_steps):
    for _ in range(n_steps):
        model.step()
    dynamics = model.datacollector.get_model_vars_dataframe()

    return dynamics

dynamic = []

#%% Running
# with tqdm(total=n_simulation) as pbar:
#     for i in range(n_simulation):
#         model = copy.deepcopy(sev_model)
#         dyn = progress_model(model)
#         dynamic.append(dyn)
        
#         pbar.set_description(f'# simulation {i+1}/{n_simulation}')
#         pbar.update(1)


#%% save dynamic

# with open('dynamics1.pkl', 'wb') as file:
#     pickle.dump(dynamic, file)

# Define the content to write to the file
# file_content = """
# 1. total population (6th~10th floor): 319
# 2. observation period: 90 days
# 3. step: 1hour
# 4. # simulation: 100
# 5. consumed time: 90*100 secs
# 6. model info:
# sev_model = EpidemicsModel(
#     beta = 0.001,
#     traffic_data = sev_traffic_data,
#     place = 'severance',
#     incub_p_dist = stats.lognorm(s=0.547, scale=np.exp(1.857)),
#     m_incub_p_dist = 24,
#     presym_I_p_dist = stats.truncnorm(a=(0 - 2.3) / 0.49,
#                                       b=np.inf,
#                                       loc=2.3,
#                                       scale=0.49),
#     m_presym_I_p_dist = 24,
#     I_p_dist = stats.truncnorm(a=(0 - 7.2) / 4.96,
#                                b=np.inf,
#                                loc = 7.2,
#                                scale = 4.96),
#     m_I_p_dist = 24,
#     is_only_am_therapy = True,
#     test_num_per_week = 1,
#     )
# """

# # Define the file name
# file_name = "dynamics1.txt"

# # Write the content to the file
# with open(f"{file_name}", "w") as file:
#     file.write(file_content)

#%% load dynamics

with open('dynamics1.pkl', 'rb') as file:
    dynamic = pickle.load(file)

#%%

stat_num_ls = np.zeros([n_simulation, 4, T])
for i in range(n_simulation):
    for j in range(T):
        stat_num_ls[i, :, j] = np.sum((dynamic[i]['occ_stat'][j]), axis=0)

#%%
t_step = np.linspace(1, T, T)/24
fig, ax1 = plt.subplots()
for i in range(n_simulation):
    ax1.plot(t_step, stat_num_ls[i, 0], 'b', alpha=0.1)
ax1.set_ylabel('S')
plt.ylim([259, 319])

ax2 = ax1.twinx()
for i in range(n_simulation):
    ax2.plot(t_step, stat_num_ls[i, 1], 'y', alpha=0.1)
    ax2.plot(t_step, stat_num_ls[i, 2], 'r', alpha=0.1)
    ax2.plot(t_step, stat_num_ls[i, 3], 'g', alpha=0.1)
ax2.set_ylabel('E, I, R')
plt.ylim([0, 60])

plt.title('Dynamics')
plt.xlabel('Days')
plt.show()

# %%
