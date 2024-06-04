# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from traffic import *
from agent_7 import *
from model_7 import *


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

# %% 7th floor: 19 nurses + 7 transfers + 8*4 patients + 8*4 caregivers (1 infectious)

# nurses
for i in range(19):
    classifier = "N" + str(i)
    sev_model.add_agent(classifier=classifier,
                        occ=0,
                        mr=13,
                        stat=0)


# transfers
for i in range(7):
    classifier = "T" + str(i)
    sev_model.add_agent(classifier=classifier,
                        occ=1,
                        mr=14,
                        stat=0)

# patients
for i in range(4):
    for j in range(8):
        classifier = "P" + str(8*i + j)
        sev_model.add_agent(classifier=classifier,
                            occ=2,
                            mr=i+1,
                            stat=0)

# caregivers
for i in range(3):
    for j in range(8):
        classifier = "C" + str(8*i + j)
        sev_model.add_agent(classifier=classifier,
                            occ=3,
                            mr=i+1,
                            stat=0)

for j in range(7):
    classifier = "C" + str(24 + j)
    sev_model.add_agent(classifier=classifier,
                        occ=3,
                        mr=4,
                        stat=0)
    
# caregiver (Infectious)
sev_model.add_agent(classifier="C31",
                    occ=3,
                    mr=4,
                    stat=2)

#%%

n_simulation = 50
T = 24 * 90
dynamics_list = []

def progress_model(model):
    with tqdm(total=T) as pbar:
        for i in range(T):
            model.step()
            pbar.set_description(f'Time step {i+1}/{T}')
            pbar.update(1)
        dynamics = model.datacollector.get_model_vars_dataframe()
    
        return dynamics

dynamic = []

for i in range(n_simulation):
    model = copy.deepcopy(sev_model)
    dyn = progress_model(model)
    dynamic.append(dyn)

#%%

stat_num_ls = np.zeros([n_simulation, 4, T])
for i in range(n_simulation):
    for j in range(T):
        stat_num_ls[i, :, j] = np.sum((dynamic[i]['occ_stat'][j]), axis=0)

#%%
t_step = np.linspace(1, T, T)/24
fig, ax1 = plt.subplots()
for i in range(n_simulation):
    ax1.plot(t_step, stat_num_ls[i, 0], 'b', alpha=0.3)
ax1.set_ylabel('S')
plt.ylim([70, 90])

ax2 = ax1.twinx()
for i in range(n_simulation):
    ax2.plot(t_step, stat_num_ls[i, 1], 'y', alpha=0.3)
    ax2.plot(t_step, stat_num_ls[i, 2], 'r', alpha=0.3)
    ax2.plot(t_step, stat_num_ls[i, 3], 'g', alpha=0.3)
ax2.set_ylabel('E, I, R')
plt.ylim([0, 20])

plt.title('Dynamics')
plt.xlabel('Days')
plt.show()

# %%
