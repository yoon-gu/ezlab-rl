#%%
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from input_data import *
from make_abm import *
from collect_data import *
from add_agents import *

#%%
path = "../data/sev_hospital/survey"
sev_traffic_data = call_traffic_data(path)

sev_model = EpidemicsModel(place = 'severance',
                           beta = 0.001,
                           incubation_period_distn = stats.lognorm(s=0.547, scale=np.exp(1.857)),
                           multiplier_incubation_distn = 24,
                           presymptomatic_infectious_period_distn = stats.truncnorm(a=(0 - 2.3) / 0.49,
                                                                                    b=np.inf,
                                                                                    loc=2.3,
                                                                                    scale=0.49
                                                                                    ),
                           multiplier_presymptomatic_distn = 24,
                           infectious_period_distn = stats.truncnorm(a=(0 - 7.2) / 4.96,
                                                                     b=np.inf,
                                                                     loc = 7.2,
                                                                     scale = 4.96),
                           multiplier_infectious_distn = 24,
                           is_only_am_therapy = True,
                           test_num_per_week = 1,
                           traffic_data = sev_traffic_data
                           )


#%%
# Add nurse
sum_agent_start = 0
# n_nurses = 19 + 17 + 12
n_nurses_7 = 19
sum_agent_end = sum_agent_start + n_nurses_7
u_id_list = list(range(sum_agent_start, sum_agent_end))
sum_agent_start = sum_agent_end
add_n_nurses_on_floor_whos_status(model=sev_model,
                                  u_id_list=u_id_list,
                                  number=19,
                                  main_floor=7,
                                  status='S')

n_nurses_8 = 17
sum_agent_end = sum_agent_start + n_nurses_8
u_id_list = list(range(sum_agent_start, sum_agent_end))
sum_agent_start = sum_agent_end
add_n_nurses_on_floor_whos_status(model=sev_model,
                                  u_id_list=u_id_list,
                                  number=17,
                                  main_floor=8,
                                  status='S')

n_nurses_10 = 12
sum_agent_end = sum_agent_start + n_nurses_10
u_id_list = list(range(sum_agent_start, sum_agent_end))
sum_agent_start = sum_agent_end
add_n_nurses_on_floor_whos_status(model=sev_model,
                                  u_id_list=u_id_list,
                                  number=12,
                                  main_floor=10,
                                  status='S')

# Add transfers
# n_transfers = 2 + 6 + 2 + 2

n_transfers_7 = 2
sum_agent_end = sum_agent_start + n_transfers_7
u_id_list = list(range(sum_agent_start, sum_agent_end))
sum_agent_start = sum_agent_end
add_n_transfers_on_floor_whos_status(model=sev_model,
                                     u_id_list=u_id_list,
                                     number=2,
                                     main_floor=7,
                                     status='S')

n_transfers_8 = 6
sum_agent_end = sum_agent_start + n_transfers_8
u_id_list = list(range(sum_agent_start, sum_agent_end))
sum_agent_start = sum_agent_end
add_n_transfers_on_floor_whos_status(model=sev_model,
                                     u_id_list=u_id_list,
                                     number=6,
                                     main_floor=8,
                                     status='S')


n_transfers_9 = 2
sum_agent_end = sum_agent_start + n_transfers_9
u_id_list = list(range(sum_agent_start, sum_agent_end))
sum_agent_start = sum_agent_end
add_n_transfers_on_floor_whos_status(model=sev_model,
                                     u_id_list=u_id_list,
                                     number=2,
                                     main_floor=9,
                                     status='S')

n_transfers_10 = 2
sum_agent_end = sum_agent_start + n_transfers_10
u_id_list = list(range(sum_agent_start, sum_agent_end))
sum_agent_start = sum_agent_end
add_n_transfers_on_floor_whos_status(model=sev_model,
                                     u_id_list=u_id_list,
                                     number=2,
                                     main_floor=10,
                                     status='S')

# Add patients and caregivers
# n_patients = 8 * 4 + 8 * 6 + 8 * 2 + 2
# n_caregivers = n_patients

# 7th floor
n_patients_7 = 8
sum_agent_end = sum_agent_start + n_patients_7
u_id_list = list(range(sum_agent_start, sum_agent_end))
sum_agent_start = sum_agent_end
add_n_patients_on_floor_at_room_whos_status(model=sev_model,
                                            u_id_list=u_id_list,
                                            number=8,
                                            main_floor=7,
                                            room_number=1,
                                            status='S')

n_caregivers_7 = 7
sum_agent_end = sum_agent_start + n_caregivers_7
u_id_list = list(range(sum_agent_start, sum_agent_end))
sum_agent_start = sum_agent_end
add_n_caregivers_on_floor_at_room_whos_status(model=sev_model,
                                              u_id_list=u_id_list,
                                              number=7,
                                              main_floor=7,
                                              room_number=1,
                                              status='S')

n_caregivers_7 = 1
sum_agent_end = sum_agent_start + n_caregivers_7
u_id_list = list(range(sum_agent_start, sum_agent_end))
sum_agent_start = sum_agent_end
add_n_caregivers_on_floor_at_room_whos_status(model=sev_model,
                                              u_id_list=u_id_list,
                                              number=1,
                                              main_floor=7,
                                              room_number=1,
                                              status='I')

for room_num in range(2, 5):
    n_patients_7 = 8
    sum_agent_end = sum_agent_start + n_patients_7
    u_id_list = list(range(sum_agent_start, sum_agent_end))
    sum_agent_start = sum_agent_end
    add_n_patients_on_floor_at_room_whos_status(model=sev_model,
                                                u_id_list=u_id_list,
                                                number=8,
                                                main_floor=7,
                                                room_number=room_num,
                                                status='S')

    n_caregivers_7 = 8
    sum_agent_end = sum_agent_start + n_caregivers_7
    u_id_list = list(range(sum_agent_start, sum_agent_end))
    sum_agent_start = sum_agent_end
    add_n_caregivers_on_floor_at_room_whos_status(model=sev_model,
                                                  u_id_list=u_id_list,
                                                  number=8,
                                                  main_floor=7,
                                                  room_number=room_num,
                                                  status='S')
    
# 8th floor
for room_num in range(1, 7):
    n_patients_8 = 8
    sum_agent_end = sum_agent_start + n_patients_8
    u_id_list = list(range(sum_agent_start, sum_agent_end))
    sum_agent_start = sum_agent_end
    add_n_patients_on_floor_at_room_whos_status(model=sev_model,
                                                u_id_list=u_id_list,
                                                number=8,
                                                main_floor=8,
                                                room_number=room_num,
                                                status='S')

    n_caregivers_8 = 8
    sum_agent_end = sum_agent_start + n_caregivers_8
    u_id_list = list(range(sum_agent_start, sum_agent_end))
    sum_agent_start = sum_agent_end
    add_n_caregivers_on_floor_at_room_whos_status(model=sev_model,
                                                  u_id_list=u_id_list,
                                                  number=8,
                                                  main_floor=8,
                                                  room_number=room_num,
                                                  status='S')
    
# 10th floor
for room_num in range(1, 3):
    n_patients_10 = 8
    sum_agent_end = sum_agent_start + n_patients_10
    u_id_list = list(range(sum_agent_start, sum_agent_end))
    sum_agent_start = sum_agent_end
    add_n_patients_on_floor_at_room_whos_status(model=sev_model,
                                                u_id_list=u_id_list,
                                                number=8,
                                                main_floor=10,
                                                room_number=room_num,
                                                status='S')
    
    n_caregivers_10 = 8
    sum_agent_end = sum_agent_start + n_caregivers_10
    u_id_list = list(range(sum_agent_start, sum_agent_end))
    sum_agent_start = sum_agent_end
    add_n_caregivers_on_floor_at_room_whos_status(model=sev_model,
                                                  u_id_list=u_id_list,
                                                  number=8,
                                                  main_floor=10,
                                                  room_number=room_num,
                                                  status='S')

# main_room = 43!!
n_patients_10 = 2
sum_agent_end = sum_agent_start + n_patients_10
u_id_list = list(range(sum_agent_start, sum_agent_end))
sum_agent_start = sum_agent_end
add_n_patients_on_floor_at_room_whos_status(model=sev_model,
                                            u_id_list=u_id_list,
                                            number=2,
                                            main_floor=10,
                                            room_number=3,
                                            status='S')

n_caregivers_10 = 2
sum_agent_end = sum_agent_start + n_caregivers_10
u_id_list = list(range(sum_agent_start, sum_agent_end))
sum_agent_start = sum_agent_end
add_n_caregivers_on_floor_at_room_whos_status(model=sev_model,
                                              u_id_list=u_id_list,
                                              number=2,
                                              main_floor=10,
                                              room_number=3,
                                              status='S')

# Add operational therapists
# n_operational_therapists = 20 + 3
n_operational_therapists_6 = 20
sum_agent_end = sum_agent_start + n_operational_therapists_6
u_id_list = list(range(sum_agent_start, sum_agent_end))
sum_agent_start = sum_agent_end
add_n_operational_therapists_whos_status_at_6th(model=sev_model,
                                                u_id_list=u_id_list,
                                                number=20,
                                                status='S')

n_operational_therapists_9 = 3
sum_agent_end = sum_agent_start + n_operational_therapists_9
u_id_list = list(range(sum_agent_start, sum_agent_end))
sum_agent_start = sum_agent_end
add_n_operational_therapists_whos_status_at_9th(model=sev_model,
                                                u_id_list=u_id_list,
                                                number=3,
                                                status='S')

# Add robotic therapists
n_robotic_therapists = 6
sum_agent_end = sum_agent_start + n_robotic_therapists
u_id_list = list(range(sum_agent_start, sum_agent_end))
sum_agent_start = sum_agent_end
add_n_robotic_therapists_whos_status(model=sev_model,
                                     u_id_list=u_id_list,
                                     number=n_robotic_therapists,
                                     status='S')

# Add physical therapists
n_physical_therapists = 26
sum_agent_end = sum_agent_start + n_physical_therapists
u_id_list = list(range(sum_agent_start, sum_agent_end))
sum_agent_start = sum_agent_end
add_n_physical_therapists_whos_status(model=sev_model,
                                      u_id_list=u_id_list,
                                      number=n_physical_therapists,
                                      status='S')

# Add cleaners
# n_cleaners = 1 + 2 + 2 + 1 + 2
n_cleaners = 1
sum_agent_end = sum_agent_start + n_cleaners
u_id_list = list(range(sum_agent_start, sum_agent_end))
sum_agent_start = sum_agent_end
add_n_cleaners_on_floor_whos_status(model=sev_model,
                                    u_id_list=u_id_list,
                                    number=1,
                                    main_floor=6,
                                    status='S')

n_cleaners = 2
sum_agent_end = sum_agent_start + n_cleaners
u_id_list = list(range(sum_agent_start, sum_agent_end))
sum_agent_start = sum_agent_end
add_n_cleaners_on_floor_whos_status(model=sev_model,
                                    u_id_list=u_id_list,
                                    number=2,
                                    main_floor=7,
                                    status='S')

n_cleaners = 2
sum_agent_end = sum_agent_start + n_cleaners
u_id_list = list(range(sum_agent_start, sum_agent_end))
sum_agent_start = sum_agent_end
add_n_cleaners_on_floor_whos_status(model=sev_model,
                                    u_id_list=u_id_list,
                                    number=2,
                                    main_floor=8,
                                    status='S')

n_cleaners = 1
sum_agent_end = sum_agent_start + n_cleaners
u_id_list = list(range(sum_agent_start, sum_agent_end))
sum_agent_start = sum_agent_end
add_n_cleaners_on_floor_whos_status(model=sev_model,
                                    u_id_list=u_id_list,
                                    number=1,
                                    main_floor=9,
                                    status='S')

n_cleaners = 2
sum_agent_end = sum_agent_start + n_cleaners
u_id_list = list(range(sum_agent_start, sum_agent_end))
add_n_cleaners_on_floor_whos_status(model=sev_model,
                                    u_id_list=u_id_list,
                                    number=2,
                                    main_floor=10,
                                    status='S')


#%% =======================================================================================

n_simulation = 1
T = 24 * 40
dynamics_list = []

def progress_model(model):
    with tqdm(total=T) as pbar:
        for i in range(T):
            model.step()
            pbar.set_description(f'Time step {i+1}/{T}')
            pbar.update(1)
        dynamics = model.datacollector.get_model_vars_dataframe()
    
        return dynamics

model = copy.deepcopy(sev_model)
dynamic = progress_model(model)

# start = time.time()

# with ProcessPoolExecutor(max_workers=16) as pool:
#     for _ in range(n_simulation):
#         model = copy.deepcopy(sev_model)
#         dynamic = pool.submit(progress_model, model)
#         dynamics_list.append(dynamic)

# end = time.time()

# delta = end - start
# %%

dynamic
# %%
P_num_ls = []
C_num_ls = []
N_num_ls = []
Wa_num_ls = []
Wp_num_ls = []
Wo_num_ls = []
Wr_num_ls = []
Wc_num_ls = []

for i in range(T):
    P_num_ls.append(dynamic['occ_stat'][i]['P_num'])
    C_num_ls.append(dynamic['occ_stat'][i]['C_num'])
    N_num_ls.append(dynamic['occ_stat'][i]['N_num'])
    Wa_num_ls.append(dynamic['occ_stat'][i]['Wa_num'])
    Wp_num_ls.append(dynamic['occ_stat'][i]['Wp_num'])
    Wo_num_ls.append(dynamic['occ_stat'][i]['Wo_num'])
    Wr_num_ls.append(dynamic['occ_stat'][i]['Wr_num'])
    Wc_num_ls.append(dynamic['occ_stat'][i]['Wc_num'])


P_num_ls = np.array(P_num_ls)
C_num_ls = np.array(C_num_ls)
N_num_ls = np.array(N_num_ls)
Wa_num_ls = np.array(Wa_num_ls)
Wp_num_ls = np.array(Wp_num_ls)
Wo_num_ls = np.array(Wo_num_ls)
Wr_num_ls = np.array(Wr_num_ls)
Wc_num_ls = np.array(Wc_num_ls)

total_num_ls = P_num_ls + C_num_ls + N_num_ls + Wa_num_ls + Wp_num_ls + Wo_num_ls + Wr_num_ls + Wc_num_ls
S_total = total_num_ls[:, 0]
E_total = total_num_ls[:, 1]
I_total = total_num_ls[:, 2]
R_total = total_num_ls[:, 3]
# %%
t_step = np.linspace(1, T, T)/24
plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.plot(t_step, S_total, label="total S")
plt.title("S dynamic")
plt.ylim([0, 350])
plt.grid("True", alpha=0.5)

plt.subplot(2,2,2)
plt.plot(t_step, E_total, label="total E")
plt.title("E dynamic")
plt.ylim([0, 350])

plt.grid("True", alpha=0.5)

plt.subplot(2,2,3)
plt.plot(t_step, I_total, label="total I")
plt.title("I dynamic")
plt.ylim([0, 350])
plt.grid("True", alpha=0.5)

plt.subplot(2,2,4)
plt.plot(t_step, R_total, label="total R")
plt.title("R dynamic")
plt.ylim([0, 350])
plt.grid("True", alpha=0.5)

plt.show()
# %%
plt.figure(figsize=(8,5))
plt.plot(t_step, S_total, label="total S")
plt.plot(t_step, E_total, label="total E")
plt.plot(t_step, I_total, label="total I")
plt.plot(t_step, R_total, label="total R")
plt.legend()
plt.title("Dynamics")
plt.grid("True", alpha=0.5)
