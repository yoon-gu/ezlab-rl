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

class sev_abm_env():
    
    def __init__(self, path="../data/sev_hospital/survey"):
        
        self.sev_traffic_data = call_traffic_data(path)

        self.sev_model = EpidemicsModel(
            beta = 0.001,
            traffic_data = self.sev_traffic_data,
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

        ## set agents

        place_agent(self.sev_model)
        
    
    def reset_abm(self):
        self.sev_model = EpidemicsModel(
            beta = 0.001,
            traffic_data = self.sev_traffic_data,
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
        
        
    def run_step(self, action = True):
        
        self.sev_model.is_only_am_therapy = action
        for _ in range(24):
            self.sev_model.step()
    
    
    def collect_data(self):
        dynamic = self.sev_model.datacollector.get_model_vars_dataframe()

        return dynamic

        ## Running
        # with tqdm(total=n_simulation) as pbar:
        #     for i in range(n_simulation):
        #         model = copy.deepcopy(sev_model)
        #         dyn = progress_model(model)
        #         dynamic.append(dyn)
                
        #         pbar.set_description(f'# simulation {i+1}/{n_simulation}')
        #         pbar.update(1)


        ## save dynamic
    def save_dynamic(self, dynamic, file_name, explanation=False):
        with open('dynamics1.pkl', 'wb') as file:
            pickle.dump(dynamic, file)

        if explanation:
            # explanation of abm
            file_content = """
            1. total population (6th~10th floor): 319
            2. observation period: 90 days
            3. step: 1hour
            4. # simulation: 100
            5. consumed time: 90*100 secs
            6. model info:
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
                """

            # Write the content to the file
            with open(f"{file_name}", "w") as file:
                file.write(file_content)

    
    def plotting(self, dynamic, total_date, dir='/fig.png', save=True):
        stat_num_ls= np.zeros([4, total_date])
        for i in range(total_date):
            stat_num_ls[:, i] = np.sum((dynamic['occ_stat'][i]), axis=0)
            t_step = np.linspace(1, total_date, total_date)/24
            
        fig, ax1 = plt.subplots()
        ax1.plot(t_step, stat_num_ls[i, 0], 'b')
        ax1.set_ylabel('S')

        ax2 = ax1.twinx()
        ax2.plot(t_step, stat_num_ls[i, 1], 'y')
        ax2.plot(t_step, stat_num_ls[i, 2], 'r')
        ax2.plot(t_step, stat_num_ls[i, 3], 'g')
        ax2.set_ylabel('E, I, R')
        plt.ylim([0, 60])

        plt.title('Dynamics')
        plt.xlabel('Days')

        if save:
            plt.savefig(dir)
            plt.close()
        else:
            plt.show()

## load dynamics

# with open('dynamics1.pkl', 'rb') as file:
#     dynamic = pickle.load(file)
