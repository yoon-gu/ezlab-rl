import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import env_
import evaluation.Baselines as Baselines
import matplotlib.patches as mpatches

if not os.path.exists('Figures'):
    os.mkdir("Figures")   



# needs to be organized

def plot_deterministic_AI(reward_id, problem_id, exposed_0_19, exposed_19_69, exposed_70, episode_rewards):



    #population
    N = 566_994.0 + 1528112.0 + 279444.0
    
    # calling the respective Baselines
    rewards_determenisticSV, statesSV = Baselines.evaluate_deterministic_model(reward_id = reward_id, problem_id = problem_id, policy_chosen='policy_Sweden', period='autumn') #Deterministic Swedemm
 
    rewards_determenisticSV_F, statesSV_F = Baselines.evaluate_deterministic_model(reward_id = reward_id, problem_id = problem_id, policy_chosen='policy_Sweden_second', period='FOHM') #Deterministic lockdown


    # autumn period fitted parameters
    unreported_0_19_p2 = 0.9755026399435202
    unreported_20_69_p2 = 0.4763668186715981
    unreported_70_p2 = 0.01
    exposed_rate = 5.1


    # creating lists to save the observations
    DT_exposed_0_19SV = []
    DT_exposed_19_69SV = []
    DT_exposed_70SV = []

    DT_exposed_0_19SV_F = []
    DT_exposed_19_69SV_F = []
    DT_exposed_70SV_F = []



    # the incidence curves appended for Sweden ECDC
    for i in range(len(statesSV)):

        DT_exposed_0_19SV.append(statesSV[i][3])
        DT_exposed_19_69SV.append(statesSV[i][4])
        DT_exposed_70SV.append(statesSV[i][5])

    DT_exposed_0_19SV_incidence = np.array(DT_exposed_0_19SV)/N *  1/exposed_rate * (1- unreported_0_19_p2) + np.array(DT_exposed_0_19SV)/N *  1/exposed_rate * (unreported_0_19_p2)
    DT_exposed_19_69SV_incidence = np.array(DT_exposed_19_69SV)/N *  1/exposed_rate * (1- unreported_20_69_p2) + np.array(DT_exposed_19_69SV)/N *  1/exposed_rate * (unreported_20_69_p2)
    DT_exposed_70SV_incidence = np.array(DT_exposed_70SV)/N *  1/exposed_rate * (1- unreported_70_p2) + np.array(DT_exposed_19_69SV)/N *  1/exposed_rate * (unreported_20_69_p2)
    
    DT_totalSV = np.array(DT_exposed_0_19SV_incidence) + np.array(DT_exposed_19_69SV_incidence) + np.array(DT_exposed_70SV_incidence)

    # the incidence curves appended for Sweden Fitted
    for i in range(len(statesSV_F)):

        DT_exposed_0_19SV_F.append(statesSV_F[i][3])
        DT_exposed_19_69SV_F.append(statesSV_F[i][4])
        DT_exposed_70SV_F.append(statesSV_F[i][5])

    DT_exposed_0_19SV_F_incidence = np.array(DT_exposed_0_19SV_F)/N *  1/exposed_rate * (1- unreported_0_19_p2) + np.array(DT_exposed_0_19SV_F)/N *  1/exposed_rate * (unreported_0_19_p2)
    DT_exposed_19_69SV_F_incidence = np.array(DT_exposed_19_69SV_F)/N *  1/exposed_rate * (1- unreported_20_69_p2) + np.array(DT_exposed_19_69SV_F)/N *  1/exposed_rate * (unreported_20_69_p2)
    DT_exposed_70SV_F_incidence = np.array(DT_exposed_70SV_F)/N *  1/exposed_rate * (1- unreported_70_p2) + np.array(DT_exposed_70SV_F)/N *  1/exposed_rate * (unreported_70_p2) 

    DT_totalSV_F = np.array(DT_exposed_0_19SV_F_incidence) + np.array(DT_exposed_19_69SV_F_incidence) + np.array(DT_exposed_70SV_F_incidence)


    total = [None] * len(exposed_0_19)

        

    # Plotting the deterministcs vs EpideRlearn + the rewards for each
    fig, axs = plt.subplots(1, 2)
    for i in range(len(exposed_19_69)):
        l = list(range(len(exposed_19_69)))
        if i == l[-1]:
            label_ = 'EpidRLearn'
        else:
            label_ = None
        
        total[i] = np.array(exposed_0_19[i])/N *  1/exposed_rate * (1- unreported_0_19_p2) + np.array(exposed_0_19[i])/N *  1/exposed_rate * (unreported_0_19_p2)\
           + np.array(exposed_19_69[i])/N *  1/exposed_rate * (1- unreported_20_69_p2) + np.array(exposed_19_69[i])/N *  1/exposed_rate * (unreported_20_69_p2)\
            + np.array(exposed_70[i])/N *  1/exposed_rate * (1- unreported_70_p2) + np.array(exposed_70[i])/N *  1/exposed_rate * (unreported_70_p2)

        axs[0].plot(total[i],  label=label_, color = 'blue')

    
    axs[0].plot(DT_totalSV,label='Baseline Sweden',  color = 'red')
    axs[0].plot(DT_totalSV_F,label='Baseline Sweden Fitted',  color = 'orange')
    axs[0].set_xlabel('Weeks')
    axs[0].set_ylabel('Population %')
    axs[0].set_title('Infected (Incidence)')
    axs[0].legend(loc=1)

    axs[1].plot([rewards_determenisticSV[0]], marker ='o', label='Reward Baseline Sweden', color = 'red')
    axs[1].plot([rewards_determenisticSV_F[0]], marker ='o', label='Reward Baseline Sweden Fitted', color = 'orange')
    bp = axs[1].boxplot(episode_rewards,notch=True, patch_artist=True)
    for box in bp['boxes']:
        # change outline color
        box.set(color='blue')

    axs[1].set_ylabel('Rewards')
    axs[1].set_title('Rewards')

    #plt.suptitle("EpiRLearn - Baselines", size=16)
    red_patch = mpatches.Patch(color='red', label='Reward Baseline Sweden')
    orange_patch = mpatches.Patch(color='orange', label='Reward Baseline Sweden Fitted')
    blue_patch = mpatches.Patch(color='blue', label='Reward EpidRLearn')
    plt.legend(handles=[red_patch, orange_patch, blue_patch])
    plt.gcf().set_size_inches(16, 7)
    
    
    # plt.savefig(f"Figures/EpidLearnVSDeterministic_prob{problem_id}_rwd{reward_id}.svg",format="svg")
    plt.savefig(f"Figures/EpidLearnVSDeterministic_prob{problem_id}_rwd{reward_id}.png")
    plt.show()


def evaluate(model, num_episodes, reward_id, problem_id, period):

    """
    Evaluate the RL agent

    model:
    num_episodes: int
    reward_id: int
    period: string
    print: boolean

    """
    # This function will only work for a single Environment
    envr = env_.Epidemic(problem_id = problem_id, reward_id=reward_id, period=period)

    all_episode_rewards = []

    all_episode_exposed_0_19 = []
    all_episode_exposed_19_69 = []
    all_episode_exposed_70 = []


    
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
            obs, reward, done, info= envr.step(action)
            
            exposed_0_19.append(obs[3])
            exposed_19_69.append(obs[4])
            exposed_70.append(obs[5])

            episode_rewards.append(reward)

        
        all_episode_rewards.append(sum(episode_rewards))
        all_episode_exposed_0_19.append(exposed_0_19)
        all_episode_exposed_19_69.append(exposed_19_69)
        all_episode_exposed_70.append(exposed_70)

        
    envr.render()

    if period == 'spring': 

        print(obs[0])

        mydict = {'susceptible_0_19': obs[0], 'susceptible_19_69': obs[1], 'susceptible_70': obs[2], 'exposed_0_19': obs[3], 'exposed_19_69': obs[4],
            'exposed_70': obs[5], 'InU_0_19': obs[6], 'InU_19_69': obs[7], 'InU_70': obs[8], 'InR_0_19': obs[9], 'InR_19_69': obs[10], 'InR_70': obs[11], 
            'R1_0_19': obs[12], 'R1_19_69': obs[13], 'R1_70': obs[14], 'R2_0_19': obs[15], 'R2_19_69': obs[16], 'R2_70': obs[17]} 
        tosave = pd.DataFrame(mydict, index=[0])
        tosave.to_csv(f"Analysis_Files/ForWinter_prob{problem_id}_rwd{reward_id}.csv")



    # passing EpidRLearn observation to be plotted together with Baselines
    plot_deterministic_AI(reward_id, problem_id, all_episode_exposed_0_19, all_episode_exposed_19_69, all_episode_exposed_70, all_episode_rewards)

    mean_episode_reward = np.mean(all_episode_rewards)
        
    return mean_episode_reward, actions


def plot_sensitivity_analysis(e64, e73, e82, e91):

    N = 566_994.0 + 1528112.0 + 279444.0
    

    unreported_srping019 = 0.98416430249021
    unreported_srping2069 = 0.9869717837070985
    unreported_srping70 = 0.9639400113485507
    exposed_rate = 5.1



    total64 = [None] * len(e64.exposed2_0_19)
    total73 = [None] * len(e73.exposed3_0_19)
    total82 = [None] * len(e82.exposed4_0_19)
    total91 = [None] * len(e91.exposed5_0_19)


    for i in range(len(e64.exposed2_0_19)):
        
        total64[i] = np.array(e64.exposed2_0_19[i])/N *  1/exposed_rate * (1- unreported_srping019) + np.array(e64.exposed2_0_19[i])/N *  1/exposed_rate * (unreported_srping019)\
           + np.array(e64.exposed2_19_69[i])/N *  1/exposed_rate * (1- unreported_srping2069) + np.array(e64.exposed2_19_69[i])/N *  1/exposed_rate * (unreported_srping2069)\
            + np.array(e64.exposed2_70[i])/N *  1/exposed_rate * (1- unreported_srping70) + np.array(e64.exposed2_70[i])/N *  1/exposed_rate * (unreported_srping70)

        total73[i] = np.array(e73.exposed3_0_19[i])/N *  1/exposed_rate * (1- unreported_srping019) + np.array(e73.exposed3_0_19[i])/N *  1/exposed_rate * (unreported_srping019)\
           + np.array(e73.exposed3_19_69[i])/N *  1/exposed_rate * (1- unreported_srping2069) + np.array(e73.exposed3_19_69[i])/N *  1/exposed_rate * (unreported_srping2069)\
            + np.array(e73.exposed3_70[i])/N *  1/exposed_rate * (1- unreported_srping70) + np.array(e73.exposed3_70[i])/N *  1/exposed_rate * (unreported_srping70)
        
        total82[i] = np.array(e82.exposed4_0_19[i])/N *  1/exposed_rate * (1- unreported_srping019) + np.array(e82.exposed4_0_19[i])/N *  1/exposed_rate * (unreported_srping019)\
           + np.array(e82.exposed4_19_69[i])/N *  1/exposed_rate * (1- unreported_srping2069) + np.array(e82.exposed4_19_69[i])/N *  1/exposed_rate * (unreported_srping2069)\
            + np.array(e82.exposed4_70[i])/N *  1/exposed_rate * (1- unreported_srping70) + np.array(e82.exposed4_70[i])/N *  1/exposed_rate * (unreported_srping70)

        total91[i] = np.array(e91.exposed5_0_19[i])/N *  1/exposed_rate * (1- unreported_srping019) + np.array(e91.exposed5_0_19[i])/N *  1/exposed_rate * (unreported_srping019)\
           + np.array(e91.exposed5_19_69[i])/N *  1/exposed_rate * (1- unreported_srping2069) + np.array(e91.exposed5_19_69[i])/N *  1/exposed_rate * (unreported_srping2069)\
            + np.array(e91.exposed5_70[i])/N *  1/exposed_rate * (1- unreported_srping70) + np.array(e91.exposed5_70[i])/N *  1/exposed_rate * (unreported_srping70)

    fig = plt.figure()
    fig.plot(total64, label='W1= .5, W2 = .5', color='blue')
    fig.plot(total73, label='W1= .4, W2 = .6', color = 'red')
    fig.plot(total82, label='W1= .3, W2 = .7', color = 'green')
    fig.plot(total91, label='W1= .2, W2 = .8', color = 'brown')
    plt.gcf().set_size_inches(8, 7)
    plt.legend(loc=2)
    fig.update_yaxes(title_text="Population %")
    fig.update_xaxes(title_text="Weeks")
    plt.savefig("Figures/sensitivity_analysis.svg", format="svg")
    plt.savefig("Figures/sensitivity_analysis.png")

