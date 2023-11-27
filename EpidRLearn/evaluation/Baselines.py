import os
import matplotlib.pyplot as plt
import numpy as np
import env_
if not os.path.exists('Figures'):
    os.mkdir("Figures")    


"""

Function that defines the deterministic policies to be compared with the AI policies


"""

def policy_Sweden(actions, week, state):

    """
    Sweden ECDC: mild lockdown policy
    """

    free_to_move = actions[0]
    _75_lockdown = actions[1]
    _50_lockdown = actions[2]
    _25_lockdown = actions[3]



    if(week<5):
        #print("Action chosen: _50_lockdown")
        return _25_lockdown
    if(week>=5):
        #print("Action chosen: _50_lockdown")
        return _50_lockdown


def policy_Sweden_second(actions, week, state):

    """
    Sweden Fitted: the action here is always free_to_move
    as this refers to the fitted version and the actions are derived from the contact reduction matrix
    
    """
    free_to_move = actions[0]



    if(week>=0):

        return free_to_move





def evaluate_deterministic_model(reward_id, problem_id, policy_chosen, period='autumn'):
    

    problems = [0]
    rewards_per_problem =  []
    for problem in problems:
        if period == 'autumn':
            #print('autUMNN')
            envr = env_.Epidemic(problem_id = problem_id, reward_id=reward_id, period = 'autumn')
        elif period == 'FOHM':
            #print('fohm')
            envr = env_.Epidemic(problem_id = problem_id, reward_id=reward_id, period = 'FOHM')
        
        states = []
        rewards = []
        done = False
        
        state = envr.reset()
        states.append(state)
        
        actions = [0, 1, 2, 3]

        week = 0
        while not done:
            if policy_chosen == 'policy_Sweden':
                action = policy_Sweden(actions,week,state)
                state,r,done,i= envr.step(action = (action))
                states.append(state)

                rewards.append(r)
                week+=1
            
            elif policy_chosen == 'policy_Sweden_second':

                action = policy_Sweden_second(actions,week,state)
                
                state,r,done,i= envr.step(action = (action))

                states.append(state)

                rewards.append(r)
                week+=1
            

        rewards_per_problem.append(np.sum(rewards))
    return rewards_per_problem, states


