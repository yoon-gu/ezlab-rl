from gym import spaces
import numpy as np
import gym
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import helpers.SDEs as SDEs
import math
import pandas as pd


#class SEIR(Agent):
class SEIR(gym.Env):
    def __init__(self, period, step_size=7):
        """
        ----------

        ----------
        step_size = the days in the week 
        observation_space (gym.spaces.Box, shape=(18,)): at each step, the environment only returns the true values S, I, E, R
        action_space (gym.spaces.Box, shape=(1)): the value beta
        """

        self.period = period
        
        self.N = np.array([566_994.0, 1528112.0, 279444.0])
        self.Susceptible = np.array([566_994.0, 1528112.0, 279444.0])
        self.Exposed = np.array([0.0, 0.0, 0.0])
        self.Infected_ureported = np.array([0.0, 75.0, 0.0]) # unreported
        self.Infected_reported = np.array([0.0, 10.0, 0.0]) # reported
        self.Recovered1 = np.array([0.0, 0.0, 0.0]) # PCR-positive
        self.Recovered2 = np.array([0.0, 0.0, 0.0]) # PCR-negative
  

        # the contact matrix before fitting
        self.contacts = np.array([7.452397574, 4.718777779, 0.290063626, 1.764106486, 8.544229624, 0.624169322, 0.443861795, 2.55482785, 1.69]).reshape(3, 3)
        # contact reduction = 1 and will be later become percentage
        self.contact_reduction = np.array([1.0] * 9).reshape(3,3) # (0-19, 20-69, 70+) x (0-19, 20-69, 70+)


        # infectivity per period
        self.infectivity019_spring = 0.2353349159354256
        self.infectivity2069_spring = 0.15661790850166916
        self.infectivity70_spring = 0.2598042750357819
        self.infectivity_spring = np.array([self.infectivity019_spring, self.infectivity2069_spring, self.infectivity70_spring]) # 0-19, 20-69, 70+]

        self.infectivity019_autumn = 0.4462199710193084   
        self.infectivity2069_autumn = 0.041112260686976825
        self.infectivity70_autumn = 0.05360966821275955
        self.infectivity_autumn = np.array([self.infectivity019_autumn, self.infectivity2069_autumn, self.infectivity70_autumn]) # 0-19, 20-69, 70+]


        #infectivity reduction per unreported, reported
        self.infectivity_reduction = np.array([0.5, 0.0]) 

        
        self.exposed_rate = 5.1
        self.recovery_rate = 5.0



        self.unreported_srping019 = 0.998416430249021
        self.unreported_srping2069 = 0.986717837070985
        self.unreported_srping70 = 0.8639400113485507
        self.unreported_spring = np.array([self.unreported_srping019, self.unreported_srping2069, self.unreported_srping70])


        self.unreported_autumn019 = 0.5755026399435202
        self.unreported_autumn2069 = 0.4763668186715981
        self.unreported_autumn70 = 0.01
        self.unreported_autumn = np.array([self.unreported_autumn019, self.unreported_autumn2069, self.unreported_autumn70])


        # the actions to be taken
        self.action_chosen = 0

        # the step size
        self.step_size = step_size

        self.badget75 = 6 #variable for the ecml paper comparison
        self.badget50 = 6 #variable for the ecml paper comparison

        # continuous observation space
        self.observation_space = spaces.Box(
            0, 2_374_550.0, shape=(18,), dtype=np.float64)  # check dtype

        self.actions = np.array([0, 1, 2, 3])
        # discrete action space
        self.action_space = gym.spaces.Discrete(4)

    def reset(self):
        

        self.N = np.array([566_994.0, 1528112.0, 279444.0])
        self.Susceptible = np.array([566_994.0, 1528112.0, 279444.0])
        self.Exposed = np.array([0.0, 0.0, 0.0])
        self.Infected_ureported = np.array([0.0, 75.0, 0.0]) # unreported
        self.Infected_reported = np.array([0.0, 10.0, 0.0]) # reported
        self.Recovered1 = np.array([0.0, 0.0, 0.0]) # PCR-positive
        self.Recovered2 = np.array([0.0, 0.0, 0.0]) # PCR-negative
        

        self.actions = np.array([0, 1, 2, 3])
        self.contacts = np.array([7.452397574, 4.718777779, 0.290063626, 1.764106486, 8.544229624, 0.624169322, 0.443861795, 2.55482785, 1.69]).reshape(3, 3)
        self.contact_reduction = np.array([1.0] * 9).reshape(3,3) # (0-19, 20-69, 70+) x (0-19, 20-69, 70+)

        self.infectivity019_spring = 0.2353349159354256
        self.infectivity2069_spring = 0.15661790850166916
        self.infectivity70_spring = 0.2598042750357819
        self.infectivity_spring = np.array([self.infectivity019_spring, self.infectivity2069_spring, self.infectivity70_spring]) # 0-19, 20-69, 70+]

        self.infectivity019_autumn = 0.4462199710193084   
        self.infectivity2069_autumn = 0.041112260686976825
        self.infectivity70_autumn = 0.05360966821275955
        self.infectivity_autumn = np.array([self.infectivity019_autumn, self.infectivity2069_autumn, self.infectivity70_autumn]) # 0-19, 20-69, 70+]

        self.unreported_srping019 = 0.998416430249021
        self.unreported_srping2069 = 0.986717837070985
        self.unreported_srping70 = 0.8639400113485507
        self.unreported_spring = np.array([self.unreported_srping019, self.unreported_srping2069, self.unreported_srping70])


        self.unreported_autumn019 = 0.5755026399435202
        self.unreported_autumn2069 = 0.4763668186715981
        self.unreported_autumn70 = 0.01
        self.unreported_autumn = np.array([self.unreported_autumn019, self.unreported_autumn2069, self.unreported_autumn70])

  
        self.infectivity_reduction = np.array([0.5, 0.0]) # unreported, reported
        
        self.exposed_rate = 5.1
        self.recovery_rate = 5.0

        #0:level0
        #1:level3
        #2:level2
        #3:level1
        self.badget75 = 6 # for the ecml paper comparison
        self.badget50 = 6 #variable for the ecml paper comparison


        # concating the obervation space to an numpy array
        
        self.state = np.array([
            self.Susceptible, # S
            self.Exposed, # E
            self.Infected_ureported, # I
            self.Infected_reported,
            self.Recovered1,
            self.Recovered2])


        return self.state

    def step(self, action=None, steps=None):
        """performs integration step"""

        if self.state.shape[0] != 18:
            self.state = np.concatenate((self.state[0], self.state[1], self.state[2], self.state[3], self.state[4], self.state[5]))

        # integration and solving
        self.state = self.solver(self.state, action, steps)

        info = {}



        return self.state, 0, False, info

    # implementation of contact reduction for benchmark sweden fitted
    def generate_contact_profile(self, start_days, nr_change_days, contact_changes):
        contact_profile = list()
            
        # initial flat
        for x in range(0, start_days[0]-1):
            contact_profile.append(contact_changes[0])

        # changes
        for x in range(0, len(nr_change_days)):       
            # contact change
            contact_diff = contact_changes[x] - contact_changes[x+1]
            change_per_day = contact_diff / (nr_change_days[x] + 1)
            multiplier = 1

            for y in range(start_days[x], start_days[x] + nr_change_days[x]): 
                contact_profile.append(contact_changes[x] - change_per_day * multiplier)
                multiplier += 1

            # contact flat   
            for y in range(start_days[x]+nr_change_days[x], start_days[x+1]):
                contact_profile.append(contact_changes[x+1])

        return contact_profile

    def solver(self, X, action, steps):
        """Solve the ODEs with scipy package.

        Parameters
        ----------
        X : self.state of SEIR

        action: int, action taken 

        period: string, spring eller autumn eller FOHM


        Returns
        -------
        y : 1D NumPy array
        
        """
        # Arguments used by seir_ode
        self.action_chosen = action
        self.steps = steps 



        X_ = np.array(X)


        t = np.linspace(0, self.step_size, num=7)

       
        dxdt = odeint(self.seir_ode, X_, t)


        return dxdt[-1]


    def render(self, mode = 'human'):

        pass


    #@staticmethod
    def seir_ode(self, init, time):


        '''
        SEIR ODEs
        '''
        exposed_rate = self.exposed_rate
        recovery_rate = self.recovery_rate
        infectivity_autumn = self.infectivity_autumn
        infectivity_spring = self.infectivity_spring
        unreported_spring = self.unreported_spring
        unreported_autumn = self.unreported_autumn
        infectivity_reduction = self.infectivity_reduction
        population = self.N
        contacts = self.contacts
        period = self.period

        if period == 'spring':
           #level0
            if self.action_chosen == 0:
                contact_reduction = self.contact_reduction * 1

            #level3
            if self.action_chosen == 1:
                contact_reduction = self.contact_reduction * 0.25

            #level2
            if self.action_chosen == 2:

                contact_reduction = self.contact_reduction * 0.50
            #level1
            if self.action_chosen == 3:

                contact_reduction = self.contact_reduction * 0.75
            
            unreported = unreported_spring

            infectivity = infectivity_spring

            transmission_rate = contacts * contact_reduction * infectivity / self.N

        elif period == 'autumn':
            
            if self.action_chosen == 0:
                contact_reduction = self.contact_reduction * 1

            if self.action_chosen == 1:
                contact_reduction = self.contact_reduction * 0.25
    
            if self.action_chosen == 2:
                contact_reduction = self.contact_reduction * 0.50

            if self.action_chosen == 3:
                contact_reduction = self.contact_reduction * 0.75
            infectivity = infectivity_autumn
            unreported = unreported_autumn

            transmission_rate = contacts * contact_reduction * infectivity / self.N


        elif period == 'FOHM':

            infectivity = infectivity_autumn
            unreported = unreported_autumn

            start_day_period1 = 18+2 #76
            start_day_period2 = 198 - 179 +2 #256
            start_day_period3 = 256 - 179 +2 #323
            days_change_period1 = 14
            days_change_period2 = 22
            days_change_period3 = 60
            contact_change_period1 = 0.19184387072616624
            contact_change_period2 = 0.3211756937149402
            contact_change_period3 = 0.5516354340792019

            contact_profile = self.generate_contact_profile([start_day_period2, start_day_period3, 182+1],\
                [days_change_period2, days_change_period3], [1.0,contact_change_period2, contact_change_period3])
            
            transmission_rate = ((contacts * contact_profile[math.floor(self.steps * 7)] * infectivity).T / population).T




        
        susceptible, exposed, infected_u, infected_r, recovered1, recovered2 = np.split(init, [3, 6, 9, 12, 15])


        

        dS_out = ( ((transmission_rate*susceptible).T).dot( (1-infectivity_reduction[0]) * infected_u + (1-infectivity_reduction[1]) * infected_r) )
        dE_out = exposed * 1/exposed_rate
        dI_u_out = infected_u * 1/recovery_rate
        dI_r_out = infected_r * 1/recovery_rate
        dR1_out = recovered1 * 1/recovery_rate

        dS = -dS_out
        dE = dS_out - dE_out
        dI_u = (dE_out * unreported) - dI_u_out
        dI_r = (dE_out * ([1 - value for value in unreported])) - dI_r_out
        dR1 = (dI_u_out + dI_r_out) - dR1_out
        dR2 = dR1_out

        return np.concatenate((dS, dE, dI_u, dI_r, dR1, dR2))


    
    def close(self):
        pass
		