
#from re import S
import gym
from gym.core import RewardWrapper
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from .seir import SEIR
from IPython.display import clear_output
from textwrap import wrap



class Epidemic(gym.Env):

	# wrapper environment around SEIR
	def __init__(self, problem_id, reward_id , period):

		"""
		SEIR MODEL
		
		Args:

			problem_id (int): problem id refers to the sensitivity analysis and the different weights given to the reward
			reward_id (int): id for different rewards which are run for comparison (0: EpidRLearn, 1: Nature Paper, 2: ECML paper)
			period (string): which period to run (spring for training, autumn for testing, FOHM for fitted data)
			print (boolean): If True returns s, r, d, info + the 2 parts the reward for plotting, if False returns s, r, d, info
		"""
		self.period = period # string: spring or autumn
		# calling the class SEIR()
		self.env = SEIR(period=self.period)

		self.actions = self.env.actions
		self.action_space = self.env.action_space
		self.observation_space = self.env.observation_space

		self.action_repeat = 7 # number of days between actions
		self.steps_total = int(182/self.action_repeat) # episode length which is 52 weeks
		#182
		self.steps = 0
		self.max_steps = 26 #half of the year for training and then testing


		# fitted parameters for spring period
		self.unreported_srping019 = 0.998416430249021
		self.unreported_srping2069 = 0.986717837070985
		self.unreported_srping70 = 0.8639400113485507
       
		# fitted parameters for autumn period
		self.unreported_autumn019 = 0.5755026399435202
		self.unreported_autumn2069 = 0.4763668186715981
		self.unreported_autumn70 = 0.01
        
		
		self.problem_id = problem_id  #problem ids for sensitiviry analysis
		self.reward_id = reward_id #reward ids to compare epidRLearn to other papers
		#rewards list of choices 
		self.rewards = [0, 1, 2]
		#problems list of choices
		self.problems = [0, 1, 2, 3, 4, 5, 6, 7]

		# getting the parameters for rwards and problem ids
		self.weights_choice = self.problems[self.problem_id]
		self.reward_choice = self.rewards[self.reward_id]


		#lists to store the population dynamics, actions and rewards for plotting. In the beginning of the pandemic these are empty
		self.action_history = []
		self.FreeToMove = []
		self.infectionState = []
		
		
	def reset(self):
		
		# reseting the params to the initial values for after the episode ends
		self.susceptible = np.array([self.env.Susceptible])
		self.exposed = np.array([self.env.Exposed])
		self.infected_ureported = np.array([self.env.Infected_ureported])
		self.infected_reported = np.array([self.env.Infected_reported])
		self.recovered_positive = np.array([self.env.Recovered1])
		self.recovered_negative = np.array([self.env.Recovered2])

		self.action_history = []
		self.FreeToMove = []

		self.steps = 0

		
		s = self.env.reset().reshape(-1)
		return s
		
	def step(self, action):

		"""
		action (int): the action to be taken
		
		"""

		# implementing the badget depletion for ecml paper comparison
		if self.reward_choice == 1:

			# this badget is for reward_id=1, the ecml paper
			if action == 1: 
				self.env.badget75 = self.env.badget75 - 1 

			if self.env.badget75<=0 and action == 1:
				action = 2

			if action == 2: 
				self.env.badget50 = self.env.badget50 - 1 

			if self.env.badget50<=0 and action == 2:
				action = 3


		self.action_history.append(action)

		#reward
		r = 0
		
		# how many steps have we done? 0 to 26 weeks
		steps = self.steps
	

		s, _, d, info = self.env.step(action,  steps)


		s = s.reshape(-1)


		# extracting from the observations the respective groups
		susceptible = s[0:3]
		exposed = s[3:6]
		infected_ureported = s[6:9]
		infected_reported = s[9:12]
		recovered_positive = s[12:15]
		recovered_negative = s[15:18]


		# and appending those to a list to be printed
		self.susceptible = np.vstack((self.susceptible, susceptible))
		self.exposed = np.vstack((self.exposed, exposed))
		self.infected_ureported = np.vstack((self.infected_ureported, infected_ureported))
		self.infected_reported = np.vstack((self.infected_reported, infected_reported))
		self.recovered_positive = np.vstack((self.recovered_positive, recovered_positive))
		self.recovered_negative = np.vstack((self.recovered_negative, recovered_negative))




		
		
		reward = self.reward(self.susceptible, self.exposed, self.infected_ureported, self.infected_reported,self.recovered_positive, self.recovered_negative, action)

		#appending and calculating the reward as returned 
		r += reward

		# increase the step count
		self.steps += 1


		# checking when the episode is done
		d = True if (self.steps >= self.steps_total) else False

		return s, r, d, info
		
	def reward(self, susceptible, exposed, infected_ureported, infected_reported, recovered_positive, recovered_negative, action):

		if self.reward_choice == 0:

	
			healthy_week =  int(susceptible[-1].sum() +  exposed[-1].sum()  + recovered_negative[-1].sum())/sum(self.env.N)

			# the current week 
			infectedUn_019 = (infected_ureported[-1][0] )
			infectedRe_019 = (infected_reported[-1][0] )

			infectedUn_2069 = (infected_ureported[-1][1] )
			infectedRe_2069 = (infected_reported[-1][1] )

			infectedUn_70 =  (infected_ureported[-1][2] )
			infectedRe_70 =  (infected_reported[-1][2] )

			# the previous week
			infectedUn_019_previous = (infected_ureported[-2][0] )
			infectedRe_019_previous = (infected_reported[-2][0] )

			infectedUn_2069_previous = (infected_ureported[-2][1] )
			infectedRe_2069_previous = (infected_reported[-2][1] )

			infectedUn_70_previous =  (infected_ureported[-2][2] )
			infectedRe_70_previous =  (infected_reported[-2][2] )


			# sum the infected reported and unreported per week/step (as calculated above) and get the percentage
			infected019 = (infectedUn_019 + infectedRe_019)/sum(self.env.N)
			infected2069 = (infectedUn_2069 + infectedRe_2069)/sum(self.env.N)
			infected70 = (infectedUn_70 + infectedRe_70) /sum(self.env.N)


			infected019_previous = (infectedUn_019_previous + infectedRe_019_previous)/sum(self.env.N)
			infected2069_previous = (infectedUn_2069_previous + infectedRe_2069_previous)/sum(self.env.N)
			infected70_previous = (infectedUn_70_previous + infectedRe_70_previous )/sum(self.env.N)

			# calculate the differce of the above
			difference_019 = (infected019 - infected019_previous)
			difference_2069 = (infected2069 - infected2069_previous)
			difference_70 = (infected70 - infected70_previous)

			infectionState = 0 


			if action == 0:
				# free to move, 100% of health population can contribute
				people_free_to_move =  healthy_week * 1

				if  difference_019 <= 0:
					infectionState += 0.5
				if  difference_2069 <= 0:
					infectionState += 0.5
				if  difference_70 <= 0:
					infectionState += 0.5

					
				if difference_019 > 0: 
					infectionState -= 0.5
				if difference_2069 > 0: 
					infectionState -= 0.5
				if difference_70 > 0:
					infectionState -= 0.5

			elif action == 3: 
				#light-movement restriction, 75% of healthy population (stay home if sick, wear masks, limit shops) can contribute 
				people_free_to_move =  healthy_week * 0.75 
				
				
				if  difference_019 > 0:
					infectionState += 0.5
				if  difference_2069 > 0:
					infectionState += 0.5
				if  difference_70 > 0:
					infectionState += 0.5

				if difference_2069 <= 0: 
					infectionState -= 0.5
				if difference_019 <= 0: 
					infectionState -= 0.5
				if difference_70 <= 0: 
					infectionState -= 0.5


			elif action == 1: 
				#full-movement restriction, 25% of healthy population (essential workers) can contribute 
				people_free_to_move = healthy_week * 0.25

				
				if  difference_019 > 0:
					infectionState += 0.5
				if  difference_2069 > 0:
					infectionState += 0.5
				if  difference_70 > 0:
					infectionState += 0.5

				if difference_019 <= 0: 
					infectionState -= 0.5
				if difference_2069 <= 0: 
					infectionState -= 0.5
				if difference_70 <= 0: 
					infectionState -= 0.5

			elif action == 2: 
				#semi-movement restriction, 50% of healthy population (essential workers + other workers) can contribute 
				people_free_to_move =  healthy_week * 0.50
				
				if  difference_019 > 0:
					infectionState += 0.5
				
				if  difference_2069 > 0:
					infectionState += 0.5

				if  difference_70 > 0:
					infectionState += 0.5


				if difference_019 <= 0: 
					infectionState -= 0.5
				if difference_2069 <= 0: 
					infectionState -= 0.5
				if difference_70 <= 0: 
					infectionState -= 0.5




			# for the sensitivity and extreme case analysis (problem_id)
			if self.weights_choice == 0:
				W1 = 0
				W2 = 1
			elif self.weights_choice == 1:
				W1 = 1 
				W2 = 0
			elif self.weights_choice == 2:
				W1 = 0.5
				W2 = 0.5
			elif self.weights_choice == 3:
				#epidrlearn
				W1 = 0.4 
				W2 = 0.6
			elif self.weights_choice == 4:
				W1 = 0.3 
				W2 = 0.7
			elif self.weights_choice == 5:
				W1 = 0.2
				W2 = 0.8


			

			reward = W1*people_free_to_move + W2 * infectionState

			return reward


		if self.reward_choice == 1: 

			# Libin et al. ECML, reduce the attack rate
			# To reduce the attack rate, we consider an immediate reward function that quantifies
			# the negative loss in susceptiblesceptibles over one simulated week
			# - (S - S'), includes a badget depletion which is implemented in step function
			
			reward = -(susceptible[-2].sum() - susceptible[-1].sum())/sum(self.env.N) 

			return reward
						
	def render(self, mode='human'):
		
		print(self.action_history)

		# unreported rated for spring and autumn 
		if self.period == 'spring':
			unreported_019 = self.unreported_srping019
			unreported_2069 = self.unreported_srping2069
			unreported_70 = self.unreported_srping70
		elif self.period == 'autumn' or self.period == 'FOHM':
			unreported_019 = self.unreported_autumn019
			unreported_2069 = self.unreported_autumn2069
			unreported_70 = self.unreported_autumn70


		# the colors for the action bars
		colours = ['white', '#F38383', '#AED2DE', '#CADDA9']
		incidenceUn = self.exposed[:,0]/sum(self.env.N) *  1/self.env.exposed_rate * unreported_019 + \
					self.exposed[:,1]/sum(self.env.N) *  1/self.env.exposed_rate * unreported_2069 + self.exposed[:,2]/sum(self.env.N) *  1/self.env.exposed_rate * unreported_70

		# length of action bars
		lengthExposed = {0: 1, 1: max(self.exposed[:,0]/sum(self.env.N)), 2: max(self.exposed[:,0]/sum(self.env.N))/2, 3: max(self.exposed[:,0]/sum(self.env.N))/3}
		lengthInfected = {0: 1, 1: max(self.infected_ureported[:,0]/sum(self.env.N)), 2: max(self.infected_ureported[:,0]/sum(self.env.N))/2, 3: max(self.infected_ureported[:,0]/sum(self.env.N))/3}
		lengthInfected2 = {0: 1, 1: max(self.infected_reported[:,0]/sum(self.env.N)), 2: max(self.infected_reported[:,0]/sum(self.env.N))/2, 3: max(self.infected_reported[:,0]/sum(self.env.N))/3}
		
		clear_output(wait=True)

		fig, axs = plt.subplots(1, 3)

		# plot the curves
		axs[0].plot(self.exposed[:,0]/sum(self.env.N),  label='Exposed 0-19', color = '#09289D')
		axs[0].plot(self.exposed[:,1]/sum(self.env.N), label='Exposed 20-69', color = '#9D0000')
		axs[0].plot(self.exposed[:,2]/sum(self.env.N), label='Exposed 70+',  color = '#108D27')
		axs[0].bar([i for i in range(len(self.action_history))],
		[self.action_history[i]  *  lengthExposed.get(self.action_history[i]) for i in range(len(self.action_history))], color = [colours[self.action_history[i]] for i in range(len(self.action_history))],
		 alpha=0.3)
		axs[0].set_xlabel('Weeks')
		axs[0].set_ylabel('Population %')
		axs[0].set_title('Exposed (Prevalence)')
		axs[0].legend()

		axs[1].plot(self.infected_ureported[:,0]/sum(self.env.N), label='Infected Unreported 0-19', color = '#09289D')
		axs[1].plot(self.infected_ureported[:,1]/sum(self.env.N), label='Infected Unreported 20-69', color = '#9D0000')
		axs[1].plot(self.infected_ureported[:,2]/sum(self.env.N), label='Infected Unreported 70+',  color = '#108D27')
		axs[1].bar([i for i in range(len(self.action_history))],
		[self.action_history[i] * lengthInfected.get(self.action_history[i]) for i in range(len(self.action_history))], color = [colours[self.action_history[i]] for i in range(len(self.action_history))],
		alpha=0.3)
		axs[1].set_xlabel('Weeks')
		axs[1].set_title('Infected Unreported (Prevalence)')
		axs[1].legend()
		
		axs[2].plot(self.infected_reported[:,0]/sum(self.env.N), label='Infected Reported 0-19', color = '#09289D')
		axs[2].plot(self.infected_reported[:,1]/sum(self.env.N), label='Infected Reported 20-69', color = '#9D0000')
		axs[2].plot(self.infected_reported[:,2]/sum(self.env.N), label='Infected Reported 70+',  color = '#108D27')
		axs[2].bar([i for i in range(len(self.action_history))],
		[self.action_history[i] * lengthInfected2.get(self.action_history[i]) for i in range(len(self.action_history))], color = [colours[self.action_history[i]] for i in range(len(self.action_history))],
		 alpha=0.3)
		axs[2].set_xlabel('Weeks')
		axs[2].set_title('Infected Reported (Prevalence)')
		axs[2].legend()
		

		plt.gcf().set_size_inches(16, 7)
		plt.legend(loc=2)
		plt.savefig(f"Figures/Policies_prob{self.weights_choice}_rwd{self.reward_choice}.png")
		# plt.savefig("Figures/Policies_problem{}.svg".format(self.weights_choice))
		

		#plotly to matplotlib for better visualization for the paper (after reviewers' comments)
		#plotting the incidence reported and unreported per age group 
		fig, axs = plt.subplots(1, 2)
		axs[0].plot(self.exposed[:,0]/sum(self.env.N) *  1/self.env.exposed_rate * unreported_019, label="Unreported 0-19 (Incidence)", color = '#09289D')
		axs[0].plot(self.exposed[:,1]/sum(self.env.N) *  1/self.env.exposed_rate * unreported_2069, label='Unreported 20-69 (Incidence)"', color = '#9D0000')
		axs[0].plot(self.exposed[:,2]/sum(self.env.N) *  1/self.env.exposed_rate * unreported_70, label='Unreported 70+ (Incidence)',  color = '#108D27')

		axs[0].set_xlabel('Weeks')
		axs[0].set_ylabel('Population %')
		axs[0].set_title('EpidRLearn Infected Unreported (Incidence)')
		axs[0].legend(loc=1)

		axs[1].plot((self.exposed[:,0]/sum(self.env.N) *  1/self.env.exposed_rate * (1-unreported_019)), label="Reported 0-19 (Incidence)", color = '#09289D')
		axs[1].plot((self.exposed[:,1]/sum(self.env.N) *  1/self.env.exposed_rate * (1-unreported_2069)), label='Reported 20-69 (Incidence)', color = '#9D0000')
		axs[1].plot((self.exposed[:,2]/sum(self.env.N) *  1/self.env.exposed_rate * (1-unreported_70)), label='Unreported 70+ (Incidence)',  color = '#108D27')

		axs[1].set_xlabel('Weeks')
		axs[1].set_title('EpidRLearn Infected Reported (Incidence)')
		axs[1].legend(loc=1)

		plt.gcf().set_size_inches(16, 7)
		plt.legend(loc=1)
		plt.savefig(f"Figures/IncidenceEpidLearnAgeGroups__prob{self.weights_choice}_rwd{self.reward_choice}.png")
		# plt.savefig("Figures/IncidenceEpidLearnAgeGroups_problem{}.svg".format(self.weights_choice))



		#plotting the incidence reported and unreported sum of all age groups
		fig = plt.figure()

		plt.plot(self.exposed[:,0]/sum(self.env.N) *  1/self.env.exposed_rate * unreported_019 + \
					self.exposed[:,1]/sum(self.env.N) *  1/self.env.exposed_rate * unreported_2069 + \
						 self.exposed[:,2]/sum(self.env.N) *  1/self.env.exposed_rate * unreported_70, \
							label="Infected Unreported (Incidence)", color = '#09289D' ) 
		plt.plot(self.exposed[:,0]/sum(self.env.N) *  1/self.env.exposed_rate * (1-unreported_019) + \
					self.exposed[:,1]/sum(self.env.N) *  1/self.env.exposed_rate * (1-unreported_2069)+ \
						self.exposed[:,2]/sum(self.env.N) *  1/self.env.exposed_rate * (1-unreported_70),\
							 label="Infected Reported (Incidence)", color = '#9D0000')
		plt.xlabel('Weeks')
		plt.ylabel('Population %')
		plt.title('EpidRLearn Infected (Incidence)')
		plt.legend(loc=1)
		plt.gcf().set_size_inches(8, 7)
		plt.savefig(f"Figures/IncidenceEpidLearnSUMAgeGroups__prob{self.weights_choice}_rwd{self.reward_choice}.png")
		# plt.savefig("Figures/IncidenceEpidLearnSUMAgeGroups_problem{}.svg".format(self.weights_choice))
	