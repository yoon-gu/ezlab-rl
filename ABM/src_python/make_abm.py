from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid
from mesa.time import SimultaneousActivation
import networkx as nx
from scipy.stats import rv_discrete
import numpy as np
import pandas as pd
import copy
from input_data import *

def collect_occ_stat(model):
    P_S = 0  
    P_E = 0
    P_I = 0
    P_R = 0
    
    C_S = 0
    C_E = 0
    C_I = 0
    C_R = 0
    
    N_S = 0
    N_E = 0
    N_I = 0
    N_R = 0
    
    Wa_S = 0
    Wa_E = 0
    Wa_I = 0
    Wa_R = 0
    
    Wp_S = 0
    Wp_E = 0
    Wp_I = 0
    Wp_R = 0
    
    Wo_S = 0
    Wo_E = 0
    Wo_I = 0
    Wo_R = 0
    
    Wr_S = 0
    Wr_E = 0
    Wr_I = 0
    Wr_R = 0
    
    Wc_S = 0
    Wc_E = 0
    Wc_I = 0
    Wc_R = 0
    
    for agent in model.schedule.agents:
        occ = agent.occupation
        stat = agent.status
        
        if occ == "P":
            if stat == "S":
                P_S += 1
            elif stat == "E":
                P_E += 1
            elif stat == "I":
                P_I += 1
            else:
                P_R += 1
                
        elif occ == "C":
            if stat == "S":
                C_S += 1
            elif stat == "E":
                C_E += 1
            elif stat == "I":
                C_I += 1
            else:
                C_R += 1
                
        elif occ == "N":
            if stat == "S":
                N_S += 1
            elif stat == "E":
                N_E += 1
            elif stat == "I":
                N_I += 1
            else:
                N_R += 1
                
        elif occ == "Wa":
            if stat == "S":
                Wa_S += 1
            elif stat == "E":
                Wa_E += 1
            elif stat == "I":
                Wa_I += 1
            else:
                Wa_R += 1
                
        elif occ == "Wp":
            if stat == "S":
                Wp_S += 1
            elif stat == "E":
                Wp_E += 1
            elif stat == "I":
                Wp_I += 1
            else:
                Wp_R += 1
                
        elif occ == "Wo":
            if stat == "S":
                Wo_S += 1
            elif stat == "E":
                Wo_E += 1
            elif stat == "I":
                Wo_I += 1
            else:
                Wo_R += 1
                
        elif occ == "Wr":
            if stat == "S":
                Wr_S += 1
            elif stat == "E":
                Wr_E += 1
            elif stat == "I":
                Wr_I += 1
            else:
                Wr_R += 1
                
        elif occ == "Wc":
            if stat == "S":
                Wc_S += 1
            elif stat == "E":
                Wc_E += 1
            elif stat == "I":
                Wc_I += 1
            else:
                Wc_R += 1
    
    return {"P_num": [P_S, P_E, P_I, P_R],
            "C_num": [C_S, C_E, C_I, C_R],
            "N_num": [N_S, N_E, N_I, N_R],
            "Wa_num": [Wa_S, Wa_E, Wa_I, Wa_R],
            "Wp_num": [Wp_S, Wp_E, Wp_I, Wp_R],
            "Wo_num": [Wo_S, Wo_E, Wo_I, Wo_R],
            "Wr_num": [Wr_S, Wr_E, Wr_I, Wr_R],
            "Wc_num": [Wc_S, Wc_E, Wc_I, Wc_R]}
    

class PairIterator:
    def __init__(self, pairs, agents):
        self.pairs = pairs
        self.agents = agents
        self._index = 0

    def __len__(self) -> int:
        return len(self.pairs)


    # iterable object
    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= len(self):
            raise StopIteration
        
        pair = self.pairs[self._index]
        id1, id2 = pair
        agent1 = self.agents[id1]
        agent2 = self.agents[id2]
        self._index += 1
        
        return (agent1, agent2), self._index


class GraphAgent(Agent):
    # hospital_agents
    occupation:str      # N, D, Wa, Wc, Wp, Wo, Wr, P, C
    main_room:str       # where he/she is """mainly"""
    # 32진법
    # 0: Outside
    # 1~k: Ward
    # l: Physical therapy
    # m: Operational therapy
    # n: Robotic therapy
    # o: Station
    # p: Rest area
    # q: Doctor room
    # r: Bath room
    # s~v: Undetermined spaces
    hours_pre_infectious:int    # as 0 at first
    hours_infectious:int        # as 0 at first
    pre_infectious_period:int   # Pre-infectious period
    infectious_period:int       # Infectious period
    quarantine:bool             # Quarantined by test_dates
    quarantine_tick:int         # How dates
    status:str                  # 1:S, 2:e, 3:I, 4:R
    traffic:np.ndarray
    static_traffic:np.ndarray
    infect_by_me:np.ndarray     # agents infectedby this agent
    have_I_gone_outside:bool
    
    
    def __init__(self, unique_id, pos, model, occupation, main_room,
                 hours_pre_infectious, hours_infectious, pre_infectious_period,
                 infectious_period, quarantine, quarantine_tick, status,
                 traffic, static_traffic, infect_by_me, have_I_gone_outside):
        super().__init__(unique_id, model)
        self.pos = pos
        self.occupation = occupation
        self.main_room = main_room
        self.hours_pre_infectious = hours_pre_infectious
        self.hours_infectious = hours_infectious
        self.pre_infectious_period = pre_infectious_period
        self.infectious_period = infectious_period
        self.quarantine = quarantine
        self.quarantine_tick = quarantine_tick
        self.status = status
        self.traffic = traffic
        self.static_traffic = static_traffic
        self.infect_by_me = infect_by_me
        self.have_I_gone_outside = have_I_gone_outside
        
    
    def step(self):
        if self.model.tick % 24 == 0:
            self.update_agents_traffics()
        self.migrate()
        self.add_pre_infectious_hour()
        self.be_infectious()
        self.add_infectious_hour()
        self.recover()
        
    
    def de_quarantine_agent(self):
        pos = int(self.main_room, 32) + 1
        self.model.grid.move_agent(self, pos)
        self.quarantine = False
        self.quarantine_tick = 0
        self.have_I_gone_outside = False
        
        
    def quarantine_agent(self):
        self.model.grid.move_agent(self, 1)
        self.quarantine = True
        self.quarantine_tick = 0
        
        
    def update_agents_traffics(self):
        week_num = (self.model.tick // 24) % 7
        if (week_num == 0) or (week_num == 6):
            weekday = 0
        else:
            weekday = 1

        agent_occupation_traffic = call_traffic_data_of_occupation_weekday(
            self.model.traffic_data,
            self.occupation,
            weekday
        )
        
        if self.occupation == 'C':
            corr_patient = 2 if self.main_room == '43' else 8
        # ====================================
            if len(self.model.agents[self.unique_id - corr_patient].traffic):
                self.traffic = copy.deepcopy(self.model.agents[self.unique_id - corr_patient].static_traffic)
                self.static_traffic = copy.deepcopy(self.traffic)
                return
        # ====================================
        if self.occupation == 'P':
            corr_caregiver = 2 if self.main_room == '43' else 8
            
            if len(self.model.agents[self.unique_id + corr_caregiver].traffic):
                self.traffic = copy.deepcopy(self.model.agents[self.unique_id + corr_caregiver].static_traffic)
                self.static_traffic = copy.deepcopy(self.traffic)
                return
            
        if agent_occupation_traffic.empty:
            self.traffic = np.zeros(24, dtype=np.int64)
            return
                
        n_data = len(agent_occupation_traffic)
        random_index = np.random.randint(0, n_data)
        random_sampled_traffic = agent_occupation_traffic.iloc[random_index]
        self.traffic = random_sampled_traffic[7:31]
        self.static_traffic = copy.deepcopy(self.traffic)

        if (self.occupation == "P" or self.occupation == "C"):
            if self.model.is_only_am_therapy:
                lunch_time = np.where(self.traffic[11:15] == 1)[0]
                if len(lunch_time) == 0:
                    lunch_time = np.where(self.traffic[11:15] == 20)[0]
                lunch_time = lunch_time[0]
                self.traffic[10 + lunch_time:] = 1
                
    
    def migrate(self):
        present_pos = self.pos
        present_floor = self.main_room[0]
        
        if self.quarantine:
            return None
        
        next_traffic = self.traffic[0]
        self.traffic = self.traffic[1:]
        
        # outside infection
        if present_pos == 1:
            if (self.status == "S") and (np.random.rand() <= self.model.prop_outside_infection):
                self.status = "E"
        
        # to outside
        if next_traffic == 0:
            next_pos_32 = "0"
            self.have_I_gone_outside = True
            
        # to ward
        elif next_traffic == 1:
            if (self.occupation == "P") or (self.occupation == "C"):
                next_pos_32 = self.main_room
            else:
                ward_number_32 = np.base_repr(np.random.randint(1, 21), base=32)
                next_pos_32 = present_floor + ward_number_32
                
        # to physical therapy room
        elif next_traffic == 2:
            next_pos_32 = "3l"
            
        # to operational therapy room
        elif next_traffic == 3:
            if (self.main_room[0] == '3') or (self.main_room[0] == '4'):
                next_pos_32 = '3m'
            else:
                next_pos_32 = '0m'
        
        # to robotic therapy room
        elif next_traffic == 4:
            next_pos_32 = '0n'
            
        # to the (nurse) station
        elif next_traffic == 5:
            next_pos_32 = present_floor + "o"
            
        # to the restroom
        elif next_traffic == 6:
            next_pos_32 = present_floor + "p"
            
        # otherwise, all positions are outside the model
        else:
            next_pos_32 = "0"
            self.have_I_gone_outside = True
            
        next_pos = int(next_pos_32, 32) + 1
        
        if next_pos != present_pos:
            self.model.grid.move_agent(self, next_pos)
        
        
    def add_pre_infectious_hour(self):
        if self.status == 'E':
            self.hours_pre_infectious += 1
            
    
    def be_infectious(self):
        if (self.status == 'E') and (self.hours_pre_infectious >= self.pre_infectious_period):
            self.status = 'I'
            
    
    def add_infectious_hour(self):
        if self.status == 'I':
            self.hours_infectious += 1
    
    
    def recover(self):
        if (self.status == 'I') and (self.hours_infectious >= self.infectious_period):
            self.status = 'R'
            
    
    def test_all(self):
        current_day = (self.model.tick //24) % 7    # 0: 1st Sun, 1: 1st Mon, ... 6: 1st Sat, 7: 2nd Sun
        current_time = self.model.tick % 24
        
        if current_day in self.model.test_dates:
            if self.have_I_gone_outside:
                if self.status == 'I':
                    if current_time == 9:
                        self.quarantine_agent()
        
                
# ========================================================================================
"""
32진법
0: Outside
1~k: Ward
l: Physical therapy
m: Operational therapy
n: Robotic therapy
o: Station
p: Rest area
q: Doctor room
r: Bath room
s~v: Undetermined spaces
    -    0
병실    1
물리치료실    2
작업치료실    3
    로봇치료실    4
스테이션    5
휴게실    6
진찰실    7
청소도구실    8
오물처리실    9
인바디검사실 (6층)    10
계단경로    11
창고    12
화장실 (4층)    13
"""

class EpidemicsModel(Model):
    def __init__(self, beta,
                 traffic_data:pd.DataFrame,
                 place:str='severance',
                 incubation_period_distn = rv_discrete(name='dirac', values=([48], [1.0])),
                 multiplier_incubation_distn = 1,
                 presymptomatic_infectious_period_distn = rv_discrete(name='dirac', values=([48], [1.0])),
                 multiplier_presymptomatic_distn = 1,
                 infectious_period_distn = rv_discrete(name='dirac', values=([48], [1.0])),
                 multiplier_infectious_distn = 1,
                 prop_outside_infection = 1e-5,
                 is_only_am_therapy = False,
                 test_num_per_week = 0):
        
        super().__init__()
        self.place = place
        self.beta = beta
        
        self.incubation_period_distn = incubation_period_distn
        self.multiplier_incubation_distn = multiplier_incubation_distn
        
        self.presymptomatic_infectious_period_distn = presymptomatic_infectious_period_distn
        self.multiplier_presymptomatic_distn = multiplier_presymptomatic_distn
        
        self.infectious_period_distn = infectious_period_distn
        self.multiplier_infectious_distn = multiplier_infectious_distn
        
        self.prop_outside_infection =prop_outside_infection
        self.is_only_am_therapy = is_only_am_therapy
        self.test_number_per_week = test_num_per_week
        self.traffic_data = traffic_data
        self.tick = 0
        
        if self.test_number_per_week == 1:
            self.test_dates = [3]
        elif self.test_number_per_week == 2:
            self.test_dates = [1, 4]
        elif self.test_number_per_week == 3:
            self.test_dates = [1, 3, 5]
        else:
            self.test_dates = []
        
        n_vert = 500 if place == "severance" else 300
        graph = nx.complete_graph(n_vert)
        self.graph = graph
        self.grid = NetworkGrid(graph)
        self.schedule = SimultaneousActivation(self)
        
        self.datacollector = DataCollector(
            model_reporters={
                "occ_stat": collect_occ_stat
            }
        )
        
        
    def add_agent_who_at_position(self, unique_id:int, occupation:str, main_room:str, status:str,
                                    traffic:np.ndarray=[], static_traffic:np.ndarray=[]):
        """function for add agent(initial)

        Args:
            occupation (str): _description_
            main_room (str): _description_
            status (str): _description_
            traffic (np.ndarray): _description_
            static_traffic (np.ndarray): _description_
        """
        pre_infectious_period = -1
        while pre_infectious_period < 0:
            incubation_period = int(round(self.multiplier_incubation_distn * self.incubation_period_distn.rvs()))
            presymptomatic_infectious_period = int(round(self.multiplier_presymptomatic_distn * self.presymptomatic_infectious_period_distn.rvs()))
            pre_infectious_period = incubation_period - presymptomatic_infectious_period
            
        infectious_period = int(round(self.multiplier_infectious_distn * self.infectious_period_distn.rvs()))
        quarantine = False
        quarantine_tick = 0
        infect_by_me = []
        
        pos = int(main_room, 32) + 1
        
        agent = GraphAgent(unique_id=unique_id, pos=pos, model=self, occupation=occupation, main_room=main_room,
                           hours_pre_infectious=0, hours_infectious=0,
                           pre_infectious_period=pre_infectious_period, infectious_period=infectious_period,
                           quarantine=quarantine, quarantine_tick=quarantine_tick, status=status, traffic=traffic,
                           static_traffic=static_traffic, infect_by_me=infect_by_me, have_I_gone_outside=False)
        self.schedule.add(agent)
        self.grid.place_agent(agent, pos)
        
        
    def step(self):
        self.schedule.step()
        
        for (agent1, agent2), _ in self.all_pairs_vertex():
            if (agent1.quarantine or agent2.quarantine):
                continue
            
            if (agent1.pos % 32 == 0) and (agent2.pos % 32 == 0):
                continue
            
            self.transmit(agent1, agent2)
                    
        for agent in self.schedule.agents:
            if self.test_number_per_week != 0:
                agent.test_all()
            if agent.quarantine:
                agent.quarantine_tick += 1
                if agent.quarantine_tick > 24 * 14:
                    agent.de_quarantine_agent()
                    
        self.tick += 1
        self.datacollector.collect(self)
                
        
    def transmit(self, a1, a2):
        if (a1.quarantine or a2.quarantine):
            return None
        
        # If there's no infected one, nothing happens
        num_S = sum(a.status == 'S' for a in (a1, a2))
        num_I = sum(a.status == 'I' for a in (a1, a2))
        
        # Need one S and one I
        if not(num_S == 1 and num_I == 1):
            return
        
        # Choose who the infectious is
        if a1.status == 'S':
            agent_S = a1
            agent_I = a2
        else:
            agent_S = a2
            agent_I = a1
        
        # Infect the susceptible agent (this impacts on the outside variable)
        if np.random.rand() <= self.beta:
            agent_S.status = 'E'
            
            # To calculate the R_0 for model
            agent_I.infect_by_me.append(agent_S.unique_id)
        
    
    def all_pairs_vertex(self):
        pairs = []
        
        for agent in self.schedule.agents:
            pos_a = agent.pos
            ids = []
            
            for other_agent in self.schedule.agents:
                if (other_agent.unique_id != agent.unique_id) and (other_agent.pos == pos_a):
                    ids.append(other_agent.unique_id)
                    
            for nid in ids:
                if agent.unique_id < nid:
                    new_pair = (agent.unique_id, nid) 
                else:
                    new_pair = (nid, agent.unique_id)
                if new_pair not in pairs:
                    pairs.append(new_pair)

        return PairIterator(pairs, self.agents)


