from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid
from mesa.time import BaseScheduler
import networkx as nx
from scipy.stats import rv_discrete
import numpy as np
import pandas as pd
from agent_7 import *
from data_collector import *


class PairIterator:
    def __init__(self, pairs, agent_list):
        self.pairs = pairs
        self.agent_list = agent_list
        self._index = 0

    # def __len__(self) -> int:
    #     return len(self.pairs)


    # iterable object
    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= len(self.pairs):
            raise StopIteration
        
        id1, id2 = self.pairs[self._index]
        agent1 = self.agent_list[id1]
        agent2 = self.agent_list[id2]
        self._index += 1
        
        return (agent1, agent2), self._index


class EpidemicsModel(Model):
    def __init__(self, beta,
                 traffic_data:pd.DataFrame,
                 place:str='severance',
                 incub_p_dist = rv_discrete(name='dirac', values=([48], [1.0])),
                 m_incub_p_dist = 1,
                 presym_I_p_dist = rv_discrete(name='dirac', values=([48], [1.0])),
                 m_presym_I_p_dist = 1,
                 I_p_dist = rv_discrete(name='dirac', values=([48], [1.0])),
                 m_I_p_dist = 1,
                 prop_outside_infection = 1e-5,
                 is_only_am_therapy = False,
                 test_num_per_week = 0):
        
        super().__init__()
        self.place = place
        self.beta = beta
        
        self.incub_p_dist = incub_p_dist
        self.m_incub_p_dist = m_incub_p_dist
        
        self.presym_I_p_dist = presym_I_p_dist
        self.m_presym_I_p_dist = m_presym_I_p_dist
        
        self.I_p_dist = I_p_dist
        self.m_I_p_dist = m_I_p_dist
        
        self.prop_outside_infection =prop_outside_infection
        self.is_only_am_therapy = is_only_am_therapy
        self.num_test = test_num_per_week
        self.traffic_data = traffic_data
        self.tick = 0
        
        if self.num_test == 1:
            self.test_dates = [3]
        elif self.num_test == 2:
            self.test_dates = [1, 4]
        elif self.num_test == 3:
            self.test_dates = [1, 3, 5]
        else:
            self.test_dates = []
        
        n_vert = 17
        graph = nx.complete_graph(n_vert)
        self.graph = graph
        self.grid = NetworkGrid(graph)
        self.schedule = BaseScheduler(self)
        
        self.datacollector = DataCollector(
            model_reporters={
                "occ_stat": collect_occ_stat
            }
        )
        
        self.agent_list = {}
        self.unique_id = 0
        
        
    def add_agent(self, occ:str, mr:str, stat:str, classifier:str,
                  traffic:np.ndarray=[], static_traffic:np.ndarray=[]):
        """function for add agent(initial)

        Args:
            occupation (str): _description_
            main_room (str): _description_
            stat (str): _description_
            traffic (np.ndarray): _description_
            static_traffic (np.ndarray): _description_
        """
        E_p = -1
        while E_p < 0:
            incubation_period = int(round(self.m_incub_p_dist * self.incub_p_dist.rvs()))
            presymp_I_p = int(round(self.m_presym_I_p_dist * self.presym_I_p_dist.rvs()))
            E_p = incubation_period - presymp_I_p
            
        I_p = int(round(self.m_I_p_dist * self.I_p_dist.rvs()))
        qrt = False
        qrt_tick = 0
        infect_by_me = []
        
        pos = mr
        
        agent = GraphAgent(unique_id=self.unique_id, pos=pos, model=self, occ=occ, mr=mr,
                           h_E=0, h_I=0, E_p=E_p, I_p=I_p,
                           qrt=qrt, qrt_tick=qrt_tick, stat=stat, traffic=traffic,
                           static_traffic=static_traffic, infect_by_me=infect_by_me,
                           have_I_gone_outside=False, classifier = classifier)
        self.schedule.add(agent)
        self.agent_list[classifier] = agent
        self.grid.place_agent(agent, pos)
        self.unique_id += 1
        
        
    def step(self):
        # update traffic
        if self.tick%24 == 0:
            self.schedule.do_each("update_agents_traffics")
            
        # move agent
        self.schedule.do_each("step")

        # transition
        for (agent1, agent2), _ in self.all_pairs_vertex():
            # outside
            if agent1.pos == 0:
                continue
            
            # quarantine room
            if agent1.pos == 15:
                continue
            
            self.transmit(agent1, agent2)
            
        # test agent (visit outside)
        if self.num_test != 0:
            current_day = (self.tick //24) % 7    # 0: 1st Sun, 1: 1st Mon, ... 6: 1st Sat, 7: 2nd Sun
            current_time = self.tick % 24
            
            if current_day in self.test_dates:
                if current_time == 9:
                    self.schedule.do_each("test_all")
        
        # update qrt tick
        self.schedule.do_each("step_2")
        
        self.schedule.steps += 1
        self.schedule.time += 1
                    
        self.tick += 1
        self.datacollector.collect(self)
                
        
    def transmit(self, a1, a2):
        
        if a1.stat == 0:
            if a2.stat == 2:
                if np.random.rand() <= self.beta:
                    a1.stat = 1
                    a2.infect_by_me.append(a1.unique_id)
                    
        elif a1.stat == 2:
            if a2.stat == 0:
                if np.random.rand() <= self.beta:
                    a1.stat = 1
                    a2.infect_by_me.append(a1.unique_id)
        
        return
        
    
    def all_pairs_vertex(self):
        pairs = set()
        
        for agent in self.schedule.agents:
            pos_a = agent.pos
            id_a = agent.classifier
            ids = []
            
            other_agents = self.grid.get_cell_list_contents([pos_a])
            for other_agent in other_agents:
                ids.append(other_agent.classifier)
            
            ids.remove(id_a)
            
            for nid in ids:
                new_pair = tuple(sorted([id_a, nid]))
                pairs.add(new_pair)
                
        pairs = list(pairs)

        return PairIterator(pairs, self.agent_list)


