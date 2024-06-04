from mesa import Agent
import numpy as np
import copy
from traffic import *


class GraphAgent(Agent):
    # hospital_agents
    occ:str      # N, D, Wa, Wc, Wp, Wo, Wr, P, C
    mr:int       # where he/she is """mainly"""
    # 32진법
    # 0: Outside
    # 1~l: Ward
    # m: Physical therapy
    # n: Operational therapy
    # o: Robotic therapy
    # p: Station
    # q: Rest area
    # r: Doctor room
    # s: Bath room
    # t~w: Undetermined spaces
    h_E:int    # as 0 at first
    h_I:int        # as 0 at first
    E_p:int   # Pre-infectious period
    I_p:int       # Infectious period
    qrt:bool             # Quarantined by test_dates
    qrt_tick:int         # How dates
    stat:str                  # 1:S, 2:e, 3:I, 4:R
    traffic:np.ndarray
    static_traffic:np.ndarray
    infect_by_me:np.ndarray     # agents infectedby this agent
    have_I_gone_outside:bool
    
    
    def __init__(self, unique_id, pos, model, occ, mr,
                 h_E, h_I, E_p, I_p, qrt, qrt_tick, stat,
                 traffic, static_traffic, infect_by_me, have_I_gone_outside,
                 classifier):
        super().__init__(unique_id, model)
        self.pos = pos
        self.occ = occ
        self.mr= mr
        self.h_E = h_E
        self.h_I = h_I
        self.E_p = E_p
        self.I_p = I_p
        self.qrt = qrt
        self.qrt_tick = qrt_tick
        self.stat = stat
        self.traffic = traffic
        self.static_traffic = static_traffic        # traffic은 step에 따라서 하나씩 없앨 것!
        self.infect_by_me = infect_by_me
        self.have_I_gone_outside = have_I_gone_outside
        self.classifier = classifier
       
       
    def update_agents_traffics(self):
        week_num = (self.model.tick // 24) % 7
        if (week_num == 0) or (week_num == 6):
            weekday = 0     # weekend
        else:
            weekday = 1

        occ_traffic = call_traffic_data_of_occupation_weekday(
            self.model.traffic_data,
            self.occ,
            weekday
        )
        
        # 환자 동선이 있을 경우 환자 동선에 맞춰 보호자 동선 구성
        if self.occ == 3:
            corr_P = 'P' + self.classifier[1:]

            if len(self.model.agent_list[corr_P].traffic):
                self.traffic = copy.deepcopy(self.model.agent_list[corr_P].static_traffic)
                self.static_traffic = copy.deepcopy(self.traffic)
                return
        
        # 환자 동선이 없을 경우 보호자 동선에 맞춰 환자 동선 구성
        if self.occ == 2:
            corr_C = 'C' + self.classifier[1:]
            
            if len(self.model.agent_list[corr_C].traffic):
                self.traffic = copy.deepcopy(self.model.agent_list[corr_C].static_traffic)
                self.static_traffic = copy.deepcopy(self.traffic)
                return
            
        # 동선이 없을 경우 그날은 병원에 있지 않은 것으로 간주
        if occ_traffic.empty:
            self.traffic = np.zeros(24, dtype=np.int64)
            return
        
        # 동선 list 중에서 랜덤하게 선택
        n_data = len(occ_traffic)
        traffic_idx = np.random.randint(n_data)
        sampled_traffic = occ_traffic.iloc[traffic_idx]
        self.traffic = sampled_traffic[7:31]         # 7~31까지가 동선에 해당함
        self.static_traffic = copy.deepcopy(self.traffic)

        if (self.occ == 2 or self.occ == 3):
            if self.model.is_only_am_therapy:
                # 오전 근무만 할 시 점심 시간인 11시~3시 이후로 진료/치료 안함 (1: ward에 위치)
                lunch_time = np.where(self.traffic[11:15] == 1)[0]
                if len(lunch_time) == 0:
                    lunch_time = np.where(self.traffic[11:15] == 20)[0]
                lunch_time = lunch_time[0]
                self.traffic[10 + lunch_time:] = 1
    
    
    def step(self):
        self.migrate()
        
        # E -> I
        if self.stat == 1:
            self.h_E += 1
            if self.h_E >= self.E_p:
                self.stat = 2
        
        # I -> R
        elif self.stat == 2:
            self.h_I += 1
            if self.h_I >= self.I_p:
                self.stat = 3
        
    
    def de_quarantine_agent(self):
        pos = self.mr
        self.model.grid.move_agent(self, pos)
        self.qrt = False
        self.qrt_tick = 0
        self.have_I_gone_outside = False
        
        
    def quarantine_agent(self):
        self.model.grid.move_agent(self, 15)
        self.qrt = True
        self.qrt_tick = 0
                
    
    def migrate(self):
        # outside(0)
        # 7층 구조: 12 wards(1~12), station(13), restroom(14)
        # quarantine space(15)
        # other floor(16)
        
        present_pos = self.pos
        # present_floor = self.pos // 32
        # 층수는 32 * (floor-6)을 더해서 32진법 또는 k 진법으로 구현
        
        if self.qrt:
            return None
        
        next_traffic = self.traffic[0]
        self.traffic = self.traffic[1:]
        
        # outside infection
        if present_pos == 0:
            if (self.stat == 0) and (np.random.rand() <= self.model.prop_outside_infection):
                self.stat = 1
        
        # to outside
        if next_traffic == 0:
            next_pos = 0
            self.have_I_gone_outside = True
            
        # to ward
        elif next_traffic == 1:
            if (self.occ == 2) or (self.occ == 3):
                next_pos = self.mr
                
            else:
                # 만약 다른 층을 디자인 할 경우 main_floor가 7,8,10 인 경우와 9 인 경우 분리 필요!
                # 9가 main floor일 경우 해당 층에는 병실이 없기 때문!
                next_pos = np.random.randint(1, 13)
                
        # to the (nurse) station
        elif next_traffic == 5:
            next_pos = 13
            
        # to the restroom
        elif next_traffic == 6:
            next_pos = 14
            
        # other floor
        else:
            next_pos = 16
        
        if next_pos != present_pos:
            self.model.grid.move_agent(self, next_pos)
            
    
    def test_all(self):
        # 외부에 나갔다 온 사람들만을 대상으로 검사 진행
        if self.have_I_gone_outside:
            if self.stat == 2:
                self.quarantine_agent()
                        
    def step_2(self):
        if self.qrt:
            self.qrt_tick += 1
            if self.qrt_tick > 24*14:
                self.de_quarantine_agent()