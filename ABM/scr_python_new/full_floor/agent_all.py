from mesa import Agent
import numpy as np
import copy
from traffic import *

"""
**Place**
00: outside
01: quarantine room

42: physical therapy room (9 floor)
13: operational therapy room (6 floor)
43: operational therapy room (9 floor)
14: robotic therapy room (6 floor)

05: other area in hospital

*0: rest area
*1: (nurse) station (7, 8, 10 floor)
*2~: ward (7, 8, 10 floor)

* = 1: 6th floor
* = 2: 7th floor
* = 3: 8th floor
* = 4: 9th floor
* = 5: 10th floor


**Traffic**
0: outside
1: ward
2: physical therapy room
3: operational therapy room
4: robotic therapy room
5: (nurse) station
6: rest area
*: other area in hospital
"""


class GraphAgent(Agent):
    # hospital_agents
    occ:int                     # N, T, P, C, O, R, H, W
    mr:int                      # where he/she is """mainly"""
    
    h_E:int                     # time that he/she is at E, as 0 at first
    h_I:int                     # time that he/she is at i, as 0 at first
    E_p:int                     # Pre-infectious period
    I_p:int                     # Infectious period
    qrt:bool                    # Quarantined by test_dates
    qrt_tick:int                # How dates
    stat:int                    # 1:S, 2:E, 3:I, 4:R
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
                # 오전 치료만 할 시 점심 시간인 11시~3시 이후로 진료/치료 안함 (1: ward에 위치)
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
        
        present_pos = self.pos
        main_floor = self.mr // 10
        
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
            # patients and caregivers
            if (self.occ == 2) or (self.occ == 3):
                next_pos = self.mr
            
            # other occupation
            else:
                if main_floor in [1, 4]:
                    main_floor = np.random.choice([2, 3, 5])
                
                # 7th floor ward
                if main_floor == 2:
                    next_pos = 20 + np.random.randint(2, 6)
                
                # 8th floor ward
                elif main_floor == 3:
                    next_pos = 30 + np.random.randint(2, 8)
                
                # 10th floor ward
                else:
                    next_pos = 50 + np.random.randint(2, 5)
            
        # to the physical therapy room
        elif next_traffic == 2:
            next_pos = 42
        
        # to the operational therapy room
        elif next_traffic == 3:
            # 6, 7, 8th floor agents
            if main_floor <= 3:
                next_pos = 13
            
            # 9, 10th floor agents
            else:
                next_pos = 43
        
        # to the robotic therapy room
        elif next_traffic == 4:
            next_pos = 14
        
        # to the (nurse) station
        elif next_traffic == 5:
            if main_floor in [1, 4]:
                main_floor = np.random.choice([2, 3, 5])
            
            next_pos = main_floor * 10 + 1
            
        # to the rest area
        elif next_traffic == 6:
            next_pos = main_floor * 10
            
        # other floor
        else:
            next_pos = 5
        
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