import numpy as np

def collect_occ_stat(model):
    
    num_count = np.zeros((8,4))
    
    for agent in model.schedule.agents:
        occ = agent.occ
        stat = agent.stat
        
        num_count[occ, stat] += 1
    
    return num_count