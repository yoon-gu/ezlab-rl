from mesa import Agent

def susceptible(model):
    return sum(1 for a in model.schedule.agents if a.status == 'S')

def exposed(model):
    return sum(1 for a in model.schedule.agents if a.status == 'E')

def infected(model):
    return sum(1 for a in model.schedule.agents if a.status == 'I')

def recovered(model):
    return sum(1 for a in model.schedule.agents if a.status == 'R')