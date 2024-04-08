import numpy as np
from make_abm import *


def add_n_nurses_on_floor_whos_status(model, u_id_list, number:int, main_floor:int, status:str):
    main_room = str(main_floor - 6) + "l"
    
    for i in range(number):
        model.add_agent_who_at_position(unique_id=u_id_list[i],
                                        occupation="N",
                                        main_room=main_room,
                                        status=status)
        

def add_n_doctors_on_floor_whos_status(model, u_id_list, number:int, main_floor:int, status:str):
    main_room = str(main_floor - 6) + "n"
    
    for i in range(number):
        model.add_agent_who_at_position(unique_id=u_id_list[i],
                                        occupation="D",
                                        main_room=main_room,
                                        status=status)
        
        
def add_n_cleaners_on_floor_whos_status(model, u_id_list, number:int, main_floor:int, status:str):
    main_room = str(main_floor - 6) + "s"
    
    for i in range(number):
        model.add_agent_who_at_position(unique_id=u_id_list[i],
                                        occupation="Wc",
                                        main_room=main_room,
                                        status=status)
        
        
def add_n_transfers_on_floor_whos_status(model, u_id_list, number:int, main_floor:int, status:str):
    main_room = str(main_floor - 6) + "l"
    
    for i in range(number):
        model.add_agent_who_at_position(unique_id=u_id_list[i],
                                        occupation="Wa",
                                        main_room=main_room,
                                        status=status)
        
        
def add_n_physical_therapists_whos_status(model, u_id_list, number:int, status:str):
    main_room = "3l"
    
    for i in range(number):
        model.add_agent_who_at_position(unique_id=u_id_list[i],
                                        occupation="Wp",
                                        main_room=main_room,
                                        status=status)
        

def add_n_operational_therapists_whos_status_at_6th(model, u_id_list, number:int, status:str):
    main_room = "0m"
    
    for i in range(number):
        model.add_agent_who_at_position(unique_id=u_id_list[i],
                                        occupation="Wo",
                                        main_room=main_room,
                                        status=status)


def add_n_operational_therapists_whos_status_at_9th(model, u_id_list, number:int, status:str):
    main_room = "3m"
    
    for i in range(number):
        model.add_agent_who_at_position(unique_id=u_id_list[i],
                                        occupation="Wo",
                                        main_room=main_room,
                                        status=status)


def add_n_robotic_therapists_whos_status(model, u_id_list, number:int, status:str):
    main_room = "0n"
    
    for i in range(number):
        model.add_agent_who_at_position(unique_id=u_id_list[i],
                                        occupation="Wr",
                                        main_room=main_room,
                                        status=status)


def add_n_patients_on_floor_at_room_whos_status(model, u_id_list, number:int,main_floor:int,
                                                room_number:int, status:str):
    
    if (room_number > 21) or (room_number < 1):
        raise ValueError("Room number should be positive and less than 22.")
    
    main_room = str(main_floor - 6) + np.base_repr(room_number, base=32)
    
    for i in range(number):
        model.add_agent_who_at_position(unique_id=u_id_list[i],
                                        occupation="P",
                                        main_room=main_room,
                                        status=status)


def add_n_caregivers_on_floor_at_room_whos_status(model, u_id_list, number:int,main_floor:int,
                                                room_number:int, status:str):
    
    if (room_number > 21) or (room_number < 1):
        raise ValueError("Room number should be positive and less than 22.")
    
    main_room = str(main_floor - 6) + np.base_repr(room_number, base=32)
    
    for i in range(number):
        model.add_agent_who_at_position(unique_id=u_id_list[i],
                                        occupation="C",
                                        main_room=main_room,
                                        status=status)