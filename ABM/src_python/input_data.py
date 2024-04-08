import pandas as pd
import numpy.random as random

def call_traffic_data(place:str):
    file_directory = f"{place}/survey_table.csv"
    traffic_data = pd.read_csv(file_directory)
    # traffic_data: save as pd.dataframe
    
    return traffic_data


def call_traffic_data_of_occupation_weekday(traffic_data,
                                            occupation:str,
                                            weekday):
    selection = (traffic_data['occupation']==occupation) & (traffic_data['weekday']==weekday)
    selected_data = traffic_data.loc[selection]
    
    return selected_data


def make_human_traffic_from_traffic_data(selected_traffic, n_agent:int):
    n_data = selected_traffic.shape[0]
    shuffle_row_number = random.permutation(n_data)
    sampled_rows = shuffle_row_number[:n_agent]
    
    random_traffic = selected_traffic.iloc[sampled_rows]
    # check iloc!!! vs [sampled_row]
    
    return random_traffic