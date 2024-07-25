import pandas as pd

def call_traffic_data(place:str):
    file_directory = f"{place}/survey_table.csv"
    traffic_data = pd.read_csv(file_directory)
    # traffic_data: save as pd.dataframe
    
    return traffic_data


def call_traffic_data_of_occupation_weekday(traffic_data,
                                            occupation,
                                            weekday):
    occ_list = ['N', 'T', 'P', 'C', 'Wo', 'Wr', 'Wp', 'Wc']
    selection = (traffic_data['occupation']==occ_list[occupation])\
        & (traffic_data['weekday']==weekday)
    selected_data = traffic_data.loc[selection]
    
    return selected_data