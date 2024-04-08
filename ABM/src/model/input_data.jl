using CSV
using DataFrames
using Random

function call_traffic_data(place::String)
    file_directory = "../data/" * place * "/survey_table.csv"    
    traffic_data = CSV.read(file_directory, DataFrame)

    return traffic_data
end

function call_traffic_data_of_occupation_weekday(traffic_data::DataFrame, occupation::String, weekday)
    selection = (traffic_data.occupation .== occupation) .& (traffic_data.weekday .== weekday)
    selected_data = traffic_data[selection, :]
    
    return selected_data
end

function make_human_traffic_from_traffic_data(selected_traffic::DataFrame, n_agent::Int)
    n_data = nrow(selected_traffic)
    shuffle_row_number = shuffle(1:n_data)
    sampled_rows = shuffle_row_number[1:n_agent]
    
    random_traffic = selected_traffic[sampled_rows, :]

    return random_traffic
end