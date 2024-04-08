using Agents
using DrWatson: @dict
using LightGraphs
using Distributions
include("input_data.jl")

struct PairIterator{A}
    pairs::Vector{Tuple{Int,Int}}
    agents::Dict{Int,A}
end

function Base.length(iter::PairIterator)
    length(iter.pairs)
end


function Base.iterate(iter::PairIterator, i=1)
    (i > length(iter)) && return nothing

    p = iter.pairs[i]
    id1, id2 = p

    return (iter.agents[id1], iter.agents[id2]), i + 1
end


@agent hospital_agents GraphAgent begin
    occupation::String # N, D, Wa, Wc, Wp, Wo, Wr, P, C
    main_room::String # where he/she is """mainly""".
    #= 32진법
    0: Outside
    1~k: Ward
    l: Physical therapy
    m: Operational therapy
    n: Robotic therapy
    o: Station
    p: Rest area
    q: Doctor room
    r: Bath room
    s~v: Undetermined spaces =#
    hours_pre_infectious::Int # as 0 at first
    hours_infectious::Int # as 0 at first
    pre_infectious_period::Int # Pre-infectious period
    infectious_period::Int # Infectious period
    quarantine::Bool # Quarantined by test_dates
    quarantine_tick::Int # How dates
    status::Symbol # 1: S, 2: E, 3: I, 4: R
    traffic::Vector
    static_traffic::Vector
    infect_by_me::Vector # agents infected by this agent
    have_I_gone_outside::Bool
end


function initialize_model(;
    place="severance",
    β,
    incubation_period_distn=Dirac(48),
    presymptomatic_infectious_period_distn=Dirac(48),
    infectious_period_distn=Dirac(48),
    prop_outside_infection=1e-5,
    is_only_am_therapy=false,
    test_number_per_week=0,
    traffic_data::DataFrame
)

    rng = MersenneTwister()

    if (test_number_per_week == 1)
        test_dates = [3]
    elseif (test_number_per_week == 2)
        test_dates = [1, 4]
    elseif (test_number_per_week == 3)
        test_dates = [1, 3, 5]
    else
        test_dates = []
    end

    properties = @dict(
        place,
        β,
        incubation_period_distn,
        presymptomatic_infectious_period_distn,
        infectious_period_distn,
        prop_outside_infection,
        is_only_am_therapy,
        test_number_per_week,
        traffic_data,
        test_dates
    )
    properties[:tick] = 0

    (place == "severance") ? (n_vert = 500) : (n_vert = 300)
    space = GraphSpace(complete_digraph(n_vert))
    model = ABM(hospital_agents, space; properties, rng)

    return model
end


function add_agent_who_at_position!(;
    model::ABM,
    occupation::String,
    main_room::String,
    status::Symbol,
    traffic=Vector(),
    static_traffic=Vector()
)
    pre_infectious_period = -1
    while pre_infectious_period < 0
        incubation_period = round(Int, rand(model.incubation_period_distn))
        presymptomatic_infectious_period = round(Int, rand(model.presymptomatic_infectious_period_distn))
        pre_infectious_period = incubation_period - presymptomatic_infectious_period
    end
    infectious_period = round(Int, rand(model.infectious_period_distn))
    quarantine = false
    quarantine_tick = 0
    infect_by_me = Int64[]

    pos = parse(Int, main_room, base=32) + 1
    add_agent!(pos, model, occupation, main_room, 0, 0,
        pre_infectious_period, infectious_period, quarantine, quarantine_tick,
        status, traffic, static_traffic, infect_by_me,
        false)
end


function hospital_agent_step!(agent, model)
    (model.tick%24 == 0) && (update_agents_traffics!(agent, model))
    migrate!(agent, model)
    add_pre_infectious_hour!(agent, model)
    be_infectious!(agent, model)
    add_infectious_hour!(agent, model)
    recover!(agent, model)
end


function hospital_model_step!(model)
    # transmission
    for (a1, a2) in all_pairs_vertex(model)
        # transmission doesn't occure for outside agents
        (a1.quarantine || a2.quarantine) && (continue)
        (((a1.pos - 1) % 32 == 0) || ((a2.pos - 1) % 32 == 0)) && continue
        transmit!(a1, a2, model)
    end

    for a in allagents(model)
        (model.test_number_per_week != 0) && (test_all!(a,model))
        (a.quarantine) && (a.quarantine_tick += 1)
        (a.quarantine_tick > 24 * 14) && (de_quarantine_agent!(a,model))
    end

    model.tick += 1
end

function de_quarantine_agent!(a,model)
    move_agent!(a, parse(Int,a.main_room,base=32)+1, model)
    a.quarantine = false
    a.quarantine_tick = 0
    a.have_I_gone_outside = false
end

function quarantine_agent!(a,model)
    move_agent!(a,1,model)
    a.quarantine = true
    a.quarantine_tick = 0
end

function update_agents_traffics!(agent, model)
    agent_occupation_traffic = call_traffic_data_of_occupation_weekday(
        model.traffic_data,
        agent.occupation,
        ((model.tick ÷ 24) % 7 == 0 || (model.tick ÷ 24) % 7 == 6) ? 0 : 1
    )

    if (agent.occupation == "C")
        corr_patient = (agent.main_room == "43") ? 2 : 8
        
        !isempty(model.agents[agent.id-corr_patient].traffic) && 
            (agent.traffic = deepcopy(model.agents[agent.id-corr_patient].static_traffic); 
            agent.static_traffic = deepcopy(agent.traffic);
            return;)
    end

    if (agent.occupation == "P")
        corr_caregiver = (agent.main_room == "43") ? 2 : 8

        !isempty(model.agents[agent.id+corr_caregiver].traffic) &&
            (agent.traffic = deepcopy(model.agents[agent.id+corr_caregiver].static_traffic); 
            agent.static_traffic = deepcopy(agent.traffic);
            return;)
    end

    isempty(agent_occupation_traffic) && (agent.traffic = zeros(Int64, 24); return;)

    n_data = nrow(agent_occupation_traffic)
    random_sampled_traffic = agent_occupation_traffic[rand(1:n_data), :]
    agent.traffic = Vector(random_sampled_traffic[8:31])
    agent.static_traffic = deepcopy(agent.traffic)
    if ((agent.occupation == "P" || agent.occupation == "C") && model.is_only_am_therapy)
        lunch_time = findall(x -> x == 1, agent.traffic[12:15])
        (isempty(lunch_time)) && (lunch_time = findall(x -> x == 20, agent.traffic[12:15]))
        lunch_time = lunch_time[1]
        agent.traffic[(11+lunch_time):end] .= 1
    end
end

#= 32진법
                0: Outside
1~k: Ward
l: Physical therapy
m: Operational therapy
n: Robotic therapy
o: Station
p: Rest area
q: Doctor room
r: Bath room
s~v: Undetermined spaces =#
#= 
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
화장실 (4층)    13 =#
function migrate!(agent, model)
    present_pos = agent.pos
    present_floor = agent.main_room[1]

    (agent.quarantine) && (return nothing)

    next_traffic = popfirst!(agent.traffic)

    if present_pos == 1 # outside infection
        (agent.status == :S && rand(model.rng) ≤ model.prop_outside_infection) && (agent.status = :E)
    end

    if next_traffic == 0 # to outside
        next_pos_32 = "0"
        agent.have_I_gone_outside = true
    elseif next_traffic == 1 # to ward
        if (agent.occupation == "P" || agent.occupation == "C")
            next_pos_32 = agent.main_room
        else
            ward_number_32 = string(rand(1:20), base=32)
            next_pos_32 = present_floor * ward_number_32
        end
    elseif next_traffic == 2 # to physical therapy room
        next_pos_32 = "3l"
    elseif next_traffic == 3 # to operational therapy room
        next_pos_32 = (agent.main_room[1] == '3' || agent.main_room[1] == '4') ? "3m" : "0m"
    elseif next_traffic == 4 # to robotic therapy room
        next_pos_32 = "0n"
    elseif next_traffic == 5 # to the (nurse) station
        next_pos_32 = present_floor * "o"
    elseif next_traffic == 6 # to the rest room
        next_pos_32 = present_floor * "p"
    else # otherwise, all positions are outside the model
        next_pos_32 = "0"
        agent.have_I_gone_outside = true
    end

    next_pos = parse(Int, next_pos_32, base=32) + 1

    if next_pos ≠ present_pos
        move_agent!(agent, next_pos, model)
    end
end


function add_pre_infectious_hour!(agent, model)
    (agent.status == :E) && (agent.hours_pre_infectious += 1)
end


function be_infectious!(agent, model)
    (agent.status == :E && agent.hours_pre_infectious ≥ agent.pre_infectious_period) &&
        (agent.status = :I)
end


function add_infectious_hour!(agent, model)
    (agent.status == :I) && (agent.hours_infectious += 1)
end


function recover!(agent, model)
    (agent.status == :I && agent.hours_infectious ≥ agent.infectious_period) &&
        (agent.status = :R)
end


function all_pairs_vertex(
    model::ABM{<:GraphSpace},
)
    pairs = Tuple{Int,Int}[]

    for a ∈ allagents(model)
        for nid ∈ ids_in_position(a, model)
            # sort by id
            new_pair = isless(a.id, nid) ? (a.id, nid) : (nid, a.id)
            (new_pair ∉ pairs) && push!(pairs, new_pair)
        end
    end

    return PairIterator(pairs, model.agents)
end


function transmit!(a1, a2, model)
    (a1.quarantine || a2.quarantine) && (return nothing)
    
    # If there's no infected one, nothing happens
    num_S = count(a.status == :S for a in (a1, a2))
    num_I = count(a.status == :I for a in (a1, a2))
    ~(isone(num_S) && isone(num_I)) && return

    # Choose who the infectious is 
    if (a1.status == :S)
        agent_S = a1
        agent_I = a2
    else
        agent_S = a2
        agent_I = a1
    end

    # Infect the susceptible agent (this impacts on the outside variable)
    if (rand(model.rng) ≤ model.β)
        agent_S.status = :E
        # To calculate the R_o for model
        append!(agent_I.infect_by_me, agent_S.id)
    end
end


function test_all!(a, model)
    current_day = (model.tick ÷ 24) % 7 # 0: 1st Sun, 1: 1st Mon, ... 6: 1st Sat, 7: 2nd Sun
    current_time = model.tick % 24

    (current_day in model.test_dates && a.have_I_gone_outside && a.status == :I && a.pos>1) &&
        (if current_time==9; quarantine_agent!(a,model); end)
end