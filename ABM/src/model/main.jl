include("input_data.jl")
include("make_abm.jl")
include("collect_data.jl")
include("add_agents.jl")
using CairoMakie
using StatsBase

sev_traffic_data = call_traffic_data("severance")

sev_model = initialize_model(
    place="severance",
    β=0.001,
    incubation_period_distn=24 * LogNormal(1.857, 0.547),
    presymptomatic_infectious_period_distn=24 * TruncatedNormal(2.3, 0.49, 0, Inf),
    infectious_period_distn=24 * TruncatedNormal(7.2, 4.96, 0, Inf),
    is_only_am_therapy=true,
    test_number_per_week=1,
    traffic_data=sev_traffic_data
)

# Add nurses
n_nurses = 19 + 17 + 12
add_n_nurses_on_floor_whos_status!(;
    model=sev_model, number=19, main_floor=7, status=:S
)
add_n_nurses_on_floor_whos_status!(;
    model=sev_model, number=17, main_floor=8, status=:S
)
add_n_nurses_on_floor_whos_status!(;
    model=sev_model, number=12, main_floor=10, status=:S
)

# Add transferers
n_transferers = 2 + 6 + 2 + 2
add_n_transfers_on_floor_whos_status!(;
    model=sev_model, number=2, main_floor=7, status=:S
)
add_n_transfers_on_floor_whos_status!(;
    model=sev_model, number=6, main_floor=8, status=:S
)
add_n_transfers_on_floor_whos_status!(;
    model=sev_model, number=2, main_floor=9, status=:S
)
add_n_transfers_on_floor_whos_status!(;
    model=sev_model, number=2, main_floor=10, status=:S
)

# Add patients and caregivers
n_patients = 8 * 4 + 8 * 6 + 8 * 2 + 2
n_caregivers = n_patients
# 7th floor
add_n_patients_on_floor_at_room_whos_status!(;
    model=sev_model, number=8, main_floor=7, room_number=1, status=:S
)
add_n_caregivers_on_floor_at_room_whos_status!(;
    model=sev_model, number=7, main_floor=7, room_number=1, status=:S
)
add_n_caregivers_on_floor_at_room_whos_status!(;
    model=sev_model, number=1, main_floor=7, room_number=1, status=:I
)
for rn = 2:4
    add_n_patients_on_floor_at_room_whos_status!(;
        model=sev_model, number=8, main_floor=7, room_number=rn, status=:S
    )
    add_n_caregivers_on_floor_at_room_whos_status!(;
        model=sev_model, number=8, main_floor=7, room_number=rn, status=:S
    )
end
# # 8th floor
for rn = 1:6
    add_n_patients_on_floor_at_room_whos_status!(;
        model=sev_model, number=8, main_floor=8, room_number=rn, status=:S
    )
    add_n_caregivers_on_floor_at_room_whos_status!(;
        model=sev_model, number=8, main_floor=8, room_number=rn, status=:S
    )
end
# # 10th floor
for rn = 1:2
    add_n_patients_on_floor_at_room_whos_status!(;
        model=sev_model, number=8, main_floor=10, room_number=rn, status=:S
    )
    add_n_caregivers_on_floor_at_room_whos_status!(;
        model=sev_model, number=8, main_floor=10, room_number=rn, status=:S
    )
end
add_n_patients_on_floor_at_room_whos_status!(;
    model=sev_model, number=2, main_floor=10, room_number=3, status=:S
)
add_n_caregivers_on_floor_at_room_whos_status!(;
    model=sev_model, number=2, main_floor=10, room_number=3, status=:S
)

# Add operational therapists
n_operational_therapists = 20 + 3
add_n_operational_therapists_whos_status_at_6th!(;
    model=sev_model, number=20, status=:S
)

add_n_operational_therapists_whos_status_at_9th!(;
    model=sev_model, number=3, status=:S
)


# Add robotics therapists
n_robotic_therapists = 6
add_n_robotic_therapists_whos_status!(;
    model=sev_model, number=n_robotic_therapists, status=:S
)

# Add physical therapists
n_physical_therapists = 26
add_n_physical_therapists_whos_status!(;
    model=sev_model, number=n_physical_therapists, status=:S
)

# Add cleaners
n_cleaners = 1 + 2 + 2 + 1 + 2
add_n_cleaners_on_floor_whos_status!(;
    model=sev_model, number=1, main_floor=6, status=:S
)
add_n_cleaners_on_floor_whos_status!(;
    model=sev_model, number=2, main_floor=7, status=:S
)
add_n_cleaners_on_floor_whos_status!(;
    model=sev_model, number=2, main_floor=8, status=:S
)
add_n_cleaners_on_floor_whos_status!(;
    model=sev_model, number=1, main_floor=9, status=:S
)
add_n_cleaners_on_floor_whos_status!(;
    model=sev_model, number=2, main_floor=10, status=:S
)

n_simulation = 100
sev_model_vector = Vector{ABM}(UndefInitializer(), n_simulation)
for i = 1:n_simulation
    sev_model_vector[i] = deepcopy(sev_model)
end

to_collected = [:status, :pos, :occupation]

T = 24 * 90
# rst =  Vector{DataFrame}(UndefInitializer(), n_simulation)
# Threads.@threads for i = 1:n_simulation
#     rst[i], _ = run!(sev_model_vector[i], hospital_agent_step!, hospital_model_step!, T; adata=to_collected)
#     insertcols!(rst[i], 6, :ensemble => i)
# end

# rst = vcat(rst...)

rst, _, _ = ensemblerun!(sev_model_vector, hospital_agent_step!, hospital_model_step!, T; adata=to_collected)

figure = Figure(resolution=(1080, 1920))

ax1 = figure[1, 1] = Axis(figure, xlabel="Outbreak size of patient", ylabel="Frequency")
ax2 = figure[1, 2] = Axis(figure, xlabel="Outbreak size of caregiver", ylabel="Frequency")
ax3 = figure[2, 1] = Axis(figure, xlabel="Outbreak size of nurse", ylabel="Frequency")
ax4 = figure[2, 2] = Axis(figure, xlabel="Outbreak size of transfer squads", ylabel="Frequency")
ax5 = figure[3, 1] = Axis(figure, xlabel="Outbreak size of physical therapists", ylabel="Frequency")
ax6 = figure[3, 2] = Axis(figure, xlabel="Outbreak size of operational therapists", ylabel="Frequency")
ax7 = figure[4, 1] = Axis(figure, xlabel="Outbreak size of robotic therapists", ylabel="Frequency")
ax8 = figure[4, 2] = Axis(figure, xlabel="Outbreak size of cleaners", ylabel="Frequency")

outbreak_size = zeros(Int64, 8, n_simulation)

occ_index = 1
for occ in ["P", "C", "N", "Wa", "Wp", "Wo", "Wr", "Wc"]
    sub_rst = rst[(rst.occupation.==occ), :]
    for ens in 1:n_simulation
        ens_rst = sub_rst[(sub_rst.ensemble.==ens), :]
        ens_rst_last = ens_rst[(ens_rst.step.==maximum(ens_rst.step)), :]
        e_number = count(ens_rst_last.status .== :E)
        i_number = count(ens_rst_last.status .== :I)
        r_number = count(ens_rst_last.status .== :R)
        outbreak_size[occ_index, ens] = e_number + i_number + r_number
    end
    occ_index += 1
end

outbreak_df = DataFrame(outbreak_size', :auto)

rename!(outbreak_df,
    [:patient, :caregiver, :nurse, :assign, :physical_therapist, :operation_therapist, :robotic_therapist, :cleaner]
)

CSV.write("outbreak_size_data.csv", outbreak_df)

hist!(ax1, outbreak_size[1, :]; bins=0:1:(n_patients+1), color=:blue, normalization=:pdf)
hist!(ax2, outbreak_size[2, :]; bins=0:1:(n_caregivers+1), color=:black, normalization=:pdf)
hist!(ax3, outbreak_size[3, :]; bins=0:1:(n_nurses+1), color=:pink, normalization=:pdf)
hist!(ax4, outbreak_size[4, :]; bins=0:1:(n_transferers+1), color=:cyan, normalization=:pdf)
hist!(ax5, outbreak_size[5, :]; bins=0:1:(n_physical_therapists+1), color=:skyblue, normalization=:pdf)
hist!(ax6, outbreak_size[6, :]; bins=0:1:(n_operational_therapists+1), color=:gray, normalization=:pdf)
hist!(ax7, outbreak_size[7, :]; bins=0:1:(n_robotic_therapists+1), color=:red, normalization=:pdf)
hist!(ax8, outbreak_size[8, :]; bins=0:1:(n_cleaners+1), color=:green, normalization=:pdf)

quantile_patient = quantile(outbreak_size[1, :])
quantile_caregiver = quantile(outbreak_size[2, :])
quantile_nurse = quantile(outbreak_size[3, :])
quantile_transferer = quantile(outbreak_size[4, :])
quantile_physical_therapists = quantile(outbreak_size[5, :])
quantile_operational_therapists = quantile(outbreak_size[6, :])
quantile_robotic_therapists = quantile(outbreak_size[7, :])
quantile_cleaners = quantile(outbreak_size[8, :])

#########################################
figure2 = Figure(resolution=(1920, 1080))

n_susceptible = zeros(Int64, T + 1, n_simulation)
n_exposed = zeros(Int64, T + 1, n_simulation)
n_infectious = zeros(Int64, T + 1, n_simulation)
n_recovered = zeros(Int64, T + 1, n_simulation)

for ens in 1:n_simulation
    ens_rst = rst[(rst.ensemble.==ens), :]
    for t in 1:(T+1)
        ens_rst_at_t = ens_rst[(ens_rst.step.==(t-1)), :]
        n_susceptible[t, ens] = count(ens_rst_at_t.status .== :S)
        n_exposed[t, ens] = count(ens_rst_at_t.status .== :E)
        n_infectious[t, ens] = count(ens_rst_at_t.status .== :I)
        n_recovered[t, ens] = count(ens_rst_at_t.status .== :R)
    end
end
mean_susceptible = vec(mean(n_susceptible, dims=2))
mean_exposed = vec(mean(n_exposed, dims=2))
mean_infectious = vec(mean(n_infectious, dims=2))
mean_recovered = vec(mean(n_recovered, dims=2))


ax = figure2[1, 1] = Axis(figure2, xlabel="Time [days]", ylabel="Number of people")

time_stamps = (0:T) ./ 24
for simul_case in 1:n_simulation
    lines!(ax, time_stamps, n_susceptible[:, simul_case], color=(:skyblue, 0.5))
    lines!(ax, time_stamps, n_exposed[:, simul_case], color=(:grey, 0.5))
    lines!(ax, time_stamps, n_infectious[:, simul_case], color=(:pink, 0.5))
    lines!(ax, time_stamps, n_recovered[:, simul_case], color=(:lightgreen, 0.5))
end
lS = lines!(ax, time_stamps, mean_susceptible, color=:blue)
lE = lines!(ax, time_stamps, mean_exposed, color=:black)
lI = lines!(ax, time_stamps, mean_infectious, color=:red)
lR = lines!(ax, time_stamps, mean_recovered, color=:green)

figure2[1, 2] = Legend(figure2, [lS, lE, lI, lR], ["Susceptible", "Exposed", "Infectious", "Recovered"])

CSV.write("susceptibles.csv", Tables.table(n_susceptible), writeheader=false)
CSV.write("exposed.csv", Tables.table(n_exposed), writeheader=false)
CSV.write("infectious.csv", Tables.table(n_infectious), writeheader=false)
CSV.write("recovered.csv", Tables.table(n_recovered), writeheader=false)
