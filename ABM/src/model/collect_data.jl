using Agents

total_infected(m) = count(a.status == :I for a ∈ allagents(m))
susceptible(m) = count(a == :S for a ∈ m)
exposed(m) = count(a == :E for a ∈ m)
infected(m) = count(a == :I for a ∈ m)
recovered(m) = count(a == :R for a ∈ m)