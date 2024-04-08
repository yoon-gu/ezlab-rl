include("make_abm.jl")

function add_n_nurses_on_floor_whos_status!(;
    model,
    number::Int,
    main_floor::Int,
    status::Symbol
)
    main_room = string(main_floor - 6) * "l"

    for i = 1:number
        add_agent_who_at_position!(;
            model=model,
            occupation="N",
            main_room=main_room,
            status=status
        )
    end
end


function add_n_doctors_on_floor_whos_status!(;
    model,
    number::Int,
    main_floor::Int,
    status::Symbol
)
    main_room = string(main_floor - 6) * "n"

    for i = 1:number
        add_agent_who_at_position!(;
            model=model,
            occupation="D",
            main_room=main_room,
            status=status
        )
    end
end


function add_n_cleaners_on_floor_whos_status!(;
    model,
    number::Int,
    main_floor::Int,
    status::Symbol
)
    main_room = string(main_floor - 6) * "s"

    for i = 1:number
        add_agent_who_at_position!(;
            model=model,
            occupation="Wc",
            main_room=main_room,
            status=status
        )
    end
end

function add_n_transfers_on_floor_whos_status!(;
    model,
    number::Int,
    main_floor::Int,
    status::Symbol
)
    main_room = string(main_floor - 6) * "l"

    for i = 1:number
        add_agent_who_at_position!(;
            model=model,
            occupation="Wa",
            main_room=main_room,
            status=status
        )
    end
end

function add_n_physical_therapists_whos_status!(;
    model,
    number::Int,
    status::Symbol
)
    main_room = "3l"

    for i = 1:number
        add_agent_who_at_position!(;
            model=model,
            occupation="Wp",
            main_room=main_room,
            status=status
        )
    end
end

function add_n_operational_therapists_whos_status_at_6th!(;
    model,
    number::Int,
    status::Symbol
)
    main_room = "0m"

    for i = 1:number
        add_agent_who_at_position!(;
            model=model,
            occupation="Wo",
            main_room=main_room,
            status=status
        )
    end
end

function add_n_operational_therapists_whos_status_at_9th!(;
    model,
    number::Int,
    status::Symbol
)
    main_room = "3m"

    for i = 1:number
        add_agent_who_at_position!(;
            model=model,
            occupation="Wo",
            main_room=main_room,
            status=status
        )
    end
end

function add_n_robotic_therapists_whos_status!(;
    model,
    number::Int,
    status::Symbol
)
    main_room = "0n"

    for i = 1:number
        add_agent_who_at_position!(;
            model=model,
            occupation="Wr",
            main_room=main_room,
            status=status
        )
    end
end


function add_n_patients_on_floor_at_room_whos_status!(;
    model,
    number::Int,
    main_floor::Int,
    room_number::Int,
    status::Symbol
)
    if room_number > 21 || room_number < 1
        error("Room number should be positive and less than 22.")
    end
    main_room = string(main_floor - 6) * string(room_number, base=32)

    for i = 1:number
        add_agent_who_at_position!(;
            model=model,
            occupation="P",
            main_room=main_room,
            status=status
        )
    end
end


function add_n_caregivers_on_floor_at_room_whos_status!(;
    model,
    number::Int,
    main_floor::Int,
    room_number::Int,
    status::Symbol
)
    if room_number > 21 || room_number < 1
        error("Room number should be positive and less than 22.")
    end
    main_room = string(main_floor - 6) * string(room_number, base=32)

    for i = 1:number
        add_agent_who_at_position!(;
            model=model,
            occupation="C",
            main_room=main_room,
            status=status
        )
    end
end