"""
Functions for Ride-hailing Simulation Framework in Julia

@author: Xiaotong Guo
"""

function initialize_demand(demand_data, zone_to_road_node_dict, zone_index_id_dict)
    """
    Generate passengers according to real data.

    zone_to_road_node_dict: included nodes of each zone
    zone_index_id_dict: zone id (4-263) to zone index (0-62)

    return: list of passengers, dictionary of passengers
    """
    demand_list = []
    demand_id_dict = Dict()
    demand_ind = 1

    Random.seed!(2022)

    for i in 1:size(demand_data)[1]
        request_time = Time(DateTime(demand_data[i,"pu_time"], dateformat"Y-m-d H:M:S"))
        if request_time >= start_timestamp && request_time < end_timestap
            pickup_zone = zone_index_id_dict[Int(demand_data[i, "pu_zone"])]
            dropoff_zone = zone_index_id_dict[Int(demand_data[i, "do_zone"])]
            while true
                global pickup_node = sample(zone_to_road_node_dict[pickup_zone], 1)[1]
                global dropoff_node = sample(zone_to_road_node_dict[dropoff_zone], 1)[1]
                if pickup_node != dropoff_node
                    break
                end
            end
            pax = Passenger(demand_ind, pickup_node, dropoff_node, request_time, missing, 0.0, 0.0, missing, false)
            demand_id_dict[demand_ind] = pax
            push!(demand_list, pax)
            demand_ind += 1
        end
    end

    return demand_list, demand_id_dict
end


function initialize_vehicle(fleet_size, n, zone_to_road_node_dict)
    """
    Initialize vechiles.

    return: list of vehicles, dictionary of vehicles
    """
    vehicle_list = []
    vehicle_id_dict = Dict()
    vehicle_ind = 1
    init_avail_time = Time(DateTime(2021,8,13,start_time[1],0,0))
    Random.seed!(2022)

    zone_vehicle_number = Int(floor(fleet_size / n)) # number of vehicles in each zone
    for i in 1:n
        road_node_list = zone_to_road_node_dict[i-1]
        vehicle_loc_list = sample(road_node_list, zone_vehicle_number, replace=true) # sample number of vehicles locations in zone
        for loc in vehicle_loc_list
            veh = Vehicle(vehicle_ind, loc, init_avail_time, loc, false, [], [], [], [], [])
            vehicle_id_dict[vehicle_ind] = veh
            push!(vehicle_list, veh)
            vehicle_ind += 1
        end
    end

    return vehicle_list, vehicle_id_dict
end


function optimization(r, V1, O1, P, Q, n, K, a, b, d, β, γ)
    nom_model = Model(solver=GurobiSolver(OutputFlag = 0))

    # Declare decision variables x, y, O, V, S, T, z
    @variable(nom_model, x[i=1:n, j=1:n, k=1:K] >= 0)
    @variable(nom_model, y[i=1:n, j=1:n, k=1:K] >= 0)
    @variable(nom_model, O[i=1:n, k=1:K] >= 0)
    @variable(nom_model, V[i=1:n, k=1:K] >= 0)
    @variable(nom_model, S[i=1:n, k=1:K] >= 0)
    @variable(nom_model, T[i=1:n, k=1:K] >= 0)

    # Set constraints to force initial position of occupied and vacant vehicles
    @constraint(nom_model, V[1:n, 1] .== V1)
    @constraint(nom_model, O[1:n, 1] .== O1)

    # Set constraints related to state transitions (1,2,3)
    @constraint(nom_model, [i=1:n, k=1:K], S[i,k] == V[i,k] + sum(x[j,i,k] for j in 1:n) - sum(x[i,j,k] for j in 1:n))
    @constraint(nom_model, [i=1:n, k=1:K-1], V[i,k+1] == S[i,k] - sum(y[j,i,k] for j in 1:n) + sum(Q[j,i,k] * O[j,k] for j in 1:n) )
    @constraint(nom_model, [i=1:n, k=1:K-1], O[i,k+1] == sum(y[j,i,k] for j in 1:n) + sum(P[j,i,k] * O[j,k] for j in 1:n))

    @constraint(nom_model, [i=1:n, k=1:K], sum(x[i,j,k] for j in 1:n) <= V[i,k])

    # Set rebalancing constraint (4)
    @constraint(nom_model, [i=1:n, j=1:n, k=1:K], a[i,j,k] * x[i,j,k] == 0)

    # Set surplus vehicle / passenger constraint (6, 7, 8, 9)
    @constraint(nom_model, [i=1:n, k=1:K], sum(y[j,i,k] for j in 1:n) <= S[i,k])
    @constraint(nom_model, [i=1:n, k=1:K], sum(y[i,j,k] for j in 1:n) <= r[i,k])
    @constraint(nom_model, [i=1:n, k=1:K], T[i,k] == r[i,k] - sum(y[i,j,k] for j in 1:n))

    # Set matching constraint (10)
    @constraint(nom_model, [i=1:n, j=1:n, k=1:K], b[i,j,k] * y[i,j,k] == 0)

    # Set objective
    @objective(nom_model, Min, sum(x[i,j,k] * d[i,j,k] for i in 1:n, j in 1:n, k in 1:K)
                            + β * sum(y[i,j,k] * d[j,i,k] for i in 1:n, j in 1:n, k in 1:K)
                            + γ * sum(T[i,k] for i in 1:n, k in 1:K))

    solve(nom_model)

    x_nominal = getvalue(x)
    y_nominal = getvalue(y)
    S_nominal = getvalue(S)
    T_nominal = getvalue(T)
    V_nominal = getvalue(V)
    O_nominal = getvalue(O);

    return x_nominal
end


function matching(vehicles, demands)
    """
    Match available vechiles and requests.

    vehicles: list
    demands: list

    return: matched vehicle and passenger, unserved passenger
    """
    pickup_dist = Dict()
    # All possible matching schemes
    for veh in vehicles, dem in demands
        veh_id = veh.id
        veh_loc = veh.current_location
        dem_id = dem.id
        dem_loc = dem.origin
        pick_distance = road_distance_matrix[veh_loc+1, dem_loc+1]
        pickup_time = pick_distance / average_speed
        if pickup_time * 3600 <= maximum_waiting_time
            pickup_dist[(veh_id, dem_id)] = pick_distance
        end
    end

    veh_dict = Dict(veh.id => [] for veh in vehicles)
    dem_dict = Dict(dem.id => [] for dem in demands)

    for i in keys(pickup_dist)
        push!(veh_dict[i[1]], i)
        push!(dem_dict[i[2]], i)
    end

    # Matching optimization
    model = Model(solver=GurobiSolver(OutputFlag = 0, TimeLimit = 10))
    @variable(model, x[i in keys(pickup_dist)], Bin)
    @variable(model, y[d in keys(dem_dict)], Bin)
    @constraint(model, [veh in keys(veh_dict)], sum(x[i] for i in veh_dict[veh]) <= 1)
    @constraint(model, [dem in keys(dem_dict)], sum(x[i] for i in dem_dict[dem]) + y[dem] == 1)
    @objective(model, Min, γ * sum(y[i] for i in keys(dem_dict))
                            + sum(x[i] * pickup_dist[i] for i in keys(pickup_dist)))
    solve(model)

    x_opt = getvalue(x)
    y_opt = getvalue(y)

    # Get matched vehicle and passenger as well as the prik up distance
    matching_list = []
    for i in keys(pickup_dist)
        if x_opt[i] == 1
            push!(matching_list, [i, pickup_dist[i]])
        end
    end

    # Get unserved passenger
    unserved_pax_list = []
    for d in keys(dem_dict)
        if y_opt[d] == 1
            push!(unserved_pax_list, d)
        end
    end

    return matching_list, unserved_pax_list
end


function get_current_location(matching_time, veh, demand_id_dict)
    """
    demand_id_dict: id:passenger
    """
    pax = demand_id_dict[last(veh.served_passenger)]

    if matching_time - Second(matching_window) < pax.assign_time <= matching_time
        return veh.current_location
    else
        vehicle_travel_time = 3600*hour(matching_time) + 60*minute(matching_time) + second(matching_time) - 
                            (3600*hour(pax.request_time) + 60*minute(pax.request_time) + second(pax.request_time)) - pax.wait_time
        veh_start_loc = pax.origin
        veh_end_loc = pax.destination
        trip_path = []
        global temp_node = veh_end_loc
        # Path of vehicle
        while true
            pred = Int(predecessor[veh_start_loc+1,temp_node+1])
            pushfirst!(trip_path, pred)
            if pred == veh_start_loc
                break
            end
            temp_node = pred
        end
        push!(trip_path, veh_end_loc)
    end

    travel_time = 0
    for i in 1:size(trip_path)[1]-1
        start_node = trip_path[i]
        end_node = trip_path[i+1]
        segment_dist = road_distance_matrix[start_node+1,end_node+1]
        segment_time = segment_dist / average_speed * 3600
        travel_time += segment_time
        if travel_time >= vehicle_travel_time
            return end_node
        end
    end
end


function robust_model_function(μ, σ, 𝜌, Γ, V1, O1, P, Q, d, a, b, 𝛽, γ)

    n = size(d,2)
    K = size(μ,2)

    robust_model = RobustModel(solver=GurobiSolver(OutputFlag = 0))

    # Declare decision variables x, y, O, V, S, M, N, T, z
    @variable(robust_model, x[i=1:n, j=1:n, k=1:K] >= 0)
    @variable(robust_model, y[i=1:n, j=1:n, k=1:K] >= 0)
    @variable(robust_model, O[i=1:n, k=1:K] >= 0)
    @variable(robust_model, V[i=1:n, k=1:K] >= 0)
    @variable(robust_model, S[i=1:n, k=1:K] >= 0)

    @variable(robust_model, 𝜔)

    # Set constraints to force initial position of occupied and vacant vehicles
    @constraint(robust_model, V[1:n, 1] .== V1)
    @constraint(robust_model, O[1:n, 1] .== O1)

    # Uncertainty
    @uncertain(robust_model, 𝜁[i=1:n, k=1:K])
    @constraint(robust_model, [i=1:n, k=1:K], 𝜁[i,k] <= 𝜌)
    @constraint(robust_model, [i=1:n, k=1:K], - 𝜁[i,k] <= 𝜌)
    @constraint(robust_model, [k=1:K], sum(σ[i,k] * 𝜁[i,k] for i in 1:n) <= Γ)
    @constraint(robust_model, [k=1:K], -sum(σ[i,k] * 𝜁[i,k] for i in 1:n) <= Γ)

    @constraint(robust_model, [i=1:n, k=1:K], μ[i,k] + 𝜁[i,k] * σ[i,k] >= 0)

    # Set constraints related to state transitions (1,2,3)
    @constraint(robust_model, [i=1:n, k=1:K], S[i,k] == V[i,k] + sum(x[j,i,k] for j in 1:n) - sum(x[i,j,k] for j in 1:n))
    @constraint(robust_model, [i=1:n, k=1:K-1], V[i,k+1] == S[i,k] - sum(y[j,i,k] for j in 1:n) + sum(Q[j,i,k] * O[j,k] for j in 1:n))
    @constraint(robust_model, [i=1:n, k=1:K-1], O[i,k+1] == sum(y[j,i,k] for j in 1:n) + sum(P[j,i,k] * O[j,k] for j in 1:n))

    @constraint(robust_model, [i=1:n, k=1:K], sum(x[i,j,k] for j in 1:n) <= V[i,k])

    # Set rebalancing constraint (4)
    @constraint(robust_model, [i=1:n, j=1:n, k=1:K], a[i,j,k] * x[i,j,k] == 0)

    # Set surplus vehicle / passenger constraint (6, 7, 8, 9)
    @constraint(robust_model, [i=1:n, k=1:K], sum(y[j,i,k] for j in 1:n) <= S[i,k])
    @constraint(robust_model, [i=1:n, k=1:K], sum(y[i,j,k] for j in 1:n) <= μ[i,k] + 𝜁[i,k] * σ[i,k])

    # Set matching constraint (10)
    @constraint(robust_model, [i=1:n, j=1:n, k=1:K], b[i,j,k] * y[i,j,k] == 0)

    @constraint(robust_model, sum(x[i,j,k] * d[i,j,k] for i in 1:n, j in 1:n, k in 1:K)
                            + 𝛽 * sum(y[i,j,k] * d[j,i,k] for i in 1:n, j in 1:n, k in 1:K)
                            + γ * sum((μ[i,k] + 𝜁[i,k] * σ[i,k] - sum(y[i,j,k] for j in 1:n)) for i in 1:n, k in 1:K) <= 𝜔)

    # Set objective
    @objective(robust_model, Min, 𝜔)

    solve(robust_model)

    x_robust = getvalue(x)

    return x_robust
end


function robust_model_function_interval(μ, lb, ub, Γ, V1, O1, P, Q, d, a, b, 𝛽, γ)

    n = size(d,2)
    K = size(μ,2)

    robust_model = RobustModel(solver=GurobiSolver(OutputFlag = 0))

    # Declare decision variables x, y, O, V, S, M, N, T, z
    @variable(robust_model, x[i=1:n, j=1:n, k=1:K] >= 0)
    @variable(robust_model, y[i=1:n, j=1:n, k=1:K] >= 0)
    @variable(robust_model, O[i=1:n, k=1:K] >= 0)
    @variable(robust_model, V[i=1:n, k=1:K] >= 0)
    @variable(robust_model, S[i=1:n, k=1:K] >= 0)

    @variable(robust_model, 𝜔)

    # Set constraints to force initial position of occupied and vacant vehicles
    @constraint(robust_model, V[1:n, 1] .== V1)
    @constraint(robust_model, O[1:n, 1] .== O1)

    # Uncertainty
    @uncertain(robust_model, r[i=1:n, k=1:K])
    @constraint(robust_model, [i=1:n, k=1:K], r[i, k] <= ub[i, k])
    @constraint(robust_model, [i=1:n, k=1:K], - r[i, k] <= - lb[i, k])
    @constraint(robust_model, [k=1:K], sum(r[i, k] - μ[i, k] for i in 1:n) <= Γ)
    @constraint(robust_model, [k=1:K], -sum(r[i, k] - μ[i, k] for i in 1:n) <= Γ)   


    @constraint(robust_model, [i=1:n, k=1:K], r[i, k] >= 0)

    # Set constraints related to state transitions (1,2,3)
    @constraint(robust_model, [i=1:n, k=1:K], S[i,k] == V[i,k] + sum(x[j,i,k] for j in 1:n) - sum(x[i,j,k] for j in 1:n))
    @constraint(robust_model, [i=1:n, k=1:K-1], V[i,k+1] == S[i,k] - sum(y[j,i,k] for j in 1:n) + sum(Q[j,i,k] * O[j,k] for j in 1:n))
    @constraint(robust_model, [i=1:n, k=1:K-1], O[i,k+1] == sum(y[j,i,k] for j in 1:n) + sum(P[j,i,k] * O[j,k] for j in 1:n))

    @constraint(robust_model, [i=1:n, k=1:K], sum(x[i,j,k] for j in 1:n) <= V[i,k])

    # Set rebalancing constraint (4)
    @constraint(robust_model, [i=1:n, j=1:n, k=1:K], a[i,j,k] * x[i,j,k] == 0)

    # Set surplus vehicle / passenger constraint (6, 7, 8, 9)
    @constraint(robust_model, [i=1:n, k=1:K], sum(y[j,i,k] for j in 1:n) <= S[i,k])
    @constraint(robust_model, [i=1:n, k=1:K], sum(y[i,j,k] for j in 1:n) <= r[i, k])

    # Set matching constraint (10)
    @constraint(robust_model, [i=1:n, j=1:n, k=1:K], b[i,j,k] * y[i,j,k] == 0)

    @constraint(robust_model, sum(x[i,j,k] * d[i,j,k] for i in 1:n, j in 1:n, k in 1:K)
                            + 𝛽 * sum(y[i,j,k] * d[j,i,k] for i in 1:n, j in 1:n, k in 1:K)
                            + γ * sum((r[i, k] - sum(y[i,j,k] for j in 1:n)) for i in 1:n, k in 1:K) <= 𝜔)

    # Set objective
    @objective(robust_model, Min, 𝜔)

    solve(robust_model)

    x_robust = getvalue(x)

    return x_robust
end