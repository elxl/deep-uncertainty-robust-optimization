"""
Ride-hailing Simulation Framework in Julia

@author: Xiaotong Guo
"""

using CSV, DataFrames, JuMP, Gurobi
using NPZ, Statistics, LinearAlgebra
using ProgressBars, Random
using JuMPeR, Distributions, Printf, StatsBase, BSON
using Dates, JSON

include("structures.jl")
include("parameters.jl")
include("functions.jl")

fleet_size = 2000

#matching_engine_list = ["historical", "true_demand", "graph_lstm", "single_station_lstm", "all_station_lstm", "SAA", "KNN5", "KNN10", "ORT"]
matching_engine = "graph_lstm_interval"
output_path = "output/graph_lstm_poisson_0627_95/"
ρ_list = [3]
Γ_list = [0, 5, 10]

for ρ in ρ_list, Γ in Γ_list
    
    if isfile(output_path*string(ρ)*"_"*string(Γ)*"_"*string(start_time[1])*"_"*string(end_time[1])*"_results.json")
        continue
    end

    # Initialize demand and vehicle objects
    demand_list, demand_id_dict = initialize_demand(demand_data, zone_to_road_node_dict, zone_index_id_dict)
    vehicle_list, vehicle_id_dict = initialize_vehicle(fleet_size, n, zone_to_road_node_dict)

    simulation_start_time = Time(DateTime(2021,8,13,start_time[1],0,0))
    simulation_end_time = Time(DateTime(2021,8,13,end_time[1],0,0))
    simulation_time = simulation_start_time

    while true
        if simulation_time >= simulation_end_time
            break
        end

        time_index = convert(Int, Second(simulation_time - simulation_start_time) / Second(time_interval_length)) + 1
        number_of_intervals = Int(rebalancing_time_length / time_interval_length) # number of intervals in one optimization step
        end_time_index = time_index + min(size(d[:, :, time_index:end])[3], number_of_intervals) - 1 # end time index for current optimization step

        P_matrix = P[:,:,time_index:end_time_index] # Probility of vehicle staying occupied
        Q_matrix = Q[:,:,time_index:end_time_index] # Probability of occupied vehicle becomes vacant

        # Find initial occupied & vacant vehicle distributions
        V_init = zeros(n) # vacant vehicles
        O_init = zeros(n) # occupied vehicles
        zone_vacant_veh_dict = Dict(i => [] for i in 1:n)
        for veh in vehicle_list
            veh_loc = veh.current_location
            vehicle_zone = road_node_to_zone_dict[veh_loc]
            if veh.occupied
                global O_init[vehicle_zone+1] += 1 # zone index from 1 to 63
            else
                global V_init[vehicle_zone+1] += 1
                push!(zone_vacant_veh_dict[vehicle_zone+1], veh.id)
            end
        end

        print("Rebalancing Phase ")
        println(simulation_time)

        K_sub = end_time_index - time_index + 1
        a_sub = a[:,:,time_index:end_time_index] # if traveling time is bigger than rebalancing threshold
        b_sub = b[:,:,time_index:end_time_index] # if traveling time is bigger than maximum waiting time
        d_sub = d[:,:,time_index:end_time_index] # zone centroids distance 

        if matching_engine == "graph_lstm"
            μ = graph_lstm_mean[:, time_index:end_time_index] # predicted mean
            σ = graph_lstm_var[:, time_index:end_time_index] # predicted standard deviation
            rebalancing_decision = robust_model_function(μ, σ, ρ, Γ, V_init, O_init, P_matrix, Q_matrix, d_sub, a_sub, b_sub, β, γ)
        elseif matching_engine == "graph_lstm_interval"
            μ = graph_lstm_mean[:, time_index:end_time_index] # predicted mean
            lb = graph_lstm_lb[:, time_index:end_time_index]
            ub = graph_lstm_ub[:, time_index:end_time_index]
            rebalancing_decision = robust_model_function_interval(μ, lb, ub, Γ, V_init, O_init, P_matrix, Q_matrix, d_sub, a_sub, b_sub, β, γ)
        elseif matching_engine == "historical"
            μ = demand_mean[:, time_index:end_time_index]
            σ = demand_std[:, time_index:end_time_index]
            rebalancing_decision = robust_model_function(μ, σ, ρ, Γ, V_init, O_init, P_matrix, Q_matrix, d_sub, a_sub, b_sub, β, γ)
        elseif matching_engine == "historical_interval"
            μ = demand_mean[:, time_index:end_time_index] # predicted mean
            lb = demand_lb[:, time_index:end_time_index]
            ub = demand_ub[:, time_index:end_time_index]
            rebalancing_decision = robust_model_function_interval(μ, lb, ub, Γ, V_init, O_init, P_matrix, Q_matrix, d_sub, a_sub, b_sub, β, γ)            
        elseif matching_engine == "true_demand"
            r = true_demand[:, time_index:end_time_index]
            rebalancing_decision = optimization(r, V_init, O_init, P_matrix, Q_matrix, n, K_sub, a_sub, b_sub, d_sub, β, γ)
        end

        rebalancing_decision = (Int.(floor.(rebalancing_decision[:,:,1])))

        # Rebalancing vacant vehicles
        for i in 1:n, j in 1:n
            rebalancing_veh_number = rebalancing_decision[i,j]
            if rebalancing_veh_number <= 0
                continue
            end
            #Random.seed!(2020)
            rebalancing_veh_list = sample(zone_vacant_veh_dict[i], rebalancing_veh_number, replace=false)
            for veh_id in rebalancing_veh_list
                veh = vehicle_id_dict[veh_id]
                global random_number = 0
                Flag = false
                #Random.seed!(2020)
                while true
                    global dest_node = sample(zone_to_road_node_dict[j-1], 1)[1]
                    global rebalancing_dist = road_distance_matrix[veh.current_location+1, dest_node+1]
                    global rebalancing_time = (rebalancing_dist / average_speed) * 3600
                    if rebalancing_time <= time_interval_length
                        break
                    end
                    random_number += 1
                    # Sample 10 times to get points in the two zones between which the traveling time is less than the maximum
                    if random_number >= 10
                        Flag = true
                        break
                    end
                end
                if Flag
                    continue
                end
                # Update Vehicle Objects
                push!(veh.rebalancing_travel_distance, rebalancing_dist)
                push!(veh.rebalancing_trips, 1)
                veh.location = dest_node
                veh.current_location = dest_node
                veh.available_time = simulation_time + Second(Int(floor(rebalancing_time)))
            end
        end

        # update current location for vehicles arrival at their destinations during the current time interval
        for veh in vehicle_list
            if simulation_time <= veh.available_time < simulation_time + Second(time_interval_length)
                veh.current_location = veh.location
            end
        end

        # Matching engine in the simulation
        global matching_simulation_time = simulation_time
        while true
            print("Matching phase ")
            println(matching_simulation_time)
            if matching_simulation_time >= simulation_time + Second(time_interval_length)
                break
            end

            available_vehicles = []
            for veh in vehicle_list
                if veh.available_time < matching_simulation_time + Second(matching_window)
                    push!(available_vehicles, veh)
                end
            end

            requesting_demands = []
            for dem in demand_list
                if simulation_start_time <= dem.request_time < matching_simulation_time + Second(matching_window)
                    if ismissing(dem.assign_time)
                        if !dem.leave_system
                            push!(requesting_demands, dem)
                        end
                    end
                end
            end

            matching_list, unserved_pax_list = matching(available_vehicles, requesting_demands)

            # Update Passengers not matched
            for pax_id in unserved_pax_list
                pax = demand_id_dict[pax_id]
                pax.wait_time += matching_window
                if pax.wait_time >= maximum_waiting_time
                    pax.leave_system = true
                end
            end

            for ((veh_id, pax_id), pickup_dist) in matching_list
                pax = demand_id_dict[pax_id]
                pickup_time = pickup_dist / average_speed * 3600
                pax.wait_time = pickup_time + matching_window + 3600*hour(matching_simulation_time) + 60*minute(matching_simulation_time) +
                                second(matching_simulation_time) - (3600*hour(pax.request_time) + 60*minute(pax.request_time) + second(pax.request_time))
                pax.travel_time = road_distance_matrix[pax.origin+1, pax.destination+1] / average_speed * 3600
                pax.assign_time = matching_simulation_time + Second(matching_window)
                pax.arrival_time = pax.assign_time + Second(floor(pickup_time)) + Second(floor(pax.travel_time))

                veh = vehicle_id_dict[veh_id]
                veh.location = pax.destination
                veh.available_time = pax.arrival_time
                push!(veh.served_passenger, pax.id)
                push!(veh.pickup_travel_distance, pickup_dist)
                push!(veh.occupied_travel_distance, road_distance_matrix[pax.origin+1, pax.destination+1])
            end

            matching_simulation_time += Second(matching_window)
        end

        # Update vehicle status for next rebalancing time window (availability and position when next time window starts)
        matching_time = matching_simulation_time
        for veh in vehicle_list
            if matching_time <= veh.available_time
                veh.occupied = true
                veh.current_location = get_current_location(matching_time, veh, demand_id_dict)
            else
                veh.occupied = false
                veh.current_location = veh.location
            end
        end

        simulation_time += Second(time_interval_length)
    end

    println("Simulation Ends")
    output = Dict()
    # Output simulation results
    vehicle_served_passenger_list = []
    vehicle_occupied_dist_list = []
    vehicle_pickup_dist_list = []
    vehicle_rebalancing_dist_list = []
    vehicle_rebalancing_trip_list = []
    for veh in vehicle_list
        push!(vehicle_served_passenger_list, veh.served_passenger)
        push!(vehicle_occupied_dist_list, veh.occupied_travel_distance)
        push!(vehicle_pickup_dist_list, veh.pickup_travel_distance)
        push!(vehicle_rebalancing_dist_list, veh.rebalancing_travel_distance)
        push!(vehicle_rebalancing_trip_list, veh.rebalancing_trips)
    end

    output["vehicle_served_passenger"] = vehicle_served_passenger_list
    output["vehicle_occupied_dist"] = vehicle_occupied_dist_list
    output["vehicle_pickup_dist"] = vehicle_pickup_dist_list
    output["vehicle_rebalancing_dist"] = vehicle_rebalancing_dist_list
    output["vehicle_rebalancing_trip"] = vehicle_rebalancing_trip_list

    pax_wait_time_list = []
    pax_travel_time_list = []
    pax_leave_list = []
    pax_request_time_list = []
    pax_leave_number = 0
    total_pax_number = 0
    for pax in demand_list
        total_pax_number += 1
        push!(pax_request_time_list, pax.request_time)
        if pax.wait_time > 0 && !pax.leave_system
            push!(pax_wait_time_list, pax.wait_time)
        end
        if pax.travel_time > 0 && !pax.leave_system
            push!(pax_travel_time_list, pax.travel_time)
        end
        if pax.leave_system
            push!(pax_leave_list, 1)
            pax_leave_number += 1
        end
    end

    output["pax_wait_time"] = pax_wait_time_list
    output["pax_travel_time"] = pax_travel_time_list
    output["pax_leaving"] = pax_leave_list
    output["pax_leaving_rate"] = [pax_leave_number / total_pax_number]
    output["pax_request_time"] = pax_request_time_list

    println(pax_leave_number / total_pax_number)

    if matching_engine=="true_demand"
        open(output_path*matching_engine*"_"*string(start_time[1])*"_"*string(end_time[1])*"_results.json","w") do f
            JSON.print(f, output)
        end
        break
    else
        open(output_path*string(ρ)*"_"*string(Γ)*"_"*string(start_time[1])*"_"*string(end_time[1])*"_results.json","w") do f
            JSON.print(f, output)
        end
    end
end
