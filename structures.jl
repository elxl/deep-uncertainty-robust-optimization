"""
Scripts Containing Simulation Objects

@author: Xiaotong Guo
"""

mutable struct Passenger
    id::Int64
    origin::Int64
    destination::Int64
    request_time::Any
    assign_time::Any
    wait_time::Float64
    travel_time::Float64
    arrival_time::Any
    leave_system::Bool
end

mutable struct Vehicle
    id::Int64
    location::Int64
    available_time::Time
    current_location::Int64
    occupied::Bool
    served_passenger::Vector{Any}
    rebalancing_travel_distance::Vector{Any}
    pickup_travel_distance::Vector{Any}
    occupied_travel_distance::Vector{Any}
    rebalancing_trips::Vector{Any}
end
