"""
Parameters and Data Loading

@author: Xiaotong Guo
"""

nodes = [12,88,13,261,87,209,231,45, 125,211,144,148,232,158,249,113,114,79,4,246,68,90,234,107,224,137,186,164,170,100,233]
zone_index_id = convert(Matrix,CSV.File("data/NYC/zone_index_id.csv", header=false) |> DataFrame)
zone_index_id_dict = Dict()
for i in 1:size(zone_index_id)[1]
    zone_index_id_dict[zone_index_id[i,2]] = zone_index_id[i,1]
end
nodes_index = sort([zone_index_id_dict[i] for i in nodes])
zones_index = nodes_index

start_time = (7, 0) # Hour, Minute
end_time = (9, 0) # Hour, Minute

time_interval_length = 300 # Seconds
rebalancing_time_length = 900 # 6 time intervals
matching_window = 30 # Seconds
road_distance_matrix = convert(Matrix, CSV.File("data/NYC/road_network/distance.csv", header=false) |> DataFrame) ./ 1609.34
predecessor = convert(Matrix, CSV.File("data/NYC/road_network/predecessor.csv", header=false) |> DataFrame)

start_bin = start_time[1] * 12
end_bin = end_time[1] * 12 - 1
start_timestamp = Time(DateTime(2019,6,27,start_time[1],0,0))
end_timestap = Time(DateTime(2019,6,27,end_time[1],0,0))

# Demand data
data = CSV.read("data/NYC/processed_data/normalized_data.csv", DataFrame)
# data = data[(data[!,:bin] .>= start_bin) .& (data[!,:bin] .<= end_bin) .& in.(data[!,:zone], nodes), :]
data = filter(row -> row.bin >= start_bin && row.bin <= end_bin && in(row.zone, nodes), data)
prev_data = data[(data.month .!= 6) .| (data.day .< 27), :]
June_27_data = data[(data.month .== 6) .& (data.day .== 27), :]

gd = groupby(prev_data, [:day, :month])
n = size(unique(data.zone))[1]
K = size(unique(data.bin))[1]
m = length(gd)
data_points = zeros((n, K, m))
index = 1
for df in gd
   global index
   y_i = unstack(df[:,[:zone,:bin,:demand]], :bin, :demand)[:, 2:end]
   data_points[:,:,index] = Matrix(y_i)
   index += 1
end

road_node_to_zone = convert(Matrix,CSV.File("data/NYC/road_node_to_zone.csv", header=false) |> DataFrame)
road_node_to_zone = [row for row in eachrow(road_node_to_zone) if row[2] in nodes_index]
road_node_to_zone = transpose(hcat(road_node_to_zone...))
# mask = in.(road_node_to_zone[:, 2], nodes_index)
# road_node_to_zone = road_node_to_zone[mask, :]
road_node = road_node_to_zone[:, 1]
zone_to_road_node_dict = Dict(s => [] for s in zones_index)
road_node_to_zone_dict = Dict()
for i in 1:size(road_node_to_zone)[1]
    road_node_id = road_node_to_zone[i, 1]
    zone_id = road_node_to_zone[i, 2]
    road_node_to_zone_dict[road_node_id] = zone_id
    append!(zone_to_road_node_dict[zone_id], road_node_id)
end

zone_centriod_node = convert(Matrix,CSV.File("data/NYC/centroid_ind_node.csv", header=false) |> DataFrame)
zone_centriod_node = [row for row in eachrow(zone_centriod_node) if row[1] in nodes_index]
zone_centriod_node = transpose(hcat(zone_centriod_node...))
# mask = in.(zone_centriod_node[:, 1], nodes_index)
# zone_centriod_node = zone_centriod_node[mask, :]
centroid_to_node_dict = Dict()
for i in 1:size(zone_centriod_node)[1]
    centroid_to_node_dict[zone_centriod_node[i,1]] = zone_centriod_node[i,2]
end


# Demand information used for solving optimizaiton problems
# demand_mean = mean(data_points, dims=3)[:,:]
# demand_std = std(data_points, dims=3)[:,:]
matrix_index = [x + 1 for x in zones_index]
demand_mean = npzread("historical/0627_normal_mean.npy")[matrix_index, :]
demand_std = npzread("historical/0627_normal_std.npy")[matrix_index, :]
demand_lb = npzread("historical/0627_normal_75_lb.npy")[matrix_index, :]
demand_ub = npzread("historical/0627_normal_75_ub.npy")[matrix_index, :]
true_demand = unstack(June_27_data[:, [:zone, :bin, :demand]], :bin, :demand)[:, 2:end]
hist_avg_demand = unstack(June_27_data[:, [:zone, :bin, :historical_average]], :bin, :historical_average)[:, 2:end]

# Actual demand data used for simulation
demand_data = CSV.read("data/NYC/demand/fhv_records_06272019.csv", DataFrame)
demand_data = filter(row -> row.pu_zone in nodes && row.do_zone in nodes, demand_data)
# demand_data = demand_data[(in.(demand_data.pu_zone, nodes) .& in.(demand_data.do_zone, nodes)), :]

# Problem Parameters
β = 1
γ = 1e2
average_speed = 20
maximum_waiting_time = 300    # seconds
maximum_rebalancing_time = time_interval_length
big_M = 1e5

d = npzread("data/NYC/distance_matrix.npy")[matrix_index,:][:,matrix_index] # Zone centroid distances in miles
d = repeat(convert(Matrix, d), inner = [1,1,K]) # Repeat d to create a n x n x K matrix
# Hourly travel time to 288 time intervals
w_hourly = npzread("data/NYC/hourly_tt.npy")[matrix_index,:, :][:,matrix_index, :]
a = repeat(convert(Matrix, w_hourly[:,:,1]), inner = [1,1,12]);
for i in 2:24
    b = repeat(convert(Matrix, w_hourly[:,:,i]), inner = [1,1,12])
    global a = cat(dims=3, a, b)
end
w = a .* 3600;
w = w[:,:,start_bin+1:end_bin+1] # 7 AM to 9 AM travel time matrix

a = [w[i,j,k] > maximum_rebalancing_time for i in 1:n, j in 1:n, k in 1:K]
b = [w[i,j,k] > maximum_waiting_time for i in 1:n, j in 1:n, k in 1:K];

P = npzread("data/NYC/p_matrix_occupied.npy")[matrix_index,:][:,matrix_index]
Q = npzread("data/NYC/q_matrix_occupied.npy")[matrix_index,:][:,matrix_index]
P = repeat(convert(Matrix, P), inner = [1,1,K])
Q = repeat(convert(Matrix, Q), inner = [1,1,K])

# Demand mean and variance from neural network
graph_lstm_mean = CSV.read("graph_lstm/0627_poisson_mean.csv", DataFrame)
graph_lstm_mean = filter(row -> row.zone in nodes, graph_lstm_mean)
graph_lstm_mean = unstack(graph_lstm_mean[:, [:zone, :bin, :demand]], :bin, :demand)
graph_lstm_mean = graph_lstm_mean[:,start_bin+2:end_bin+2]
graph_lstm_var = CSV.read("graph_lstm/0627_poisson_std.csv", DataFrame)
graph_lstm_var = filter(row -> row.zone in nodes, graph_lstm_var)
graph_lstm_var = unstack(graph_lstm_var[:, [:zone, :bin, :demand]], :bin, :demand)
graph_lstm_var = graph_lstm_var[:,start_bin+2:end_bin+2]

# Lower and upper bound from neural network
graph_lstm_lb = CSV.read("graph_lstm/0627_poisson_75_lb.csv", DataFrame)
graph_lstm_lb = filter(row -> row.zone in nodes, graph_lstm_lb)
graph_lstm_lb = unstack(graph_lstm_lb[:, [:zone, :bin, :demand]], :bin, :demand)
graph_lstm_lb = graph_lstm_lb[:,start_bin+2:end_bin+2]
graph_lstm_ub = CSV.read("graph_lstm/0627_poisson_75_ub.csv", DataFrame)
graph_lstm_ub = filter(row -> row.zone in nodes, graph_lstm_ub)
graph_lstm_ub = unstack(graph_lstm_ub[:, [:zone, :bin, :demand]], :bin, :demand)
graph_lstm_ub = graph_lstm_ub[:,start_bin+2:end_bin+2]


# Bounds from neural network for sum of demand
graph_lstm_sum_lb = npzread("graph_lstm/0627_nb_sum_95_lb.npy")
graph_lstm_sum_lb = graph_lstm_sum_lb[start_bin+1:end_bin+1]
graph_lstm_sum_ub = npzread("graph_lstm/0627_nb_sum_95_ub.npy")
graph_lstm_sum_ub = graph_lstm_sum_ub[start_bin+1:end_bin+1]
;
