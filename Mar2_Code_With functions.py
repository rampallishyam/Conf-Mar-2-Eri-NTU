# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 16:12:23 2020

@author: shyam.rampalli

# =============================================================================
#     Entire code logic in simple words:
# =============================================================================
    It executes first step of K-means algo where the cluster centers are vehicle locations
    and all vehciles are mapped with the stops which are nearest to them.
    The demand values of all stops are then summed up and checked with vehicle capacity
    Then only the stops with which the vehicle can satisfy the demand will it serve are considered
    All the points served are removed from the plot and vehicle capacities are revised
    Then the same procedure is executed till the vehicle capacity is exhausted.
    
    Limitations:
    1. There is not sorting of demand which means that the algorithm picks the points arbitrarily rather than an orderly manner
    2. Time constraint cannot be implemented yet
    
"""
# ==============================================================================================
#                V refers to the nd array created for all the vehicles
#    X refers to the number of stops present which contains their location and demand values
# ==============================================================================================

from __future__ import print_function
import random
import numpy as np
import shapefile #module to operate with shapefiles
import matplotlib.pyplot as plt #matplot library for plotting
import pandas as pd
from sklearn.cluster import KMeans

# =============================================================================
#                             or-tools libraries
# =============================================================================
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

n_stops = 90   #Number of stops
fleet_size = 30 
veh_capacity = 12
max_demand = 5

Objective_function_list = []
for i in range(0,50,1):
    
    random.seed(i) #initializing
    # first column is x-axis, second: y-axis, third: demand, fourth: dummy variable, no reason to assign this variable.
    
    coords_X = []
    attr_X_list = []
    reader= shapefile.Reader(r"C:\Users\shyam.rampalli\OneDrive - Nanyang Technological University\documents\Conf - Mar 2\Data\AV Stops.shp")
    fields = [field[0] for field in reader.fields[1:]]
    for feature in reader.shapeRecords():
        geom = feature.shape.__geo_interface__
        field_names = [field for field in fields] 
        atr = dict(zip(field_names, feature.record))
        attr_X_list.append(atr['name'])
        coords_X.append([geom['coordinates'][1],geom['coordinates'][0],random.randint(0,max_demand), int(0)])
    coords_X = random.sample(coords_X,n_stops)
    X = np.asarray(coords_X)
    
    # first column is x-axis, second: y-axis, third: veh capacity, fourth: capacity fulfilled.
    coords_V = []
    attr_V_list = []
    reader= shapefile.Reader(r"C:\Users\shyam.rampalli\OneDrive - Nanyang Technological University\documents\Conf - Mar 2\Data\Electronic Parking_AMK_AOI.shp")
    fields = [field[0] for field in reader.fields[1:]]
    for feature in reader.shapeRecords():
        geom = feature.shape.__geo_interface__
        field_names = [field for field in fields] 
        atr = dict(zip(field_names, feature.record))
        attr_V_list.append(atr['id'])
        coords_V.append([geom['coordinates'][1],geom['coordinates'][0],random.randint(0,veh_capacity), int(0)])
    coords_V = random.sample(coords_V,fleet_size)
    V = np.asarray(coords_V)
    
    #%%
    def Kmeans(V, X, n_clusters):
        
    # ======================================================================================================================
    #   "This code refers to the Kmeans function which does one iteration of K-means algorithm with fixed cluste centers"     
    # ======================================================================================================================
        """ 
        This functions two arrays and performs one iteration of Kmeans which means it predicts the clusters centers each point belongs to                         
        Cluster centers are the vehcile locations.
        """    
        # Artificially fix your centroids
        km = KMeans( n_clusters = fleet_size)
        km.cluster_centers_ = V[:,:-2]
        # This gives an initial solution using K-means approach
        initial_prediction = km.predict(X[:,:-2])
        return initial_prediction
    
    #%%
    # Scatter plot of stops in the selected area
    plt.scatter(X[:, 0], X[:, 1],s=10)
    # Scatter plot of vehicle locations
    plt.scatter(V[:, 0], V[:, 1],s=40)
    fig = plt.subplots()
    
    #%%
    # ======================================================================================================
    # A function is defined that draws an arrow showing the match between the vehicle and the nearest stop
    # ======================================================================================================
    def drawArrow(x, y):
        plt.arrow(x[0], x[1], y[0] - x[0], y[1] - x[1],
                  head_width=0.0001, width = 0.000034, length_includes_head=False)
        
    #%%
    # ===============================================================================================================
    # This function considers the demand factor and draws arrows and re-adjusts the arrays of vehicles and bus stops
    # ================================================================================================================
    def demand(initial_prediction, V, X):
        index = 0 
        unserved_list = []
        unserved_array = []
        solution_list = []
        for veh in list(initial_prediction):
            if index < len(X):
                if X[index][2]>0:
                    V[veh][3]+=X[index][2] #Adding all the demand values to the vehicle
                    if V[veh][3]<=V[veh][2]:
                        drawArrow(y = V[veh],x = X[index])
                        solution_list.append([list(V[veh]),list(X[index]),attr_V_list[veh],attr_X_list[index]])
                        V[veh][2]-=X[index][2] #Update vehicle capacity for every arraw that has been drawn
                        V[veh][3]-=X[index][2]
                        X[index][2] = 0
                    else:
                        if V[veh][2]>0:
                            X[index][2]-=V[veh][2]
                            drawArrow(y = V[veh],x = X[index])
                            solution_list.append([list(V[veh]),list(X[index]),attr_V_list[veh],attr_X_list[index]])
                            V[veh][2] = 0
                    unserved_list.append(X[index]) #creating a list of stops that are not served in the present iteration
                    unserved_array = np.asarray(unserved_list)
            index+=1
        
        #A loop to find all the vehicles that can still be used    
        V2 = []
        for veh in V:
            if veh[2]>0:
                V2.append(veh)
        V2= np.asarray(V2)
         
        return [V2, fleet_size, unserved_array, solution_list]
    
    #%%
        
    # =========================================================================================================
    # This functions takes in a dataframe and creates a pivot table and finally converts it into list of lists
    # =========================================================================================================
        
    def pivot_to_list(df):
        pivot_table = pd.pivot_table(df, values = 'total_cost', index = 'origin_id', columns = 'destination_id')
        return pivot_table
    
    #%%
    # =============================================================================
    #     This loop iterates over until the full capacity has been utilized
    # =============================================================================
    sol_list = []
    tot_demand = int(sum(X[:,2:3]))
    tot_supply = int(sum(V[:,2:3]))
    print("Total demand: %d" %tot_demand)
    print("Total supply: is %d" %tot_supply)
    for iter in range(0,10,1):
        
        initial_prediction = Kmeans(V=V,X=X, n_clusters=fleet_size)
        # Scatter plot of stops in the selected area
        plt.scatter(X[:, 0], X[:, 1], s=30)
        # Scatter plot of vehicle locations
        plt.scatter(V[:, 0], V[:, 1], s=90)
        #for stop in X:
            #plt.text(stop[0] * (1 + 0.01), stop[1] * (1 + 0.01) ,stop[2], fontsize=0.005)
        #for stop in V:
            #plt.text(stop[0] * (1 + 0.01), stop[1] * (1 + 0.01) ,stop[2], fontsize=0.005)
        Demand_list = demand(initial_prediction=initial_prediction, V=V,X=X)    #Update the vehicles by the non-assigned or still with existing capacity
        X = Demand_list[2]
        fleet_size = Demand_list[1]
        V = Demand_list[0]
        sol_list.append(Demand_list[3])
        if len(V) == 0 or len(X)==0:
            break
        
        else:
        #this for loop is for making the filled capacities variable to zero since capacties are already in the code above
            for val in V:
                val[3] = 0
        #plt.savefig(r"C:\Users\shyam.rampalli\OneDrive - Nanyang Technological University\documents\Conf - Mar 2\Python codes\results\Initial_sol" + str(iter) + ".png")
        plt.rcParams["figure.figsize"] = [10,8]
        plt.show()
    
    print ('Either Vehicle capacity exhausted or all demands are matched')
    
    
    #%%
    
    # =============================================================================
    #           Now this code is to process the result to a dataframe
    # =============================================================================
        
    
    all_sol_list = []  
    for index in range(0,len(sol_list),1):
        all_sol_list+=sol_list[index]
    
    for val in all_sol_list:
        del val[0][2:]
        del val[1][2:]
    #Converted into a dataframe and added all points in the form of a dataframe
    all_sol_df = pd.DataFrame.from_records(all_sol_list)
    all_sol_df.columns = ['Vehicle location','Stop location','origin_id','destination_id']
    all_points_df = pd.read_csv(r"C:\Users\shyam.rampalli\OneDrive - Nanyang Technological University\documents\Conf - Mar 2\Data\OD_Matrix_All_points.csv")
    
    #Change the datatype to string for all the values that have names(for ex: Blk202 will be a string even if it has integers)
    all_sol_df['origin_id'] = all_sol_df['origin_id'].astype(str)
    all_sol_df['destination_id'] = all_sol_df['destination_id'].astype(str)
    all_points_df['origin_id'] = all_points_df['origin_id'].astype(str)
    all_points_df['destination_id'] = all_points_df['destination_id'].astype(str)
         
    final_df = pd.merge(all_sol_df, all_points_df, on=['origin_id', 'destination_id'],how = 'inner')
    #Create a dict where each vehicle contains the points it is sorted to along with the vehicle and last point as MRT
    all_sol_dict = all_sol_df.groupby('origin_id')['destination_id'].apply(lambda grp: list(grp)).to_dict()
    for key in all_sol_dict.keys():
        all_sol_dict[key].insert(0, key)
        
    #%%
    
    # =============================================================================
    #              Generate an OD-MATRIX and apply TSP for it
    # =============================================================================
    VMT_list=[]
    for key in all_sol_dict.keys():
        all_sol_df_temp = all_points_df[all_points_df['origin_id'].isin(all_sol_dict[key])]
        all_sol_df_temp = all_sol_df_temp[all_sol_df_temp['destination_id'].isin(all_sol_dict[key])]
        all_sol_df_temp_pivot = pd.pivot_table(all_sol_df_temp, values = 'total_cost', index= 'origin_id', columns= 'destination_id')
        all_sol_list_temp = []
        for val in list(all_sol_df_temp_pivot):
            all_sol_list_temp.append(list(all_sol_df_temp_pivot[val]))
        
        def create_data_model():
            data = {}
            data['distance_matrix'] = all_sol_list_temp  # yapf: disable
            data['num_vehicles'] = 1
            data['depot'] = 0
            return data
    
    
        def print_solution(manager, routing, assignment):
            """Prints assignment on console."""
            #print('Objective: {} miles'.format(VMT_shuttle))
            index = routing.Start(0)
            plan_output = 'Route for vehicle 0:\n'
            route_distance = 0
            while not routing.IsEnd(index):
                plan_output += ' {} ->'.format(manager.IndexToNode(index))
                previous_index = index
                index = assignment.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
            plan_output += ' {}\n'.format(manager.IndexToNode(index))
            print(plan_output)
            plan_output += 'Route distance: {}miles\n'.format(route_distance)
            return plan_output
    
    
        def main():
            """Entry point of the program."""
            # Instantiate the data problem.
            data = create_data_model()
        
            # Create the routing index manager.
            manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                                   data['num_vehicles'], data['depot'])
        
            # Create Routing Model.
            routing = pywrapcp.RoutingModel(manager)
        
        
            def distance_callback(from_index, to_index):
                """Returns the distance between the two nodes."""
                # Convert from routing variable Index to distance matrix NodeIndex.
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                return data['distance_matrix'][from_node][to_node]
        
            transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        
            # Define cost of each arc.
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
            # Setting first solution heuristic.
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            
            """
            Routing Options - Heuristic algorithms
            """
            
            #search_parameters.first_solution_strategy = (
                #routing_enums_pb2.FirstSolutionStrategy.SWEEP)
            
            """
            Routing Options - Meta - heuristic Algorithms
            """
            
            search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH)
            search_parameters.time_limit.seconds = 1
            search_parameters.log_search = True
        
            # Solve the problem.
            assignment = routing.SolveWithParameters(search_parameters)
            
            # Print solution on console.
            if assignment:
                temp_var = print_solution(manager, routing, assignment)
                temp_var
                
                #This code is to remove the distance from last stop that the vehicle has visited to the vehicle itself and add distance to MRT 
                main_split = temp_var.split('\n')
                index_main = int(main_split[1][-6])
                dist_to_remove = all_sol_list_temp[index_main][0]
                last_stop = list(all_sol_df_temp_pivot)[index_main]
                dist_last_stop2MRT = float(all_points_df[(all_points_df['origin_id']=='AMK_MRT') & (all_points_df['destination_id']==last_stop)]['total_cost'].iloc[0])
                VMT_shuttle = assignment.ObjectiveValue() - dist_to_remove + dist_last_stop2MRT
                print('Objective: {} miles'.format(int(VMT_shuttle)))
                
            
            
            return [VMT_shuttle, temp_var]
        
        
        if __name__ == '__main__':
            x = main()
            x
            VMT_list.append(x[0])
    
    
    
    print ("The total distance travelled is = {} vehicle miles".format(int(sum(VMT_list))))
        
    if len(X)>0:
        unserved_demand = int(sum([val for val in X[:,2:3]]))
        per_unserved_demand = (unserved_demand/tot_demand)*100
    else:
        per_unserved_demand = 0
          
    Objective_function_list.append([int(sum(VMT_list)),per_unserved_demand])    
df = pd.DataFrame.from_records(Objective_function_list)
df.to_csv(r"C:\Users\shyam.rampalli\OneDrive - Nanyang Technological University\documents\Conf - Mar 2\Python codes\Result files\Heuristic\V="+str(fleet_size)+" S="+ str(n_stops)+".csv")    

    
    
    
    
    
    
    
    
    
    
    