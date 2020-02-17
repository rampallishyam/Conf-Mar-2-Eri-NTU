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


import random
import numpy as np
import matplotlib.pyplot as plt #matplot library for plotting

n_stops = 30   #Number of stops
fleet_size = 20
veh_capacity = 12
max_demand = 10

random.seed(7) #initializing
# first column is x-axis, second: y-axis, third: demand, fourth: dummy variable, no reason to assign this variable.
coords = [[random.random()*100.0, random.random()*100.0, random.randint(0,max_demand), int(0)] for _ in range(n_stops)]
X = np.asarray(coords)

# first column is x-axis, second: y-axis, third: veh capacity, fourth: capacity fulfilled.
coords = [[random.random()*100.0, random.random()*100.0, int(veh_capacity), int(0) ] for _ in range(fleet_size)]
V = np.asarray(coords)

#%%
def Kmeans(V, X, n_clusters):
    
# ======================================================================================================================
#   "This code refers to the Kmeans function which does one iteration of K-means algorithm with fixed cluste centers"     
# ======================================================================================================================
    """ 
    This functions two arrays and performs one iteration of Kmeans which means it predicts the clusters centers each point belongs to                         
    Cluster centers are the vehcile locations.
    """    
    from sklearn.cluster import KMeans
    # Artificially fix your centroids
    km = KMeans( n_clusters = fleet_size)
    km.cluster_centers_ = V[:,:-2]
    # This gives an initial solution using K-means approach
    initial_prediction = km.predict(X[:,:-2])
    return initial_prediction

#%%
# Scatter plot of stops in the selected area
plt.scatter(X[:, 0], X[:, 1])
# Scatter plot of vehicle locations
plt.scatter(V[:, 0], V[:, 1])
#fig, ax = plt.subplots()

#%%
# ======================================================================================================
# A function is defined that draws an arrow showing the match between the vehicle and the nearest stop
# ======================================================================================================
def drawArrow(x, y):
    plt.arrow(x[0], x[1], y[0] - x[0], y[1] - x[1],
              head_width=0.05, length_includes_head=True)
    
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
                    solution_list.append([list(V[veh]),list(X[index])])
                    V[veh][2]-=X[index][2] #Update vehicle capacity for every arraw that has been drawn
                    V[veh][3]-=X[index][2]
                    X[index][2] = 0
                else:
                    if V[veh][2]>0:
                        X[index][2]-=V[veh][2]
                        drawArrow(y = V[veh],x = X[index])
                        solution_list.append([list(V[veh]),list(X[index])])
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
    plt.scatter(X[:, 0], X[:, 1])
    # Scatter plot of vehicle locations
    plt.scatter(V[:, 0], V[:, 1])
    for stop in X:
        plt.text(stop[0] * (1 + 0.01), stop[1] * (1 + 0.01) ,stop[2], fontsize=12)
    for stop in V:
        plt.text(stop[0] * (1 + 0.01), stop[1] * (1 + 0.01) ,stop[2], fontsize=12)
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
        plt.show()

print ('Either Vehicle capacity exhausted or all demands are matched')


#%%
"""
# =============================================================================
#                Now this code is to process the result
# =============================================================================
    
"""
all_sol_list = []
for index in range(0,len(sol_list),1):
    all_sol_list+=sol_list[index]

for val in all_sol_list:
    del val[0][2:]
    del val[1][2:]
    

    
