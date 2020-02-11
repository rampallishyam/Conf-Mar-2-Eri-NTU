# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 16:12:23 2020

@author: shyam.rampalli
"""
# =============================================================================
# V refers to the nd array created for all the vehicles
# X refers to the number of stops present which contains their location and demand values
# =============================================================================


import random
import numpy as np
import matplotlib.pyplot as plt #matplot library for plotting

n_stops = 25   #Number of stops
fleet_size = 10
veh_capacity = 12
max_demand = 15

random.seed(6) #initializing
# first column is x-axis, second: y-axis, third: demand, fourth: dummy variable, no reason to assign this variable.
coords = [[random.random()*100.0, random.random()*100.0, random.randint(2,max_demand), int(0)] for _ in range(n_stops)]
X = np.asarray(coords)

# first column is x-axis, second: y-axis, third: veh capacity, fourth: capacity fulfilled.
coords = [[random.random()*100.0, random.random()*100.0, int(veh_capacity), int(0) ] for _ in range(fleet_size)]
V = np.asarray(coords)

#%%
def Kmeans(V, X, n_clusters):
    
# =============================================================================
#   "This code refers to the Kmeans function which does one iteration of K-means algorithm with fixed cluste centers"     
# =============================================================================
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
# =============================================================================
# A function is defined that draws an arrow showing the match between the vehicle and the nearest stop
# =============================================================================
def drawArrow(x, y):
    plt.arrow(x[0], x[1], y[0] - x[0], y[1] - x[1],
              head_width=0.05, length_includes_head=True)
    
#%%
# =============================================================================
# This function considers the demand factor and draws arrows and re-adjusts the arrays of vehicles and bus stops
# =============================================================================
def demand(initial_prediction, V, X):
    index = 0 
    unserved_list = []
    unserved_array = []
    for veh in list(initial_prediction):
        if index < len(X):
            V[veh][3]+=X[index][2] #Adding all the demand values to the vehicle
            if V[veh][3]<=V[veh][2]:
                drawArrow(y = V[veh],x = X[index])
                V[veh][2]-=X[index][2] #Update vehicle capacity for every arraw that has been drawn
                V[veh][3]-=X[index][2]
                X[index][2] = 0
            else:
                if V[veh][2]>0:
                    X[index][2]-=V[veh][2]
                    drawArrow(y = V[veh],x = X[index])
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
     
    return [V2, fleet_size, unserved_array]

#%%
# =============================================================================
#     This loop iterates over until the full capacity has been utilized
# =============================================================================

for iter in range(0,5,1): 
    
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
    if len(V) == 0:
        print(V)
        break
    else:
    #this for loop is for making the filled capacities variable to zero since capacties are already in the code above
        for val in V:
            val[3] = 0
        plt.savefig(r"C:\Users\shyam.rampalli\OneDrive - Nanyang Technological University\documents\Conf - Mar 2\Python codes\results\Initial_sol" + str(iter) + ".png")
        plt.show()

print ('Vehicle capacity exhausted')
    


