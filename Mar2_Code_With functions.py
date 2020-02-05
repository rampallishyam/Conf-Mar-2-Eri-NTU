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

n_stops = 50   #Number of stops
fleet_size = 20
veh_capacity = 12
max_demand = 10

random.seed(1) #initializing
coords = [[random.random()*100.0, random.random()*100.0, random.randint(2,max_demand), int(0)] for _ in range(n_stops)]
X = np.asarray(coords)
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
    km = KMeans( n_clusters )
    km.cluster_centers_ = V
    # This gives an initial solution using K-means approach
    initial_prediction = km.predict(X)
    return initial_prediction

#%%
# Scatter plot of stops in the selected area
plt.scatter(X[:, 0], X[:, 1])
# Scatter plot of vehicle locations
plt.scatter(V[:, 0], V[:, 1])
fig, ax = plt.subplots()

#%%
# =============================================================================
# A function is defined that draws an arrow showing the match between the vehicle and the nearest stop
# =============================================================================
def drawArrow(V, X):
    plt.arrow(X[0], X[1], V[0] - X[0], V[1] - X[1],
              head_width=0.05, length_includes_head=True)
    
#%%
# =============================================================================
# This function considers the demand factor and draws arrows and re-adjusts the arrays of vehicles and bus stops
# =============================================================================
def demand(initial_prediction, V, X):
    index = 0 
    unserved_list = []
    for veh in list(initial_prediction):
        if index < len(X):
            V[veh][3]+=X[index][2] #Adding all the demand values to the vehicle
            if V[veh][3]<=V[veh][2]:
                drawArrow(V[veh],X[index])
                #plt.text(V[veh] * (1 + 0.01), X[index] * (1 + 0.01) , X[index][2], fontsize=12)
            else:
                unserved_list.append(X[index]) #creating a list of stops that are not served in the present iteration
                unserved_array = np.asarray(unserved_list)
            index+=1
    #A loop to filter all the points that need to served and all the vehicles that can still be used    
    V2 = []
    for veh in V:
        if veh[2]>0:
            V2.append(veh)
    V2= np.asarray(V2)
     
    return V, fleet_size, unserved_array

#%%
# =============================================================================
#     This loop iterates over until the full capacity has been utilized
# =============================================================================

for iter in range(0,5,1): 
    
    Kmeans(V=V,X=X, n_clusters=fleet_size)
    initial_prediction = Kmeans(V=V,X=X, n_clusters=fleet_size)
    demand(initial_prediction= initial_prediction,V=V,X=X)
    # Scatter plot of stops in the selected area
    plt.scatter(X[:, 0], X[:, 1])
    # Scatter plot of vehicle locations
    plt.scatter(V[:, 0], V[:, 1])
    V = demand(initial_prediction=initial_prediction, V=V,X=X)[0]    #Update the vehicles by the non-assigned or still with existing capacity
    X = demand(initial_prediction=initial_prediction, V=V,X=X)[2]
    fleet_size = demand(initial_prediction=initial_prediction, V=V,X=X)[1]
    plt.show()

    
    


