# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:56:50 2020

@author: shyam.rampalli
"""

from __future__ import print_function
import random
import shapefile
import numpy as np
import matplotlib.pyplot as plt #matplot library for plotting
import pandas as pd
from ortools.linear_solver import pywraplp  # Or-tools library

n_stops = 50   #Number of stops
fleet_size = 25
veh_capacity = 12
max_demand = 10
depo = 1

random.seed(2) #initializing
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
    coords_X.append([geom['coordinates'][1],geom['coordinates'][0],atr['name'],random.randint(0,max_demand), int(0)])
coords_X = random.sample(coords_X,n_stops)
X1 = np.asarray(coords_X)
X1_df = pd.DataFrame(np.asarray(coords_X)[:, :3], index = [i for i in range(len(np.asarray(coords_X)))])
X1_df.columns = ['lat','lon','Name']

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
    coords_V.append([geom['coordinates'][1],geom['coordinates'][0],atr['id'],random.randint(0,veh_capacity), int(0)])
coords_V = random.sample(coords_V,fleet_size)
V1 = np.asarray(coords_V)
V1_df = pd.DataFrame(np.asarray(coords_V)[:, :3], index = [i for i in range(len(np.asarray(coords_V)))])
V1_df.columns = ['lat','lon','Name']

coords_M = []
attr_M_list = []
reader= shapefile.Reader(r"C:/Users/shyam.rampalli/OneDrive - Nanyang Technological University/documents/Conf - Mar 2/Data/AMK_MRT.shp")
fields = [field[0] for field in reader.fields[1:]]
for feature in reader.shapeRecords():
    geom = feature.shape.__geo_interface__
    field_names = [field for field in fields] 
    atr = dict(zip(field_names, feature.record))
    attr_M_list.append(atr['Name'])
    coords_M.append([geom['coordinates'][1],geom['coordinates'][0],atr['Name'],random.randint(0,max_demand), int(0)])
M = np.asarray(coords_M)
M_df = pd.DataFrame(np.asarray(coords_M)[:, :3], index = [i for i in range(len(np.asarray(coords_M)))])
M_df.columns = ['lat','lon','Name']

all_df = pd.concat([X1_df, V1_df, M_df])
list_points1 = all_df.values.tolist()
for val in list_points1:
    if type(val[2]) != str:
        val[2] = str(int(val[2]))


#%%

plt.scatter(X1[:, 0].astype('float64'), X1[:, 1].astype('float64'))
plt.scatter(V1[:, 0].astype('float64'), V1[:, 1].astype('float64'))
plt.scatter(M[:, 0].astype('float64'), M[:, 1].astype('float64'),s=120)



#%%

#Creating a pivot table with distances
all_points_df = pd.read_csv(r"C:\Users\shyam.rampalli\OneDrive - Nanyang Technological University\documents\Conf - Mar 2\Data\OD_Matrix_All_points.csv")
all_points_df_pivot = pd.pivot_table(all_points_df, values = 'total_cost', index= 'origin_id', columns= 'destination_id')
all_points_df = pd.DataFrame(all_points_df_pivot)
columns_pivot_table = [val[2] for val in list_points1]


#%%
dist_mat = []
for row in list_points1:
    dist_mat2 = []
    name = row[2]
    for j in columns_pivot_table:
        if j==name:
            dist_mat2.append(0)
        else:
            dist_mat2.append(all_points_df[j][name])
    dist_mat.append(dist_mat2)
    
#%%    
solver = pywraplp.Solver('simple_mip_program',pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)    
Nodes = n_stops
Veh = fleet_size
depo = 1
x = [[0]* (Nodes+Veh+depo) for i in range(Nodes+Veh+depo)]

for i in range(Nodes+Veh+depo):
    for j in range(Nodes+Veh+depo):
        x[i][j] = solver.IntVar(0.0, 1.0, 'x['+str(i)+']['+str(j)+']')

print('Number of variables =', solver.NumVariables())

#%%        
# constraint for diagonal elements to be zero       
constr = 0
for i in range(Nodes+Veh+depo):
    for j in range(Nodes+Veh+depo):
        if i==j:
            constr = constr + x[i][j]    

solver.Add(constr == 0)

# Constraint to prevent trips from MRT to other stops/vehicles
solver.Add(sum(x[Nodes+Veh+depo-1]) == 0)

# constraint to provent any trips to vehicle from stops and to prevent trips from MRT
transpose = list(zip(*x))
for i in range(len(transpose)):
    if i > Nodes-1 and i<= Nodes+Veh-1:
            solver.Add(sum(transpose[i])==0)
    elif i == Nodes+Veh+depo -1:
            solver.Add(sum(transpose[-1][Nodes:Nodes+Veh-1])==0)
            solver.Add(sum(transpose[-1]) == Veh)
            
#Distance from a to b is equal to distance from b to a
for i in range(Nodes+Veh+depo-1):
        for j in range(i+1,Nodes+Veh+depo):
            solver.Add(x[i][j] + x[j][i] <=1)
            
# All veh that reach stop will leave the stop
for i in range(Nodes):
    solver.Add(sum(x[i])-sum(transpose[i]) == 0)
    
# Vehicle leaves the parking locations to visit a single node at once
for i in range(Nodes,Nodes+Veh):
    solver.Add(sum(x[i]) == 1)


#Objective function
obj = 0
for i in range(Nodes+Veh+depo):
    for j in range(Nodes+Veh+depo):
        #objective.SetCoefficient(x[i][j], val[i][j])
        obj = obj + x[i][j]*dist_mat[i][j]

solver.Minimize(obj)
status = solver.Solve()

print('Solution:')
print('Objective value =', solver.Objective().Value())

final_x_val = []
for i in range(Nodes+Veh+depo):
    final_x_val1 = []
    for j in range(Nodes+Veh+depo):
        final_x_val1.append(x[i][j].solution_value())
        for val in list_points1:
            val[0] = float(val[0])
            val[1] = float(val[1])
        if (x[i][j].solution_value()==1.0):
            plt.arrow(list_points1[i][0],list_points1[i][1],list_points1[j][0]-list_points1[i][0],list_points1[j][1]-list_points1[i][1],head_width=0.001,width = 0.00034, length_includes_head=True)
            
    final_x_val.append(final_x_val1)
       
print(final_x_val)



            