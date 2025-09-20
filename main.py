from copy import deepcopy

import numpy as np

# Create V table for the grid
V = np.array([[0,0,0,1],
              [0,np.nan,0,-1],
              [0,0,0,0]])

V_buffer = np.array([[0,0,0,1],
              [0,np.nan,0,-1],
              [0,0,0,0]])
#reward function
def r(i,j):
    if i == 0 and j == 3:
        return 1.0965
    elif i == 1 and j == 3:
        return -1.3182
    elif i == 1 and j == 1:
        return np.nan
    else:
        return -0.1657

x = np.zeros((3,4))
print(x)

#Function to update value of V grid using bellman equation
def v(r, y):
    for i in range(0,3):
        for j in range(0,4):
            x[i,j] = r(i,j)

v(r,3)
print(x)

#transition function to determine new coords
def move(i, j, direction):
    prev_i = i
    prev_j = j
    match direction:
        case 'RIGHT':
            j = j+1
        case 'LEFT':
            j = j-1
        case 'UP':
            i = i-1
        case 'DOWN':
            i = i+1
    if j>3 or j<0 or i>2 or i<0 or (i == 1 and j == 1):
        i = prev_i
        j = prev_j
    return i,j

def val_iteration(y, init_val):
    prev_val = deepcopy(init_val)
    new_val = deepcopy(init_val)
    for i in range(0,3):
        for j in range(0,4):
            val_up = 0.8*(prev_val[move(i,j,'UP')]) + 0.1*(prev_val[move(i,j,'LEFT')]) + 0.1*(prev_val[move(i,j,'RIGHT')])
            val_down = 0.8*(prev_val[move(i,j,'DOWN')]) + 0.1*(prev_val[move(i,j,'LEFT')]) + 0.1*(prev_val[move(i,j,'RIGHT')])
            val_left = 0.8*(prev_val[move(i,j,'LEFT')]) + 0.1*(prev_val[move(i,j,'UP')]) + 0.1*(prev_val[move(i,j,'DOWN')])
            val_right = 0.8*(prev_val[move(i,j,'RIGHT')]) + 0.1*(prev_val[move(i,j,'UP')]) + 0.1*(prev_val[move(i,j,'DOWN')])
            #print("We are at ", i, j, " and vals are ", val_up, val_down, val_left, val_right)
            new_val[i,j] = y * max(val_up,val_down,val_left,val_right)
    new_val[0,3] = prev_val[0,3]
    new_val[1,3] = prev_val[1,3]
    new_val[1,1] = prev_val[1,1]
    return new_val

v1 = val_iteration(1, x)
v2 = val_iteration(1, v1)
v3 = val_iteration(1, v2)
v4 = val_iteration(1, v3)
v5 = val_iteration(1, v4)
v6 = val_iteration(1, v5)
v7 = val_iteration(1, v6)
v8 = val_iteration(1, v7)
v9 = val_iteration(1, v8)
v10 = val_iteration(1, v9)

print(v1, "\n")
print(v2, "\n")
print(v3, "\n")
print(v4, "\n")
print(v5, "\n")
print(v6, "\n")
print(v7, "\n")
print(v8, "\n")
print(v9, "\n")
print(v10, "\n")

