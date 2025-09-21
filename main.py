from copy import deepcopy
import math
import numpy as np
from enum import Enum
from functools import partial

# Create initial v table for the grid
v0 = np.array([[0,0,0,1],
              [0,np.nan,0,-1],
              [0,0,0,0]])

#reward function
def reward(i,j):
    if i == 0 and j == 3:
        return 1.0965
    elif i == 1 and j == 3:
        return -1.3182
    elif i == 1 and j == 1:
        return np.nan
    else:
        return -0.1657

#Reward customized for (2, 0)/(1,1) in lecture
def rewardCustom(val,i,j):
    if i == 0 and j == 3:
        return 1.0965
    elif i == 1 and j == 3:
        return -1.3182
    elif i == 1 and j == 1:
        return np.nan
    elif i == 2 and j == 0:
        return val
    else:
        return -0.1657

#Enums representing actions
class Action(Enum):
    U = 0
    D = 1
    L = 2
    R = 3

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


def val_iteration(r, y, prev_val):
    new_val = deepcopy(prev_val)
    for i in range(0,3):
        for j in range(0,4):
            val_up = 0.8*(prev_val[move(i,j,'UP')]) + 0.1*(prev_val[move(i,j,'LEFT')]) + 0.1*(prev_val[move(i,j,'RIGHT')])
            val_down = 0.8*(prev_val[move(i,j,'DOWN')]) + 0.1*(prev_val[move(i,j,'LEFT')]) + 0.1*(prev_val[move(i,j,'RIGHT')])
            val_left = 0.8*(prev_val[move(i,j,'LEFT')]) + 0.1*(prev_val[move(i,j,'UP')]) + 0.1*(prev_val[move(i,j,'DOWN')])
            val_right = 0.8*(prev_val[move(i,j,'RIGHT')]) + 0.1*(prev_val[move(i,j,'UP')]) + 0.1*(prev_val[move(i,j,'DOWN')])
            #print("We are at ", i, j, " and vals are ", val_up, val_down, val_left, val_right)
            new_val[i,j] = r(i,j) +  y * max(val_up,val_down,val_left,val_right)
    #Non changing states
    new_val[0,3] = prev_val[0,3]
    new_val[1,3] = prev_val[1,3]
    new_val[1,1] = prev_val[1,1]
    return new_val

#loop iterations until they converge at the 1st 4 decimal places, returns the finished iteration and counter
def converge(r, y):
    prev_val = v0
    counter = 0
    max_iter = 1000
    while True:
        converged_yet = True
        counter = counter + 1
        curr_val = val_iteration(r,y, prev_val)
        for i in range(0,3):
            for j in range(0,4):
                if i == 1 and j == 1:
                    continue
                trunc_curr = math.trunc(curr_val[i,j]*10000)/10000
                trunc_prev = math.trunc(prev_val[i,j]*10000)/10000
                #compare 1st 4 decimal places
                converged_yet = converged_yet and (trunc_curr == trunc_prev)
        if converged_yet:
            return curr_val, counter
        elif counter >= max_iter:
            print("maxIter reached, terminating program, returning the last grid")
            return curr_val, counter
        else:
            prev_val = curr_val

#find optimal policy
def optimalPolicy(grid):
    policy = [] #list of empty str
    for i in range(0, 3):
        policy.append([])
        for j in range(0, 4):
            val_up = 0.8 * (grid[move(i, j, 'UP')]) + 0.1 * (grid[move(i, j, 'LEFT')]) + 0.1 * (
            grid[move(i, j, 'RIGHT')])
            val_down = 0.8 * (grid[move(i, j, 'DOWN')]) + 0.1 * (grid[move(i, j, 'LEFT')]) + 0.1 * (
            grid[move(i, j, 'RIGHT')])
            val_left = 0.8 * (grid[move(i, j, 'LEFT')]) + 0.1 * (grid[move(i, j, 'UP')]) + 0.1 * (
            grid[move(i, j, 'DOWN')])
            val_right = 0.8 * (grid[move(i, j, 'RIGHT')]) + 0.1 * (grid[move(i, j, 'UP')]) + 0.1 * (
            grid[move(i, j, 'DOWN')])

            policy[i].append(Action(np.argmax([val_up, val_down, val_left, val_right])).name)
    policy[0][3] = "-"
    policy[1][3] = "-"
    policy[1][1] = "-"

    return np.array(policy)


def rewardBoundaryBruteForce(arr):
    for val in arr:
        print("Current iteration:", val)
        customReward = partial(rewardCustom, val)#fill the first argument with val
        finalGrid, nIter = converge(customReward,1)

        print(finalGrid)
        print("# iter:", nIter, "\n")
        if(nIter == 1000):
            print("Terminating program.....\n")
            return finalGrid,nIter

#Main code starts here

# v1 = val_iteration(reward,1, v0)
# v2 = val_iteration(reward,1, v1)
# v3 = val_iteration(reward,1, v2)
# v4 = val_iteration(reward,1, v3)
# v5 = val_iteration(reward,1, v4)
# v6 = val_iteration(reward,1, v5)
# v7 = val_iteration(reward,1, v6)
# v8 = val_iteration(reward,1, v7)
# v9 = val_iteration(reward,1, v8)
# v10 = val_iteration(reward,1, v9)
# v11 = val_iteration(reward,1, v10)
# v12 = val_iteration(reward,1, v11)
# v13 = val_iteration(reward,1, v12)
# v14 = val_iteration(reward,1, v13)
# v15 = val_iteration(reward,1, v14)
# v16 = val_iteration(reward,1, v15)
# v17 = val_iteration(reward,1, v16)
# v18 = val_iteration(reward,1, v17)
# print("v18:")
# print(v18, "\n")
#
# #test our code
# final_grid, n_iter = converge(reward, 1)
# opt_policy = optimalPolicy(final_grid)
#
# print("Result:")
# print(final_grid)
# print("# iter:", n_iter, "\n")
# print("Optimal Policy: ")
# print(opt_policy)

arr = np.linspace(-0.16,0,17)
arr2 = np.linspace(0,0.2, 21)
print(arr2)

#rewardBoundaryBruteForce(arr)
failGrid, nIter = rewardBoundaryBruteForce(arr2) #maxIter reached when iterating 0.03, so reward boundary is 0.02
opt_policy_failGrid = optimalPolicy(failGrid)
print("Optimal Policy for Fail Grid: ")
print(opt_policy_failGrid)




