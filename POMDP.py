from copy import deepcopy
import numpy as np
from enum import Enum

#starting gr9d
b0 = np.array([[0.111, 0.111, 0.111, 0.000],
               [0.111, np.nan, 0.111, 0.000],
               [0.111, 0.111, 0.111, 0.111]])

#Move function
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

#Enum representing possible evidences of sensor
class Evidence(Enum):
    END = 0
    ONEWALL = 1
    TWOWALLS = 2

#sensor model, returns P(e | s)
def sensor(e, s):
    (i, j) = s
    if j == 2: #third column
        if e == Evidence.END:
            return 0
        elif e ==Evidence.ONEWALL:
            return 0.6806
        elif e ==Evidence.TWOWALLS:
            return 0.3194
    elif (i == 0 and j == 3) or (i == 1 and j == 3):
        if e == Evidence.END:
            return 1
        else:
            return 0
    else:
        if e == Evidence.END:
            return 0
        elif e == Evidence.ONEWALL:
            return 0.2929
        elif e == Evidence.TWOWALLS:
            return 0.7071

#returns p(s | s_b,a)
def transition(s, a, s_b):
    (i,j) = s
    (k,l) = s_b
    sum = 0
    if (a == 'UP'):
        if move(k,l, 'UP') ==  (i,j):
            sum += 0.8
        if move(k,l, 'LEFT') ==  (i,j) or move(k,l, 'RIGHT') ==  (i,j):
            sum += 0.1
    elif (a == 'DOWN'):
        if move(k, l, 'DOWN') == (i, j):
            sum += 0.8
        if move(k, l, 'LEFT') == (i, j) or move(k, l, 'RIGHT') == (i, j):
            sum += 0.1
    elif (a == 'LEFT'):
        if move(k, l, 'LEFT') == (i, j):
            sum += 0.8
        if move(k, l, 'UP') == (i, j) or move(k, l, 'DOWN') == (i, j):
            sum += 0.1
    elif (a == 'RIGHT'):
        if move(k, l, 'RIGHT') == (i, j):
            sum += 0.8
        if move(k, l, 'UP') == (i, j) or move(k, l, 'DOWN') == (i, j):
            sum += 0.1
    return sum

#Iterate one time , returns updated b grid
def b_iter(prev_b, e, a):
    new_b = deepcopy(prev_b)
    for i in range(0, 3):
        for j in range(0, 4):
            summation = 0
            neighbors = [move(i, j, 'UP'), move(i, j, 'RIGHT'),
                         move(i, j, 'DOWN'), move(i, j, 'LEFT')]
            neighbors = [n for n in neighbors if n != (i,j)] #remove all (i,j) from neighbors
            neighbors.append((i,j)) #then add back one copy
            for n in neighbors:
                summation += transition((i,j), a, n) * prev_b[n[0],n[1]]

            new_b[i,j] = sensor(e, (i,j)) * summation
    #normalize
    alpha = 1 / (np.nansum(new_b))
    for i in range(0, 3):
        for j in range(0, 4):
            new_b[i,j] = alpha * new_b[i,j]

    return new_b


def b_sequence(actions, observations, **kwargs):
    prev_b = b0
    for i in range(len(actions)):
        curr_b = b_iter(prev_b, observations[i], actions[i])
        prev_b = curr_b

    return prev_b


#First sequence
a1 =['UP', 'LEFT', 'UP']
e1 =[Evidence.TWOWALLS, Evidence.TWOWALLS, Evidence.TWOWALLS]
result1 = b_sequence(a1, e1)

print("Grid for ['UP', 'LEFT', 'UP'] and [2 walls, 2 walls, 2 walls]:\n")
print(result1, "\n")
print(np.round(result1, 3),'\n')

#Second sequence
a2 =['LEFT', 'DOWN', 'RIGHT']
e2 =[Evidence.ONEWALL, Evidence.ONEWALL, Evidence.ONEWALL]
result2 = b_sequence(a2, e2)

print("Grid for ['LEFT', 'DOWN', 'RIGHT'] and [1 walls, 1 walls, 1 walls]:\n")
print(result2, "\n")
print(np.round(result2, 3),'\n')
print(np.nansum(result2))



