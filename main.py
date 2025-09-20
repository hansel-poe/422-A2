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