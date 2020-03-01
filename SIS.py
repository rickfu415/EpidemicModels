# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 12:16:38 2020

@author: RickFu
"""
import pandas as pd
import numpy as np
import time



def initialize(para):
    """ Initialize key data
    
    Return: a dictionary
    """
    numberOfNode = para['numberOfNode']
    numOfTimeStep = para['numberOfTimeStep']
    
    Xic = para['IC value']
    X = np.full((numberOfNode, 1), Xic)
    X0 = np.full((numberOfNode, 1), Xic)
    XProfile = np.zeros((numberOfNode, numOfTimeStep + 1))
    F = np.zeros((numberOfNode, 1))
    Jacobian = np.zeros((numberOfNode, numberOfNode))
    XProfile[:,0] = X.reshape(1,-1)
    cache = {'X':X,'X0':X0,'XProfile':XProfile,
             'F':F,'Jacobian':Jacobian,
             'Log':pd.DataFrame(),
             'Stop':False}
    return cache



def storeUpdateResult(cache):
    """ Store results
    Update T0
    Store temperaure results into a dataframe and 
    save it in the cache.
    """
    
    timeStep = cache['ts']
    XProfile = cache['XProfile']
    X = cache['X']
    cache['X0'] = X.copy()
    XProfile[:,timeStep] = X.reshape(1,-1)
    return cache



def solveLinearSystem(para, cache):
    """ Solve Ax=B
    
    Process:
        1. Get A = Jacobian matrix (Jacobian)
        2. Get B = Right hand side equation (F)
        3. Calculate dT
        4. Update T
        5. Store in cache
        
    Return: a dictionary
    """
    relax = para['relaxation']
    J = cache['Jacobian']
    F = cache['F']
    #dx = np.linalg.solve(J, F)
    dx = F/J
    #print(" J=",J," F=",F,' dx=',dx)
    X = cache['X']
    X = dx * relax + X
    cache['X']=X
    cache['dx'] = dx
    return cache



def newtonIteration(para, cache):
    """ Newton's Iteration for Equation System
    
    Process:
        1. Get max iteratino, convergence limit
        2. Call assemble function to get Jacobian and F(RHS)
        3. Solve for dT, update solution
        4. Evaluate F, get value of 2-norm
        5. If solution converged, break, output to screen and
           return cache.
    
    """
    
    maxIteration = para['maxIteration']
    convergence = para['convergence']
    dt = para['deltaTime']
    log = cache['Log']
    ts = cache['ts']
    for n in range(maxIteration):
        cache = assemble(para, cache)
        F = cache['F']
        norm = np.linalg.norm(F)
        if norm < convergence or cache['Stop']==True:
            log.loc[ts,'PhysicalTime'] = dt*ts
            log.loc[ts,'Iteration'] = n+1
            log.loc[ts,'Residual'] = norm
            break
        cache = solveLinearSystem(para, cache)
    print(' [','{:3.0f}'.format(ts), ']',
          ' [','{:6.2f}'.format(ts*dt),']',
          ' [','{:2.0f}'.format(n+1), ']',
          ' [','{:8.2E}'.format(norm),']')
    return cache



def assemble(para, cache):
    """ Assemble linear system Jacobian * dx = F
    
    Process:
        0. Obtain relevant informations
        1. Assemble Jacobian
        3. Assemble F
    
    Return: dictionary containing cache data
    """
    if para['N']<cache['X'] or cache['X']<0:
        cache['Stop']=True
        return cache
    cache['F'] = func(para, cache)
    cache['Jacobian'] = jacobian(para, cache)
    return cache



def func(para, cache):
    N = para['N']
    beta = para['beta']
    r = para['r']
    R = para['R']
    dt = para['deltaTime']
    X0 = cache['X0']
    X = cache['X']
    
    F = (X-X0)/dt - r*beta*X0 + r*beta/N*X0**2 + R*X0
    return F



def jacobian(para, cache):
    epsilon = 1E-8
    cache['X0'] += epsilon
    F_p1 = func(para,cache)
    cache['X0'] -= 2*epsilon
    F_m1 = func(para,cache)
    cache['X0'] += epsilon
    J = (F_p1 - F_m1) / 2 / epsilon
    return J



def solve(para):
    """ Main function to solve PDE
    
    Input: a Pandas series containing all parameters
    
    Process:
        1. Initialize cache
        2. Time marching 
        3. Newton's iteration for discretized PDE for singe time 
           step
        4. Update X, save result to X profile
    
    Return: X profile as final result
    """
    
    print(" SIS Solver")
    start = time.time()
    cache = initialize(para)
    numOfTimeStep = para['numberOfTimeStep']
    print(' [Step] [Pysical Time] [Iteration] [Residue]')
    for timeStep in range(1, numOfTimeStep+1):
        cache['ts'] = timeStep
        if cache['Stop']==True:
            break
        cache = newtonIteration(para, cache)
        cache = storeUpdateResult(cache)
    runtime = time.time() - start
    print('[Cost] CPU time spent','%.3f'%runtime,'s')
    return cache



def plot(para, cache):
    x = cache['XProfile']
    df = pd.DataFrame(x).T
    df.columns=['Infectious']
    df.index.name='Day'
    df['Susceptible'] = para['N'] - df['Infectious']
    df.plot()
    return


def parameter():
    """ Generate parameter
    
    1. Generate system-level parameters
    2. Generate material properties, grid, time, bcs
    
    Return: a Pandas series
    """
    column = 'values'
    df = pd.Series(name = column)
    df = df.astype('object')
    
    # System-level 
    df.at['problem'] = 'SIS'
    df.at['SpatialDiscretize'] = 'CenteredDifferencing'
    df.at['TimeDiscretize'] = 'BackwardEular'
    df.at['ODEsolver'] = 'NewtonIteration'
    df.at['linearSolver'] = 'numpy linalg'
    
    # Problem parameters
    df.at['N'] = 1000 #total population
    df.at['beta'] = 0.005 #infection rate
    df.at['r'] = 100 #contact rate
    df.at['R'] = 0.1 #recovery rate
    
    # Grid
    df.at['numberOfNode'] = 1
    
    # Solution
    df.at['numberOfTimeStep'] =50#365*20#400
    df.at['deltaTime'] = 1.
    df.at['maxIteration'] = 100
    df.at['convergence'] = 1E-5
    df.at['relaxation'] = 0.9 # value in [0-1] Very sensitive!!!
    
    # Initial conditions
    df.at['IC value'] = 1.
    
    # Boundary conditions
    return df



if __name__ == "__main__":
    para = parameter()
    cache = solve(para)
    plot(para, cache)







