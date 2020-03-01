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
    numberOfVariable = para['numberOfVariable']
    numOfTimeStep = para['numberOfTimeStep']
    
    Xic = para['IC value']
    X = np.full((numberOfNode*numberOfVariable, 1), Xic)
    X0 = np.full((numberOfNode*numberOfVariable, 1), Xic)
    XProfile = np.zeros((numberOfNode*numberOfVariable, 
                         numOfTimeStep + 1))
    F = np.zeros((numberOfNode*numberOfVariable, 1))
    Jacobian = np.zeros((numberOfNode*numberOfVariable, 
                         numberOfNode*numberOfVariable))
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
    #print(" J=",J," F=",F)
    dx = np.linalg.solve(J, F)
    
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
        #print(norm)
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
    #print(cache['X'])
    if para['N']<any(cache['X']) or any(cache['X'])<0:
        cache['Stop']=True
        return cache
    cache['F'] = func(para, cache)
    cache['Jacobian'] = jacobian(para, cache)
    return cache



def func(para, cache):
    N = para['N']
    beta1 = para['beta1']
    r1 = para['r1']
    beta2 = para['beta2']
    r2 = para['r2']
    R1 = para['R1']
    R2 = para['R2']
    sigma = para['sigma']
    dt = para['deltaTime']
    X0 = cache['X0']
    X = cache['X']
    F = cache['F']
    
    F[0] = (X[0]-X0[0])/dt - sigma*X0[2] + R2*X0[0]
    F[1] = (X[1]-X0[1])/dt - R1*X0[2] - R2 * X0[0]
    F[2] = (X[2]-X0[2])/dt - r1*beta1*X0[0]/N*(N-X0[0]-X0[1]-X0[2])\
           +(sigma+R1)*X0[2]- r2*beta2*X[2]/N*(N-X0[0]-X0[1]-X0[2])
    return F



def jacobian(para, cache):
    J = cache['Jacobian']
    numberOfVariable = para['numberOfVariable']
    epsilon = 1E-10
    for v in range(numberOfVariable):
        cache['X0'][v] += epsilon
        F_p1 = func(para, cache).flatten()
        cache['X0'][v] -= 2*epsilon
        F_m1 = func(para, cache).flatten()
        cache['X0'][v] += epsilon
        tempJ = (F_p1 - F_m1) / 2 / epsilon
        J[:,v] = tempJ.flatten()
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
    
    print("SEIR Solver")
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
    df.columns=['Infectious','Recovered','Exposed']
    df.index.name='Day'
    df['Susceptible'] = para['N'] - df['Infectious'] \
                        - df['Recovered'] - df['Exposed']
    df.plot()
    return



def parameter():
    """ Generate parameter
    
    1. Generate system-level parameters
    2. Generate material properties, grid, time, bcs
    
    Return: a Pandas series
    """
    column = 'values'
    df = pd.Series(dtype=None, name = column)
    df = df.astype('object')
    
    # System-level 
    df.at['problem'] = 'SEIR'
    df.at['SpatialDiscretize'] = 'CenteredDifferencing'
    df.at['TimeDiscretize'] = 'BackwardEular'
    df.at['ODEsolver'] = 'NewtonIteration'
    df.at['linearSolver'] = 'numpy linalg'
    
    # Problem parameters
    df.at['N'] = 10000       #总人口 total population
    df.at['beta1'] = 0.03    #致病率(由感染者传染) infectious rate (from infected to susceptible)
    df.at['r1'] = 10         #接触率(由感染者传染) contacting rate (from infected to susceptible)
    df.at['beta2'] = 0.03    #致病率(由潜伏者传染) infectious rate (from latent to susceptible)
    df.at['r2'] = 10         #接触率(由潜伏者传染) contacting rate (from latent to susceptible)
    df.at['sigma'] = 0.07    #潜伏患病率 incubation rate (unit,1/sigma days)
    df.at['R1'] = 0.01       #潜伏康复率 recovery rate (from latent)
    df.at['R2'] = 0.01       #患病康复率 recovery rate (from infected)
    
    # Grid
    df.at['numberOfNode'] = 1
    df.at['numberOfVariable'] = 3
    
    # Solution control
    df.at['numberOfTimeStep'] =300 
    df.at['deltaTime'] = 1.
    df.at['maxIteration'] = 100
    df.at['convergence'] = 1E-8
    df.at['relaxation'] = 0.8
    
    # Initial conditions
    # x1 = Infectious, x2 = recovered, x3 = exposed
    # suspectible = N - x1 - x2 - x3
    df.at['IC value'] = np.array([1., 0., 0.]).reshape(-1,1)    
    return df



if __name__ == "__main__":
    para = parameter()
    cache = solve(para)
    plot(para, cache)




