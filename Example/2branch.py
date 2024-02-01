#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 11:01:15 2024

@author: c65779
"""

import openturns as ot
import math
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/c65779/Documents/Projet_VIGIE/otlinesampling/otlinesampling")

from LineSampling import LineSampling

################## 2 branch function #######################   
X = ot.RandomVector(ot.Normal(2))

def twoBranch(x):
    x1 = x[0]
    x2 = x[1]
    g1 = 5 + 0.1 * (x1 - x2) ** 2 - (x1 + x2) / math.sqrt(2)
    g2 = 5 + 0.1 * (x1 - x2) ** 2 + (x1 + x2) / math.sqrt(2)
    return [min((g1, g2))]

def firstBranch(x):
    x1 = x[0]
    x2 = x[1]
    g1 = 5 + 0.1 * (x1 - x2) ** 2 - (x1 + x2) / math.sqrt(2)
    return [g1]

def secondBranch(x):
    x1 = x[0]
    x2 = x[1]
    g2 = 5 + 0.1 * (x1 - x2) ** 2 + (x1 + x2) / math.sqrt(2)
    return [g2]

#### Defintion of the failure event ############### 
g_twoBranch = ot.PythonFunction(2, 1, twoBranch)
Y_twoBranch = ot.CompositeRandomVector(g_twoBranch, X)
threshold = 1.5
event_twoBranch = ot.ThresholdEvent(Y_twoBranch, ot.Less(), threshold)

optimAlgo = ot.Cobyla()
algo = ot.FORM(optimAlgo, event_twoBranch, X.getMean())
algo.run()
result = algo.getResult()
alpha_twoBranch = result.getStandardSpaceDesignPoint()

g1 = ot.PythonFunction(2, 1, firstBranch)
Y1 = ot.CompositeRandomVector(g1, X)
event1 = ot.ThresholdEvent(Y1, ot.Less(), threshold)

g2 = ot.PythonFunction(2, 1, secondBranch)
Y2 = ot.CompositeRandomVector(g2, X)
event2 = ot.ThresholdEvent(Y2, ot.Less(), threshold)
unionEvent = ot.UnionEvent([event1,event2])

##### MC reference ################
experiment = ot.MonteCarloExperiment()
algo = ot.ProbabilitySimulationAlgorithm(event_twoBranch, experiment)
algo.setMaximumOuterSampling(1000000)
algo.setBlockSize(10)
algo.setMaximumCoefficientOfVariation(0.01)
algo.run()
print('MC Probability estimate=%.6f' % algo.getResult().getProbabilityEstimate())
#########################################   

##### Line Sampling twoBranch function only along alpha direction ####################################
LS_twoBranch_oneDirection = LineSampling(event_twoBranch,alpha_twoBranch, rootSolver=ot.MediumSafe(ot.Brent(1e-3,1e-3,1e-3,5)),
                                         oppositeDirection = False, activeLS = True, minCoV = 0.05, maxLines=1000, batchSize=1, fixedSeed=True)
LS_twoBranch_oneDirection.run()
LS_result = LS_twoBranch_oneDirection.getResults()
LS_Probability = LS_result['Pf']
LS_CoV = LS_result['CoV']
print("LS_twoBranch_oneDirection Probability = ", LS_Probability[-1])
print("LS_twoBranch_oneDirection CoV = ", LS_CoV[-1])

##### Line Sampling twoBranch function along alpha and opposite alpha directions ####################################
LS_twoBranch_twoDirections = LineSampling(event_twoBranch,alpha_twoBranch, rootSolver=ot.MediumSafe(ot.Brent(1e-3,1e-3,1e-3,5)),
                                          oppositeDirection = True, activeLS = True, minCoV = 0.05, maxLines=1000, batchSize=1, fixedSeed=True)
LS_twoBranch_twoDirections.run()
LS_result = LS_twoBranch_twoDirections.getResults()
LS_Probability = LS_result['Pf']
LS_CoV = LS_result['CoV']
print("LS_twoBranch_twoDirection Probability = ", LS_Probability[-1])
print("LS_twoBranch_twoDirection CoV = ", LS_CoV[-1])

##### Line Sampling union ####################################
LS_union = LineSampling(unionEvent,alpha_twoBranch, rootSolver=ot.MediumSafe(ot.Brent(1e-3,1e-3,1e-3,5)),
                        oppositeDirection = True, activeLS = False, minCoV = 0.05, maxLines=1000, batchSize=1, fixedSeed=True)
LS_union.run()
LS_result = LS_union.getResults()
LS_Probability = LS_result['Pf']
LS_CoV = LS_result['CoV']
print("LS_unionEvent Probability = ", LS_Probability[-1])
print("LS_unionEvent CoV = ", LS_CoV[-1])

x1 = [i[0][0] for i in LS_result["rootPoints"]]
x2 = [i[1][0] for i in LS_result["rootPoints"]]
y1 = [i[0][1] for i in LS_result["rootPoints"]]
y2 = [i[1][1] for i in LS_result["rootPoints"]]
plt.rcParams.update({"text.usetex": True,"font.family": "sans-serif","font.sans-serif": ["Helvetica"],'font.size': 22})
plt.figure(figsize=(8,8))
plt.grid()
plt.scatter(x1,y1,label=r'$g_1(X_1,X_2) = 0$',s=5)
plt.scatter(x2,y2,label=r'$g_2(X_1,X_2) = 0$',s=5)
plt.legend()
plt.xlabel(r'$X_1$')
plt.ylabel(r'$X_2$')
