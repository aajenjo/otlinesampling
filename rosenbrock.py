#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 13:54:51 2024

@author: c65779
"""

import openturns as ot
import numpy as np
import math
from classLineSampling import LineSampling

X = ot.RandomVector(ot.Normal(2))

def g_rosen(X):
    res = sum([100*(X[i+1]-X[i]**2)**2+(X[i]-1)**2 for i in range(len(X)-1)])
    return [res]

#### Defintion of the failure event ############### 
g = ot.PythonFunction(2, 1, g_rosen)
Y = ot.CompositeRandomVector(g, X)
threshold = 0.1
event = ot.ThresholdEvent(Y, ot.Less(), threshold)
###################################################

#### FORM analysis to initialize the direction alpha #### 
optimAlgo = ot.AbdoRackwitz()
algo = ot.FORM(optimAlgo, event, X.getMean())
algo.run()
result = algo.getResult()
alpha=result.getStandardSpaceDesignPoint()
###################################################

##### MC for reference ################
experiment = ot.MonteCarloExperiment()
algo = ot.ProbabilitySimulationAlgorithm(event, experiment)
algo.setMaximumCoefficientOfVariation(0.05)
algo.setMaximumOuterSampling(int(1e6))
algo.run()
result = algo.getResult()
probability = result.getProbabilityEstimate()
print("Pf=", probability)
#########################################################

##### Line Sampling ####################################
LS = LineSampling(event,alpha, rootSolver=ot.MediumSafe(ot.Brent(1e-5,1e-5,1e-8,10)),oppositeDirection = False, activeLS = True,
             minCoV = 0.05, maxLines=3000, batchSize=1, fixedSeed=True)
LS.run()
LS_result = LS.getResults()
LS_Probability = LS_result['Pf_MarginalEvent']
LS_CoV = LS_result['CoV']

print("LS Probability = ", LS_Probability[-1])
print("LS CoV = ", LS_CoV[-1])
print("Number of line searches = ", len(LS_Probability))

##### Line Sampling ####################################
LS_med = LineSampling(event,alpha, rootSolver=ot.MediumSafe(ot.Brent(1e-5,1e-5,1e-8,10),3,0.1),oppositeDirection = False, activeLS = True,
             minCoV = 0.05, maxLines=3000, batchSize=1, fixedSeed=True)
LS_med.run()
LS_result = LS_med.getResults()
LS_Probability = LS_result['Pf_MarginalEvent']
LS_CoV = LS_result['CoV']

print("LS Probability = ", LS_Probability[-1])
print("LS CoV = ", LS_CoV[-1])
print("Number of line searches = ", len(LS_Probability))

##### Line Sampling ####################################
LS_safe = LineSampling(event,alpha, rootSolver=ot.SafeAndSlow(ot.Brent(1e-5,1e-5,1e-8,10),3,0.1),oppositeDirection = False, activeLS = True,
             minCoV = 0.05, maxLines=3000, batchSize=1, fixedSeed=True)
LS_safe.run()
LS_result = LS_safe.getResults()
LS_Probability = LS_result['Pf_MarginalEvent']
LS_CoV = LS_result['CoV']

print("LS Probability = ", LS_Probability[-1])
print("LS CoV = ", LS_CoV[-1])
print("Number of line searches = ", len(LS_Probability))
