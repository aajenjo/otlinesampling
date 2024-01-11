#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 11:01:15 2024

@author: c65779
"""

import openturns as ot
import numpy as np
import math
from classLineSampling import LineSampling

################## NAIS 4 branch #######################   
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

def thirdBranch(x):
    x1 = x[0]
    x2 = x[1]
    g3 = (x1 - x2) + 9 / math.sqrt(2)
    return [g3]

def fourthBranch(x):
    x1 = x[0]
    x2 = x[1]
    g4 = (x2 - x1) + 9 / math.sqrt(2)
    return [g4]

#### Defintion of the failure event ############### 
g_twoBranch = ot.PythonFunction(2, 1, twoBranch)
Y_twoBranch = ot.CompositeRandomVector(g_twoBranch, X)
threshold = 1
event_twoBranch = ot.ThresholdEvent(Y_twoBranch, ot.Less(), threshold)
optimAlgo = ot.Cobyla()
algo = ot.FORM(optimAlgo, event_twoBranch, X.getMean())
algo.run()
result = algo.getResult()
alpha_twoBranch = result.getStandardSpaceDesignPoint()

g1 = ot.PythonFunction(2, 1, firstBranch)
Y1 = ot.CompositeRandomVector(g1, X)
threshold = 1
event1 = ot.ThresholdEvent(Y1, ot.Less(), threshold)
g2 = ot.PythonFunction(2, 1, secondBranch)
Y2 = ot.CompositeRandomVector(g2, X)
threshold = 1
event2 = ot.ThresholdEvent(Y2, ot.Less(), threshold)
unionEvent = ot.UnionEvent([event1,event2])

##### MC reference ################
experiment = ot.MonteCarloExperiment()
algo = ot.ProbabilitySimulationAlgorithm(event_twoBranch, experiment)
algo.setMaximumOuterSampling(1000000)
algo.setBlockSize(10)
algo.setMaximumCoefficientOfVariation(0.01)
algo.run()
print('Probability estimate=%.6f' % algo.getResult().getProbabilityEstimate())
#########################################   

##### Line Sampling ####################################
LS_twoBranch_oneDirection = LineSampling(event_twoBranch,alpha_twoBranch, rootSolver=ot.MediumSafe(ot.Brent(1e-5,1e-5,1e-8,10)),oppositeDirection = False, activeLS = False,
              minCoV = 0.05, maxLines=1000, batchSize=1, fixedSeed=True)
LS_twoBranch_oneDirection.run()
LS_result = LS_twoBranch_oneDirection.getResults()
LS_Probability = LS_result['Pf_MarginalEvent']
LS_CoV = LS_result['CoV']
print("LS Probability = ", LS_Probability[-1])
print("LS CoV = ", LS_CoV[-1])

LS_twoBranch_twoDirections = LineSampling(event_twoBranch,alpha_twoBranch, rootSolver=ot.MediumSafe(ot.Brent(1e-5,1e-5,1e-8,10)),oppositeDirection = True, activeLS = False,
              minCoV = 0.05, maxLines=1000, batchSize=1, fixedSeed=True)
LS_twoBranch_twoDirections.run()
LS_result = LS_twoBranch_twoDirections.getResults()
LS_Probability = LS_result['Pf_MarginalEvent']
LS_CoV = LS_result['CoV']
print("LS Probability = ", LS_Probability[-1])
print("LS CoV = ", LS_CoV[-1])

##### Line Sampling union ####################################
LS_union = LineSampling(unionEvent,alpha_twoBranch, rootSolver=ot.MediumSafe(ot.Brent(1e-5,1e-5,1e-8,10)),oppositeDirection = True, activeLS = False,
             minCoV = 0.05, maxLines=1000, batchSize=1, fixedSeed=True)
LS_union.run()
LS_result = LS_union.getResults()
LS_Probability = LS_result['Pf_MarginalEvent']
LS_CoV = LS_result['CoV']
print("LS Probability = ", LS_Probability[-1])
print("LS CoV = ", LS_CoV)

