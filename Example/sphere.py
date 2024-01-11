#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 14:00:30 2024

@author: c65779
"""

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

def sphere(x):
    x1 = x[0]
    x2 = x[1]
    res = x1**2+x2**2
    return [res]

#### Defintion of the failure event ############### 
g = ot.PythonFunction(2, 1, sphere)
Y = ot.CompositeRandomVector(g, X)
threshold = 0.2
event = ot.ThresholdEvent(Y, ot.Less(), threshold)
###################################################

#### FORM analysis to initialize the direction alpha #### 
optimAlgo = ot.Cobyla()
algo = ot.FORM(optimAlgo, event, X.getMean())
algo.run()
result = algo.getResult()
alpha=result.getStandardSpaceDesignPoint()
###################################################

# ##### MC for reference ################
experiment = ot.MonteCarloExperiment()
algo = ot.ProbabilitySimulationAlgorithm(event, experiment)
algo.setMaximumCoefficientOfVariation(0.01)
algo.setMaximumOuterSampling(int(1e6))
algo.run()
result = algo.getResult()
probability = result.getProbabilityEstimate()
print("Pf=", probability)
# #########################################################

# ##### Line Sampling ####################################
# LS_oneDirection = LineSampling(event,alpha, rootSolver=ot.MediumSafe(ot.Brent(1e-5,1e-5,1e-8,10)),oppositeDirection = False, activeLS = False,
#               minCoV = 0.05, maxLines=1000, batchSize=1, fixedSeed=True)
# LS_oneDirection.run()
# LS_result = LS_oneDirection.getResults()
# LS_Probability = LS_result['Pf_MarginalEvent']
# LS_CoV = LS_result['CoV']
# print("LS Probability = ", LS_Probability[-1])
# print("LS CoV = ", LS_CoV[-1])

LS_twoDirection = LineSampling(event,alpha, rootSolver=ot.SafeAndSlow(ot.Brent(1e-3,1e-3,1e-3,5),2,0.25),oppositeDirection = True, activeLS = False,
              minCoV = 0.01, maxLines=5000, batchSize=1, fixedSeed=True)
LS_twoDirection.run()
LS_result = LS_twoDirection.getResults()
LS_Probability = LS_result['Pf_MarginalEvent']
LS_CoV = LS_result['CoV']
print("LS Probability = ", LS_Probability[-1])
print("LS CoV = ", LS_CoV[-1])

