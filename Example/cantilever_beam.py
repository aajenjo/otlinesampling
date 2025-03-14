#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 11:01:15 2024

@author: c65779
"""

import openturns as ot
from openturns.usecases import cantilever_beam
import numpy as np
import sys
sys.path.append("/home/c65779/Documents/Projet_VIGIE/otlinesampling/otlinesampling")
from LineSampling import LineSampling

################## cantilever_beam #######################

#### Defintion of the failure event ############### 
cb = cantilever_beam.CantileverBeam()
distribution = cb.distribution
model = cb.model
vect = ot.RandomVector(distribution)
G = ot.CompositeRandomVector(model, vect)
event = ot.ThresholdEvent(G, ot.Greater(), 0.30)
###################################################

#### FORM analysis to initialize the direction alpha #### 
optimAlgo = ot.AbdoRackwitz()
optimAlgo.setMaximumEvaluationNumber(100)
optimAlgo.setMaximumAbsoluteError(1.0e-10)
optimAlgo.setMaximumRelativeError(1.0e-10)
optimAlgo.setMaximumResidualError(1.0e-10)
optimAlgo.setMaximumConstraintError(1.0e-10)
algo = ot.FORM(optimAlgo, event, distribution.getMean())
algo.run()
result = algo.getResult()
alpha=result.getStandardSpaceDesignPoint()
#########################################################

##### Importance Sampling for reference ################
standardSpaceDesignPoint = result.getStandardSpaceDesignPoint()
dimension = distribution.getDimension()
myImportance = ot.Normal(dimension)
myImportance.setMean(standardSpaceDesignPoint)
experiment = ot.ImportanceSamplingExperiment(myImportance)
standardEvent = ot.StandardEvent(event)
algo = ot.ProbabilitySimulationAlgorithm(standardEvent, experiment)
algo.setMaximumCoefficientOfVariation(0.01)
algo.setMaximumOuterSampling(100000)
algo.run()
result = algo.getResult()
IS_probability = result.getProbabilityEstimate()
print("IS Probability = ", IS_probability)
#########################################################

##### Line Sampling ####################################
LS = LineSampling(event,alpha, rootSolver=ot.MediumSafe(ot.Brent(1e-3,1e-3,1e-3,5)),oppositeDirection = False, activeLS = True,
             minCoV = 0.05, maxLines=1000, batchSize=1)
LS.run()
LS_result = LS.getResults()
LS_Probability = LS_result['Pf']
LS_CoV = LS_result['CoV']

print("LS_default Probability = ", LS_Probability[-1])
print("LS_default CoV = ", LS_CoV[-1])
print("Number of line searches = ", len(LS_Probability))

# ##### Change the stopping criteria CoV=0.01 ##############
LS.setMinCov(0.01)
LS.setMaximumLines(5000)
LS.run(reset=False)
LS_result = LS.getResults()
LS_Probability = LS_result['Pf']
LS_CoV = LS_result['CoV']

print("LS_newCoV Probability = ", LS_Probability[-1])
print("LS_newCoV CoV = ", LS_CoV[-1])
print("Number of line searches = ", len(LS_Probability))
# #########################################################

# ##### update alpha with non optimal direction and set activeLS to False ####################################
alpha_nonOptimal = [-0.3,0.7,0.4,-0.4]
LS_noUpdate = LineSampling(event,alpha_nonOptimal, rootSolver=ot.MediumSafe(ot.Brent(1e-3,1e-3,1e-3,5)),oppositeDirection = False, activeLS = False,
              minCoV = 0.05, maxLines=1000, batchSize=1)
LS_noUpdate.run()
LS_result = LS_noUpdate.getResults()
LS_Probability = LS_result['Pf']
LS_CoV = LS_result['CoV']

print("LS_noUpdate Probability = ", LS_Probability[-1])
print("LS_noUpdate CoV = ", LS_CoV[-1])
print("Number of line searches = ", len(LS_Probability))
print("Updated directions alpha = ", LS_result['alpha'])

###### set activeLS to True ####################
LS_Update = LineSampling(event,alpha_nonOptimal, rootSolver=ot.MediumSafe(ot.Brent(1e-3,1e-3,1e-3,5)),oppositeDirection = False, activeLS = True,
              minCoV = 0.05, maxLines=1000, batchSize=1)
LS_Update.run()
LS_result = LS_Update.getResults()
LS_Probability = LS_result['Pf']
LS_CoV = LS_result['CoV']

print("LS_Update Probability = ", LS_Probability[-1])
print("LS_Update CoV = ", LS_CoV[-1])
print("Number of line searches = ", len(LS_Probability))
print("Updated directions alpha = ", LS_result['alpha'])

# ##### compare root solvers ####################################
LS_fast = LineSampling(event,alpha, rootSolver=ot.RiskyAndFast(ot.Brent(1e-3,1e-3,1e-3,5)),oppositeDirection = False, activeLS = False,
              minCoV = 0.05, maxLines=1000, batchSize=1)
LS_fast.run()
LS_result = LS_fast.getResults()
LS_Probability = LS_result['Pf']
LS_CoV = LS_result['CoV']

print("LS_RiskyAndFast Probability = ", LS_Probability[-1])
print("LS_RiskyAndFast CoV = ", LS_CoV[-1])
print("LS_RiskyAndFast Number of line searches = ", len(LS_Probability))
print("LS_RiskyAndFast Mean of function evaluations along one line = ", np.mean(LS_result['lineFunctionCalls']))
print("LS_RiskyAndFast Total number of function evaluations along one line = ", np.sum(LS_result['lineFunctionCalls']))


LS_slow = LineSampling(event,alpha, rootSolver=ot.SafeAndSlow(ot.Brent(1e-3,1e-3,1e-3,5)),oppositeDirection = False, activeLS = False,
              minCoV = 0.05, maxLines=1000, batchSize=1)
LS_slow.run()
LS_result = LS_slow.getResults()
LS_Probability = LS_result['Pf']
LS_CoV = LS_result['CoV']

print("LS_SafeAndSlow Probability = ", LS_Probability[-1])
print("LS_SafeAndSlow CoV = ", LS_CoV[-1])
print("LS_SafeAndSlow Number of line searches = ", len(LS_Probability))
print("LS_SafeAndSlow Mean of function evaluations along one line = ", np.mean(LS_result['lineFunctionCalls']))
print("LS_SafeAndSlow Total number of function evaluations along one line = ", np.sum(LS_result['lineFunctionCalls']))

