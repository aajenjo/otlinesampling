#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 11:01:15 2024

@author: c65779
"""

import openturns as ot
from openturns.usecases import stressed_beam
import sys
sys.path.append("/home/c65779/Documents/Projet_VIGIE/otlinesampling/otlinesampling")
from LineSampling import LineSampling


################## stressed_beam MC #######################   

sm = stressed_beam.AxialStressedBeam()
limitStateFunction = sm.model
R_dist = sm.distribution_R
F_dist = sm.distribution_F
myDistribution = sm.distribution
inputRandomVector = ot.RandomVector(myDistribution)
outputRandomVector = ot.CompositeRandomVector(limitStateFunction, inputRandomVector)
myEvent = ot.ThresholdEvent(outputRandomVector, ot.Less(), -0)
myCobyla = ot.Cobyla()
algoFORM = ot.FORM(myCobyla, myEvent, myDistribution.getMean())
algoFORM.run()
resultFORM = algoFORM.getResult()
alpha=resultFORM.getStandardSpaceDesignPoint()

standardSpaceDesignPoint = resultFORM.getStandardSpaceDesignPoint()
dimension = myDistribution.getDimension()
myImportance = ot.Normal(dimension)
myImportance.setMu(standardSpaceDesignPoint)
experiment = ot.ImportanceSamplingExperiment(myImportance)
standardEvent = ot.StandardEvent(myEvent)
algo = ot.ProbabilitySimulationAlgorithm(standardEvent, experiment)
algo.setMaximumCoefficientOfVariation(0.01)
algo.setMaximumOuterSampling(100000)
algo.run()
result = algo.getResult()
probabilityFORMIS = result.getProbabilityEstimate()
print("IS Probability = ", probabilityFORMIS)

#########################################

##### Line Sampling ####################################
LS = LineSampling(myEvent,alpha, rootSolver=ot.MediumSafe(ot.Brent(1e-3,1e-3,1e-3,5)),oppositeDirection = False, activeLS = True,
             minCoV = 0.01, maxLines=1000, batchSize=1, fixedSeed=True)
LS.run()
LS_result = LS.getResults()
LS_Probability = LS_result['Pf']
LS_CoV = LS_result['CoV']

print("LS Probability = ", LS_Probability[-1])
print("LS CoV = ", LS_CoV[-1])
print("Number of line searches = ", len(LS_Probability))