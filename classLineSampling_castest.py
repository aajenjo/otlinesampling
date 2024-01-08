#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 11:01:15 2024

@author: c65779
"""

import openturns as ot
import numpy as np
import math
from openturns.usecases import stressed_beam,flood_model,cantilever_beam
from classLineSampling import LineSampling

################## NAIS 4 branch #######################   
X = ot.RandomVector(ot.Normal(2))
def fourBranch(x):
    x1 = x[0]
    x2 = x[1]

    g1 = 5 + 0.1 * (x1 - x2) ** 2 - (x1 + x2) / math.sqrt(2)
    g2 = 5 + 0.1 * (x1 - x2) ** 2 + (x1 + x2) / math.sqrt(2)
    g3 = (x1 - x2) + 9 / math.sqrt(2)
    g4 = (x2 - x1) + 9 / math.sqrt(2)

    return [min((g1, g2, g3, g4))]


g = ot.PythonFunction(2, 1, fourBranch)
Y = ot.CompositeRandomVector(g, X)
threshold = 1
event = ot.ThresholdEvent(Y, ot.Less(), threshold)
optimAlgo = ot.Cobyla()
optimAlgo.setMaximumEvaluationNumber(1000)
optimAlgo.setMaximumAbsoluteError(1.0e-10)
optimAlgo.setMaximumRelativeError(1.0e-10)
optimAlgo.setMaximumResidualError(1.0e-10)
optimAlgo.setMaximumConstraintError(1.0e-10)
algo = ot.FORM(optimAlgo, event, X.getMean())
algo.run()
result = algo.getResult()
alpha=result.getStandardSpaceDesignPoint()
experiment = ot.MonteCarloExperiment()

algo = ot.ProbabilitySimulationAlgorithm(event, experiment)

algo.setMaximumOuterSampling(1000000)

algo.setBlockSize(4)

algo.setMaximumCoefficientOfVariation(0.01)

algo.run()

print('Probability estimate=%.6f' % algo.getResult().getProbabilityEstimate())
#########################################   

################## stressed_beam MC #######################   

# sm = stressed_beam.AxialStressedBeam()
# distribution = sm.distribution
# model = sm.model
# vect = ot.RandomVector(distribution)
# G = ot.CompositeRandomVector(model, vect)
# event = ot.ThresholdEvent(G, ot.Less(), -500000)
# optimAlgo = ot.Cobyla()
# optimAlgo.setMaximumEvaluationNumber(1000)
# optimAlgo.setMaximumAbsoluteError(1.0e-10)
# optimAlgo.setMaximumRelativeError(1.0e-10)
# optimAlgo.setMaximumResidualError(1.0e-10)
# optimAlgo.setMaximumConstraintError(1.0e-10)
# algo = ot.FORM(optimAlgo, event, distribution.getMean())
# algo.run()
# result = algo.getResult()
# alpha=result.getStandardSpaceDesignPoint()
# experiment = ot.MonteCarloExperiment()
# algo = ot.ProbabilitySimulationAlgorithm(event, experiment)
# algo.setMaximumCoefficientOfVariation(0.05)
# algo.setMaximumOuterSampling(int(1e6))
# algo.run()
# result = algo.getResult()
# probability = result.getProbabilityEstimate()
# print("Pf=", probability)
#########################################   


################## Flooding MC #######################   

# fm = flood_model.FloodModel()
# distribution = fm.distribution
# model = fm.model
# vect = ot.RandomVector(distribution)
# G = ot.CompositeRandomVector(model, vect)
# event = ot.ThresholdEvent(G, ot.Greater(), -5)
# event.setName("overflow")
# optimAlgo = ot.Cobyla()
# optimAlgo.setMaximumEvaluationNumber(1000)
# optimAlgo.setMaximumAbsoluteError(1.0e-10)
# optimAlgo.setMaximumRelativeError(1.0e-10)
# optimAlgo.setMaximumResidualError(1.0e-10)
# optimAlgo.setMaximumConstraintError(1.0e-10)
# startingPoint = distribution.getMean()
# algo = ot.FORM(optimAlgo, event, startingPoint)
# algo.run()
# result = algo.getResult()
# alpha=result.getStandardSpaceDesignPoint()
# experiment = ot.MonteCarloExperiment()
# algo = ot.ProbabilitySimulationAlgorithm(event, experiment)
# algo.setMaximumCoefficientOfVariation(0.05)
# algo.setMaximumOuterSampling(int(1e6))
# algo.run()
# result = algo.getResult()
# probability = result.getProbabilityEstimate()
# print("Pf=", probability)
#########################################

################## cantilever_beam IS #######################   
cb = cantilever_beam.CantileverBeam()
distribution = cb.distribution
model = cb.model
vect = ot.RandomVector(distribution)
G = ot.CompositeRandomVector(model, vect)
event = ot.ThresholdEvent(G, ot.Greater(), 0.30)
optimAlgo = ot.Cobyla()
optimAlgo.setMaximumEvaluationNumber(1000)
optimAlgo.setMaximumAbsoluteError(1.0e-10)
optimAlgo.setMaximumRelativeError(1.0e-10)
optimAlgo.setMaximumResidualError(1.0e-10)
optimAlgo.setMaximumConstraintError(1.0e-10)
algo = ot.FORM(optimAlgo, event, distribution.getMean())
algo.run()
result = algo.getResult()
alpha=result.getStandardSpaceDesignPoint()
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
probability = result.getProbabilityEstimate()
print("Probability = ", probability)




# def g1(x):
#     x1 = x[0]
#     x2 = x[1]
#     g = 5 + 0.1 * (x1 - x2) ** 2 - (x1 + x2) / math.sqrt(2)
#     return [g]
# def g2(x):
#     x1 = x[0]
#     x2 = x[1]
#     g = 5 + 0.1 * (x1 - x2) ** 2 + (x1 + x2) / math.sqrt(2)
#     return [g]
# def g3(x):
#     x1 = x[0]
#     x2 = x[1]
#     g = (x1 - x2) + 9 / math.sqrt(2)
#     return [g]
# def g4(x):
#     x1 = x[0]
#     x2 = x[1]
#     g = (x2 - x1) + 9 / math.sqrt(2)
#     return [g]
# dim = 2
# X = ot.RandomVector(ot.Normal(dim))
# Y1 = ot.CompositeRandomVector(ot.PythonFunction(2,1,g1), X)
# Y2 = ot.CompositeRandomVector(ot.PythonFunction(2,1,g2), X)
# Y3 = ot.CompositeRandomVector(ot.PythonFunction(2,1,g3), X)
# Y4 = ot.CompositeRandomVector(ot.PythonFunction(2,1,g4), X)
# e1 = ot.ThresholdEvent(Y1, ot.Less(), 1)
# e2 = ot.ThresholdEvent(Y2, ot.Less(), 1)
# e3 = ot.ThresholdEvent(Y3, ot.Less(), 1)
# e4 = ot.ThresholdEvent(Y4, ot.Less(), 1)

# event = ot.IntersectionEvent([e1, e2,e3, e4])
# optimAlgo = ot.AbdoRackwitz()
# optimAlgo.setMaximumEvaluationNumber(1000)
# optimAlgo.setMaximumAbsoluteError(1.0e-10)
# optimAlgo.setMaximumRelativeError(1.0e-10)
# optimAlgo.setMaximumResidualError(1.0e-10)
# optimAlgo.setMaximumConstraintError(1.0e-10)
# algo = ot.FORM(optimAlgo, e1, X.getMean())
# algo.run()
# result = algo.getResult()
# alpha=result.getStandardSpaceDesignPoint()