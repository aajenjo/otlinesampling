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