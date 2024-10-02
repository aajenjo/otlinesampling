#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 18:23:26 2024

@author: G29933
"""
# %%
import openturns as ot
import openturns.experimental as otexp
import matplotlib.pyplot as plt
import openturns.viewer as viewer
import math as m
from scipy.integrate import quad
import numpy as np
# import openturns.experimental as otexp 

paraboloid = ot.SymbolicFunction(['u1', 'u2', 'u3','u4','u5'], ['- u5 + u1^2 + u2^2 + u3^2 + u4^2'])

b = 3.5
db = 0.02
t=0.
dim = 5
U = ot.Normal(dim)

# marginals = [u1, u2, u3]
# joint_distribution = ot.ComposedDistribution(marginals)
# joint_distribution.setDescription(['u1', 'u2', 'u3'])
# dim = joint_distribution.getDimension()

X = ot.RandomVector(U)
Y = ot.CompositeRandomVector(paraboloid, X)
E1 = ot.ThresholdEvent(Y, ot.Less(), -b)
E2 = ot.ThresholdEvent(Y, ot.Less(), -(b+db))
E3 = ot.ThresholdEvent(Y, ot.Greater(), -(b+db))
E4 = ot.IntersectionEvent([E1, E3])
# SE4 = ot.StandardEvent(E4)

# # calcul de la proba de référence par integration (u1² + u2² est une loi du khi2 à 2 ddl)
# amin = 0
# amax = 1000
# f = ot.SymbolicFunction(['t','x'],['0.5*m.exp(-t/2)/ot.Gamma(1)*m.exp(-(x-b)²/2)/m.sqrt(2*m.pi)'])
# lb = [-1000]
# ub = [ot.SymbolicFunction(['x'],['x'])]

# calcul de la proba de référence par integration (u1² + u2² est une loi du khi2 à 2 ddl)
tmin = 0
tmax = 1000

# def CDF_khi_2(t):
#     y = ot.ChiSquare(2).computeCDF(t)
#     return [y]

khi2 = ot.ChiSquare(dim-1)

def integrand(t):
    y = khi2.computeCDF(t)*m.exp(-((t+b)*(t+b))/2)/m.sqrt(2*m.pi)
    return y

# integrande = ot.PythonFunction(1,1, integrand)
# print("integrande :", integrande([0]))

def integrand2(t):
    y = ot.ChiSquare(dim-1).computeCDF(t)*m.exp(-((t+b+db)*(t+b+db))/2)/m.sqrt(2*m.pi)
    return y
# CDF_khi2 = ot.PythonFunction(1,1, CDF_khi_2)

# -------------- integration with OpenTURNS -----------
# I2 = ot.IteratedQuadrature().integrate(integrande, tmin, tmax)
# print(I2)

# -------------- intégration with scipy -------------
erabs = 1.e-20

f1 = lambda t: integrand(t)
y1, err1 = quad(f1, 0, np.inf,epsabs=erabs)
print("-------- Numerical integration by quad: Pf = ", y1, "error = ",err1)
print()

f2 = lambda t: integrand2(t)
y2, err2 = quad(f2, 0, np.inf,epsabs=erabs)
print("-------- Numerical integration by quad: Pf = ", y2, "error = ",err2)
print()


print("For beta = ", b, "and dbeta = ", db)
print()
print("Reference probability by numerical integration - Probability of the narrow strip : ", y1 - y2)
# print(CDF_khi_2(1.39))

# f = ot.PythonFunction(1,1,integrand)

# print(f([0]))
# I2 = ot.IteratedQuadrature().integrate(f, tmin, tmax)
# print('Proba par intégration numérique: ',I2)

# FORM-SORM probabilities (cumulative event and narrow strip)
# Many other optimization algorithms (i.e., solvers) available.
# See: https://openturns.github.io/openturns/latest/user_manual/optimization.html
# AbdoRackwitz is considered as the most efficient solver for FORM evaluations
 
solver = ot.AbdoRackwitz()
starting_point = U.getMean()
# See FORM API: https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.FORM.html
FORM_algo = ot.FORM(solver, E1, starting_point)
FORM_algo.run()
FORM_result = FORM_algo.getResult()
beta = FORM_result.getHasoferReliabilityIndex()
print('Beta_FORM =', beta)
importance_factors = FORM_result.getImportanceFactors()
u_star = FORM_result.getStandardSpaceDesignPoint()
FORM_pf = FORM_result.getEventProbability()
print(f"FORM pf : {FORM_pf:.2e}")
graph = FORM_result.drawImportanceFactors()
view = viewer.View(graph)


# %%
# See SORM API: https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.SORM.html
SORM_algo = ot.SORM(solver, E1, starting_point)
SORM_algo.run()# %%
standard_event = ot.StandardEvent(E1)
SORM_result = SORM_algo.getResult()
beta = SORM_result.getHasoferReliabilityIndex()
importance_factors = SORM_result.getImportanceFactors()
u_star = SORM_result.getStandardSpaceDesignPoint()
SORM_pf_Brei = SORM_result.getEventProbabilityBreitung()
print(f"SORM pf Breitung: {SORM_pf_Brei:.4e}")
SORM_pf_HB = SORM_result.getEventProbabilityHohenbichler()
print(f"SORM pf Hohenbichler: {SORM_pf_HB:.4e}")
SORM_pf_Tvedt = SORM_result.getEventProbabilityTvedt()
print(f"SORM pf Tvedt: {SORM_pf_Tvedt:.4e}")
print()

print("Pour beta = ", b+db)
SORM_algo = ot.SORM(solver, E2, starting_point)
SORM_algo.run()# %%
standard_event = ot.StandardEvent(E2)
SORM_result = SORM_algo.getResult()
beta = SORM_result.getHasoferReliabilityIndex()
importance_factors = SORM_result.getImportanceFactors()
u_star = SORM_result.getStandardSpaceDesignPoint()
SORM_pf_Brei_2 = SORM_result.getEventProbabilityBreitung()
print(f"SORM pf Breitung: {SORM_pf_Brei_2:.4e}")
SORM_pf_HB_2 = SORM_result.getEventProbabilityHohenbichler()
print(f"SORM pf Hohenbichler: {SORM_pf_HB_2:.4e}")
SORM_pf_Tvedt_2 = SORM_result.getEventProbabilityTvedt()
print(f"SORM pf Tvedt: {SORM_pf_Tvedt_2:.4e}")
print()
print(f"SORM-Breitung - Probability of the narrow strip: {SORM_pf_Brei - SORM_pf_Brei_2:.4e}")
print(f"SORM-Hohenbichler - Probability of the narrow strip: {SORM_pf_HB - SORM_pf_HB_2:.4e}")
print()
print(f"SORM-Tvedt - Probability of the narrow strip: {SORM_pf_Tvedt - SORM_pf_Tvedt_2:.4e}")
print()
print(f"SORM-Tvedt - Relative error of the probability of the narrow strip: {(SORM_pf_Tvedt - SORM_pf_Tvedt_2) / y1 - y2:.4e}")
# cv for simulation methods

cv = 0.1

## Monte Carlo analysis using OpenTURNS (for any probability and threshold)

# mc_experiment = ot.MonteCarloExperiment()
# algoMC = ot.ProbabilitySimulationAlgorithm(E1, mc_experiment)
# algoMC.setMaximumCoefficientOfVariation(cv)
# algoMC.setMaximumOuterSampling(int(1e6))

# in_num_LSF_calls = paraboloid.getEvaluationCallsNumber()

# algoMC.run()
# result = algoMC.getResult()
# Pf = result.getProbabilityEstimate()
# num_LSF_calls = paraboloid.getEvaluationCallsNumber() - in_num_LSF_calls
# cv_real = result.getCoefficientOfVariation()
# print("Monte Carlo Probability Pf = ", Pf)
# print("MC Accuracy (CV) = ", cv_real)
# print("Confidence Interval (0.95) = [", Pf -result.getConfidenceLength()/2,", ", Pf + result.getConfidenceLength()/2,"]")
# print("Verified Coefficient of variation (= 1/sqrt(N*Pf)) =", 1/(num_LSF_calls * Pf)**0.5)
# print("Number of LSF calls = ", num_LSF_calls)
# print()

# %%
palette = ot.Drawable.BuildDefaultPalette(10)
# graph = algoMC.drawProbabilityConvergence(0.95)
# graph.setColors([palette[3], palette[1], palette[1]])
# view = viewer.View(graph)

# %% [markdown]
# ### FORM / SORM

## FORM - Importance sampling
ustar = ot.Point([0,0,0,0,b+db/2])
print(ustar)
standard_importance_density = ot.Normal(u_star, [1.0] * dim) # Standard Gaussian centered around the medium of the strip
is_experiment_std = ot.ImportanceSamplingExperiment(standard_importance_density)

Sigma = ot.CovarianceMatrix(dim)
Sigma[dim-1,dim-1] = 11*db
print(Sigma)
refined_importance_density = ot.Normal(u_star, Sigma) # Standard Gaussian centered around the medium of the strip
is_experiment_ref = ot.ImportanceSamplingExperiment(refined_importance_density)

# FORM-Importance Sampling standard
algo = ot.ProbabilitySimulationAlgorithm(E4, is_experiment_std) # The user can change the experiment with one defined above 
# Algorithm stopping criteria
algo.setMaximumOuterSampling(int(1e6))
algo.setBlockSize(1)
algo.setMaximumCoefficientOfVariation(cv)
in_num_LSF_calls = paraboloid.getEvaluationCallsNumber()

# Perform the simulation
algo.run()

num_LSF_calls = paraboloid.getEvaluationCallsNumber() - in_num_LSF_calls
FIS_results = algo.getResult()
FIS_pf = FIS_results.getProbabilityEstimate()
print(f'FORM-IS failure probability = {FIS_pf:.2e}')
print("FORM-IS accuracy (CV) = ", FIS_results.getCoefficientOfVariation())
print("Confidence Interval (0.95) = [", FIS_pf-FIS_results.getConfidenceLength()/2,", ", FIS_pf + FIS_results.getConfidenceLength()/2,"]")
print("Number of LSF calls = ", num_LSF_calls)
print()

# FORM-Importance Sampling refined
algo = ot.ProbabilitySimulationAlgorithm(E4, is_experiment_ref) # The user can change the experiment with one defined above 
# Algorithm stopping criteria
algo.setMaximumOuterSampling(int(1e6))
algo.setBlockSize(1)
algo.setMaximumCoefficientOfVariation(cv)
in_num_LSF_calls = paraboloid.getEvaluationCallsNumber()

# Perform the simulation
algo.run()

num_LSF_calls = paraboloid.getEvaluationCallsNumber() - in_num_LSF_calls
FIS_results = algo.getResult()
FIS_pf = FIS_results.getProbabilityEstimate()
print(f'FORM-IS failure probability = {FIS_pf:.2e}')
print("FORM-IS accuracy (CV) = ", FIS_results.getCoefficientOfVariation())
print("Confidence Interval (0.95) = [", FIS_pf-FIS_results.getConfidenceLength()/2,", ", FIS_pf + FIS_results.getConfidenceLength()/2,"]")
print("Number of LSF calls = ", num_LSF_calls)
print()

# ### Subset sampling

# %%
SS_algo = ot.SubsetSampling(E4)
SS_algo.setMaximumOuterSampling(int(2.5e4))
SS_algo.setBlockSize(1)
SS_algo.setConditionalProbability(0.1)
in_num_LSF_calls = paraboloid.getEvaluationCallsNumber()

# Perform the simulation
SS_algo.run()

num_LSF_calls = paraboloid.getEvaluationCallsNumber() - in_num_LSF_calls

SS_results = SS_algo.getResult()
levels = SS_algo.getThresholdPerStep()
SS_pf = SS_results.getProbabilityEstimate()
print(f'Subset sampling failure probability = {SS_pf:.2e}')
print("Number of steps for Subset Simulation: ",SS_algo.getStepsNumber())
print("SS Accuracy (CV) : ", SS_results.getCoefficientOfVariation())
print("Confidence length (0.95): ", SS_results.getConfidenceLength())
print("Confidence Interval (0.95) = [", SS_pf-SS_results.getConfidenceLength()/2,", ", SS_pf+SS_results.getConfidenceLength()/2,"]")
print("Levels of g = ", SS_algo.getThresholdPerStep())
print("Number of LSF calls = ", num_LSF_calls)

# %%
graph = SS_algo.drawProbabilityConvergence(0.95)
graph.setColors([palette[3], palette[1], palette[1]])
view = viewer.View(graph)

# %%

#  Directional Simulation

# %% %%
# Root finding algorithm.

# %%
solver = ot.Brent()
rootStrategy = ot.SafeAndSlow(solver)

# %%
# Direction sampling algorithm.
# With the advanced root finding algorithm implemented by Antoine
# directional sampling should also perform better.

# %%
samplingStrategy = ot.OrthogonalDirection()

# %%
# Create a simulation algorithm.

# %%
algo = ot.DirectionalSampling(E4, rootStrategy, samplingStrategy)
algo.setMaximumCoefficientOfVariation(cv)
algo.setMaximumOuterSampling(50000)
algo.setConvergenceStrategy(ot.Full())
in_num_LSF_calls = paraboloid.getEvaluationCallsNumber()

# Perform the simulation
algo.run()

num_LSF_calls = paraboloid.getEvaluationCallsNumber() - in_num_LSF_calls
# %%
# Retrieve results.

# %%
result = algo.getResult()
Pf = result.getProbabilityEstimate()
print()
print("Directional Simulation - Pf = ", Pf)
print('DS Accuracy (CV) : ', result.getCoefficientOfVariation())
print("Confidence Interval (0.95) = [", Pf -result.getConfidenceLength()/2,", ", Pf + result.getConfidenceLength()/2,"]")
print("Number of LSF calls = ", num_LSF_calls)

# %%
# We can observe the convergence history with the `drawProbabilityConvergence`
# method.
graph = algo.drawProbabilityConvergence()
graph.setLogScale(ot.GraphImplementation.LOGX)
view = viewer.View(graph)

# %%
# Evaluate the probability with the NAIS technique
# ------------------------------------------------

# %%
quantileLevel = 0.1
algo = ot.NAIS(E4.asComposedEvent(), quantileLevel) # asComposedEvent() necessary due to issue #2773

# %%
# In order to get all the inputs and outputs that realize the event, you have to mention it now:

# %%
# algo.setKeepSample(True)
algo.setMaximumCoefficientOfVariation(cv)
algo.setMaximumOuterSampling(50)
in_num_LSF_calls = paraboloid.getEvaluationCallsNumber()

# We are forced to set a low maximum outer sampling size because NAIS is unable
# to reach the desired coefficient of variation in reasonable time.

# %%
# Now you can run the algorithm.

# %%
algo.run()
num_LSF_calls = paraboloid.getEvaluationCallsNumber() - in_num_LSF_calls
result = algo.getResult()
proba = result.getProbabilityEstimate()
print()
print("Proba NAIS = ", proba)
print("Current coefficient of variation = ", result.getCoefficientOfVariation())
print("Number of LSF calls = ", num_LSF_calls)

# %%

# -------------------------------------------  IS cross entropy ---------------
standardSpaceIS = otexp.StandardSpaceCrossEntropyImportanceSampling(E4.asComposedEvent(), 0.35) # asComposedEvent() necessary due to issue #2773

# %%
# The sample size at each iteration can be changed by the following accessor:

# %%
standardSpaceIS.setMaximumOuterSampling(200)
standardSpaceIS.setMaximumCoefficientOfVariation(cv)
in_num_LSF_calls = paraboloid.getEvaluationCallsNumber()
# %%
# Now we can run the algorithm and get the results.

# %%
standardSpaceIS.run()
num_LSF_calls = paraboloid.getEvaluationCallsNumber() - in_num_LSF_calls
standardSpaceISResult = standardSpaceIS.getResult()
proba = standardSpaceISResult.getProbabilityEstimate()
print("Proba Standard Space Cross Entropy IS = ", proba)
print(
    "Current coefficient of variation = ",
    standardSpaceISResult.getCoefficientOfVariation(),
)
print("Number of LSF calls = ", num_LSF_calls)

# This does not work !

# %%

# Line Sampling
import sys
sys.path.append("/home/c65779/Documents/Projet_VIGIE/otlinesampling/otlinesampling")
from LineSampling import LineSampling

# %%

alpha = [0.0, 0.0, 0.0, 0.0, 1.0]
ls = LineSampling(E4, alpha, maxLines=2000)

# %%
ls.run()

# %%

res = ls.getResults()
Pf = res['Pf']
print(Pf[-1])