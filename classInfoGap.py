#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 08:23:19 2023

@author: c65779
"""
import openturns as ot
import scipy.optimize as opt
from classRandomSets import RandomSets
### Problème avec DomainEvent : 'DomainEvent' object has no attribute 'getLevel' + 'getOperator'. Pareil pour DomainEvent.getDomain() 

class InfoGap():
    """
    Description.
    """
    def __init__(self,function,uncertainParameters,nominalPoint,knownParametersValues,h,worstPointSearch='vertex',convexSetType='interval'):
        self._paramFunction, self._uncertainParameters, self._nominalPoint, self._knownParametersValues = self._checkfunction(function,uncertainParameters,knownParametersValues)
        self._worstPointSearch = self._checkWorstPoint(worstPointSearch)
        
    def _checkfunction(self,function,uncertainparameters,nominalpoint,knownparametersvalues):
        if type(function) not in [ot.func.Function,ot.func.SymbolicFunction]:
            raise TypeError("Function is not OpenTURNS Python function or OpenTURNS Symbolic function, but " + str(type(function)))
        if type(uncertainparameters) != list or type(knownparametersvalues) != list:
            raise TypeError("uncertainParameters or knownParametersValues is not a list")
        if len(uncertainparameters)+len(knownparametersvalues)!=len(function.getInputDescription()):
            raise TypeError("The number of uncertain parameters and the number of known parameters is do not add up to the number of inputs expected by the given function")
        if len(uncertainparameters)!=len(nominalpoint):
            raise TypeError("The length of uncertainParameters has to be equal to the length of nominalPoint")
        for i in uncertainparameters:
            if type(i) != int or i<0:
                raise TypeError("uncertainParameters contains the component" +str(i) +"which is either not an integer or negative")
        if len(knownparametersvalues)>0:
            return ot.ParametricFunction([i for i in range(len(function.getInputDescription())) if i not in uncertainparameters],
                                         knownparametersvalues,function),uncertainparameters,nominalpoint,knownparametersvalues
        else:
            return function,uncertainparameters,nominalpoint,knownparametersvalues
        
    def _checkWorstPoint(self,worstpointsearch):
        if worstpointsearch not in ['given','vertex','optimization']:
            raise TypeError("worstPointSearch is neither of given, vertex or optimization, but " + str(worstpointsearch))
        return worstpointsearch
    
    def setConvexSet(self,params=None):
        if not params:
            
            
        return None
    
    def setOptimizationAlgorithm(self,params=None):
        return None
    
    def computeRobustnessCurve(self):
        return None
    
    
        