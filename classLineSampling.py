#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 08:23:19 2023

@author: c65779
"""
import openturns as ot
import scipy.optimize as opt
from copy import deepcopy
import numpy as np
### Problème avec DomainEvent : 'DomainEvent' object has no attribute 'getLevel' + 'getOperator'. Pareil pour DomainEvent.getDomain() 

class LineSampling():
    """
    Description.
    """
    def __init__(self,event,alpha,rootSolver=ot.MediumSafe(ot.Brent(1e-5,1e-5,1e-8,10)),oppositeDirection = False, activeLS = False,
                 minCoV = 0.05, maxLines=1000, batchSize=1, seed=1234):
        self._isIntersection = True
        self._eventCollection = self._checkEvent(event)
        self._dim = self._eventCollection[0].getAntecedent().getDimension()
        self._alpha = self._checkAlpha(alpha)
        self._rootSolver = self._checkSolver(rootSolver)
        self._activeLS = self._checkActiveLS(activeLS)
        self._oppositeDirection = self._checkOppositeDirection(oppositeDirection)
        self._minCoV = self._checkMinCoV(minCoV)
        self._maximumLines = self._checkMaxLines(maxLines)
        self._standardFunctions = self._getFunctionInStandardSpace()
        self._roots = []
        self._missedRoots = []
        self._pf = []
        self._thresholds = [i.getThreshold() for i in self._eventCollection]
        self._totalFunctionCalls = []
        self._CoV = [1000]
        self._batchSize = batchSize
        self._seed = seed

    def _checkDomainEvent(self,domainEvent):
        # Useful from OT 1.21 onwards with DomainEvents
        # With 1.20, only ThresholdEvent-based events can be used.
        # try:
        #     domainEvent.getImplementation()
        # except AttributeError:
        #     pass
        event_repr = repr(domainEvent).split(' ')[0]+repr(domainEvent).split(' ')[1]
        if "ThresholdEvent" in event_repr:
            return deepcopy(domainEvent)
        elif "DomainEvent" in event_repr:
            representation = repr(domainEvent.getDomain()).split(' ')[0]
            print(representation)
            if representation != "class=LevelSet":
                raise TypeError("Domain not LevelSet, but " + representation)
            thresholdEvent = ot.ThresholdEvent(domainEvent.getAntecedent(),
                                               domainEvent.getDomain().getOperator(),
                                               domainEvent.getDomain().getLevel())
            return thresholdEvent
        else:
            raise TypeError("Wrong event: " + str(event_repr))

    def _checkUnionIntersectionEvent(self, system_event):
        collection = system_event.getEventCollection()
        new_collection = []
        for event in collection:
            new_collection.append(self._checkDomainEvent(event))
        return new_collection
            
    
    def _checkEvent(self,event):
        msg = "Wrong event type"
        class_name = event.getClassName()
        if class_name not in ['ThresholdEvent','DomainEvent','UnionEvent','IntersectionEvent']:
            raise TypeError(msg)
        if class_name in ['ThresholdEvent','DomainEvent']:
            return [self._checkDomainEvent(event)]
        else:
            if class_name == 'UnionEvent':
                self._isIntersection = False
            return self._checkUnionIntersectionEvent(event)
        
    def _checkAlpha(self,alpha):
        if len(alpha) != self._dim:
            raise ValueError("The dimension of alpha should be " +str(self._dim))
        else:
            return [alpha/np.linalg.norm(alpha)]
        
    def _checkSolver(self,rootSolver):
        if rootSolver.getClassName() not in ['MediumSafe','RiskyAndFast','SafeAndSlow']:
            raise TypeError("The solver is not one of the required OpenTURNS classes (ot.MediumSafe(),ot.RiskyAndFast() or ot.SafeAndSlow())")
        return rootSolver
        
    
    def _checkActiveLS(self,activeLS):
        if type(activeLS) is not bool:
            raise TypeError("activeLS should be True or False")
        else:
            return activeLS
        
    def _checkOppositeDirection(self,oppositeDirection):
        if type(oppositeDirection) is not bool:
            raise TypeError("activeLS should be True or False")
        else:
            return oppositeDirection
        
    def _checkMinCoV(self,minCoV):
        if type(minCoV) is not float or minCoV<=0:
            raise ValueError("minCoV should be a float value strictly greater than 0")
        else:
            return minCoV
        
    def _checkMaxLines(self,maxLines):
        if type(maxLines) is not int or maxLines<=0:
            raise ValueError("maxLines should be an integer value strictly greater than 0")
        else:
            return maxLines
        
    def _getFunctionInStandardSpace(self):
        """Returns a list of functions in the standard space."""
        standard_functions = []
        for i in self._eventCollection:
            function = i.getFunction()
            tr_iso_inv = i.getAntecedent().getDistribution().getInverseIsoProbabilisticTransformation()
            standard_functions.append(ot.ComposedFunction(function, tr_iso_inv))
        return standard_functions
        
    def _evaluate_line_function(self, c):
        """Compute the function along a specific line."""
        # les racines au dela de +-10 ne nous intéressent pas car elles n'auront pas d'impact sur la Pf
        inputs = [self._currentLine[i]+self._sign*c[0]*self._alpha[-1][i] for i in range(len(self._alpha[-1]))]
        res = self._currentStandFunction(tuple([i for i in inputs]))
        return res
    
    def getSolver(self):
        return self._rootSolver
    
    def setSolver(self,rootSolver):
        self._rootSolver = self._checkSolver(rootSolver)
            

    def _sampleLines(self,N):
        """Sample the lines."""
        ot.RandomGenerator.SetSeed(self._seed)
        U = ot.Normal(self._dim).getSample(N)
        U_prod_scalaire = [sum([i[j]*self._alpha[-1][j] for j in range(len(self._alpha[-1]))]) for i in U]
        U_alpha = [[i*self._alpha[-1][j] for j in range(self._dim)] for i in U_prod_scalaire]
        U_alpha_ortho = [[U[i][j]-U_alpha[i][j] for j in range(self._dim)] for i in range(len(U))]
        ''' Ordonner les points '''
        # if self._activeLS:
        #     ...
        return U_alpha_ortho
        
    def getMaximumLines(self):
        return self._maximumLines
        
    def setMaximumLines(self,maxLines):
        self._maxLines = self._checkMaxLines(maxLines)
        
    def getMinCov(self):
        return self._minCoV
        
    def setMinCov(self,minCoV):
        self._minCoV = self._checkMinCoV(minCoV)
        
    def getOppositeDirection(self):
        return self._oppositeDirection
        
    def setOppositeDirection(self,oppositeDirection):
        self._oppositeDirection = self._checkOppositeDirection(oppositeDirection)
        
    def _evaluatePf(self,roots,originValue,event):
        if None in roots or len(roots)==0:
            if event.getOperator()(originValue,event.getThreshold()):
                Pf = 1.
            else:
                Pf = 0.
        else:
            roots_pos = np.sort([i for i in roots if i>=0])
            roots_neg = np.sort([i for i in roots if i<0])
            if event.getOperator()(originValue,event.getThreshold()):
                Pf_pos = 1-sum([(-1)**-k*ot.Normal().computeCDF(-roots_pos[k]) for k in range(len(roots_pos))])
                Pf_neg = 1-sum([(-1)**k*ot.Normal().computeCDF(-roots_neg[-1-k]) for k in range(len(roots_neg))])
            else:
                Pf_pos = sum([(-1)**-k*ot.Normal().computeCDF(-abs(roots_pos[k])) for k in range(len(roots_pos))])
                Pf_neg = sum([(-1)**k*ot.Normal().computeCDF(-abs(roots_neg[-1-k])) for k in range(len(roots_neg))])
            Pf = Pf_pos+Pf_neg
        return Pf
    
        
    def _find_roots(self,line_sample,index):
        """Compute the roots along a specific line. Maybe compute proba at the same time."""
        # rootStrat = ot.RootStrategy(self._rootStrategy)
        self._currentLine = line_sample
        self._currentStandFunction = self._standardFunctions[index]
        missedRoot = False
        self._sign=1.
        pythonFunc = ot.PythonFunction(1,1,self._evaluate_line_function)
        pythonFunc = ot.MemoizeFunction(pythonFunc)
        try:
            roots = list(self._rootSolver.solve(pythonFunc,self._thresholds[index]))
            functionCalls = len(pythonFunc.getInputHistory())
            if self._oppositeDirection:
                self._sign=-1.
                pythonFunc_opp = ot.PythonFunction(1,1,self._evaluate_line_function)
                pythonFunc_opp = ot.MemoizeFunction(pythonFunc_opp)
                roots_opposite = list(self._rootSolver.solve(pythonFunc_opp,self._thresholds[index]))
                roots_opposite = [-i for i in roots_opposite]
                roots+=roots_opposite
                functionCalls += len(pythonFunc_opp.getInputHistory())
            originValue = self._rootSolver.getOriginValue()
            if self._activeLS:
                uCandidate = [self._currentLine[i]+roots[0]*self._alpha[-1][i] for i in range(len(self._alpha[-1]))]
                if np.linalg.norm(uCandidate)<np.linalg.norm(self._ustar[-1]):
                    self._ustar.append(uCandidate)
                    self._alpha.append([i/np.linalg.norm(self._ustar[-1]) for i in self._ustar[-1]])
        except:
            missedRoot = True
            roots=[None]
            functionCalls = len(pythonFunc.getInputHistory())
            originValue = 1.
        return [roots,originValue,functionCalls,missedRoot]
        
    def run(self):
        if self._activeLS:
            self._activeLS = False
            roots = self._find_roots([0.]*self._dim,0)[0][0]
            self._ustar = [[roots*self._alpha[-1][i] for i in range(len(self._alpha[-1]))]]
            self._activeLS = True
        self._linePf = []
        length_alpha = len(self._alpha)
        """Sample lines and find corresponding roots."""
        # my_func = ot.PythonFunction(self._dim,1,self._find_roots,n_cpus=self._batchSize)
        lines = self._sampleLines(self._maximumLines)
        nb_evaluation = 0
        while nb_evaluation<self._maximumLines and self._CoV[-1] > self._minCoV:
            res = [self._find_roots(lines[nb_evaluation],i) for i in range(len(self._eventCollection))]
            self._roots.append([i[0] for i in res])
            self._totalFunctionCalls.append([i[2] for i in res])
            self._missedRoots.append([i[3] for i in res])
            nb_evaluation += self._batchSize
            self._linePf.append([self._evaluatePf(list(self._roots[-1][i]),res[i][1],self._eventCollection[i]) for i in range(len(self._eventCollection))])
            self._pf.append([np.mean(np.array(self._linePf)[:,i]) for i in range(len(self._eventCollection))])
            if len(self._eventCollection)==1:
                if len(self._pf)>1:
                    Var = 1/(len(self._pf)*(len(self._pf)-1))*sum([(i[0]-self._pf[-1][0])**2 for i in self._linePf])
                    self._CoV.append(np.sqrt(Var)/self._pf[-1])
            if length_alpha<len(self._alpha):
                lines = self._sampleLines(self._maximumLines)
                length_alpha = len(self._alpha)
        
    def getResults(self):
        return self._linePf,self._pf,self._CoV[1:],self._missedRoots,self._roots,self._totalFunctionCalls
    
