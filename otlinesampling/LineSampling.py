#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 08:23:19 2023

@author: c65779
"""
import openturns as ot
from copy import deepcopy
import numpy as np
### Problème avec DomainEvent : 'DomainEvent' object has no attribute 'getLevel' + 'getOperator'. Pareil pour DomainEvent.getDomain() 

class LineSampling():
    """
    Estimates un failure probability using the line sampling technique.

    Parameters
    ----------
    event : ot.ThresholdEvent or ot.DomainEvent or ot.UniondEvent or ot.IntersectionEvent
        Input dataset to be plotted. Must be a pandas DataFrame object.
        A preprocessing removing every missing data is applied.

    Example
    --------
    >>> import openturns as ot
    >>> 

    >>> 
    >>> 
    >>> 
    """
    def __init__(
            self,
            event,
            alpha,
            rootSolver=ot.MediumSafe(ot.Brent(1e-3,1e-3,1e-3,5),8,1),
            oppositeDirection = False,
            activeLS = True,
            minCoV = 0.05,
            maxLines=1000,
            batchSize=1,
            fixedSeed=True,
    ):
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
        self._fixedSeed = fixedSeed
        self._seed = self.setSeed()
        self._solvedLines = 0
        self._linePf = []
        self._rootPoints = []
        

    def _checkDomainEvent(self,domainEvent):
        """
        Checks the validity of the given event for the OpenTURNS classes DomainEvent and ThresholdEvent.

        Parameters
        ----------
        domainEvent : OpenTURNS class DomainEvent
            The associated domain must either be a ThresholdEvent or a LevelSet.

        Returns
        -------
        thresholdEvent : Openturns class ThresholdEvent
        """
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
        """
        Checks the validity of the given event for the OpenTURNS classes UnionEvent and IntersectionEvent.

        Parameters
        ----------
        system_event : OpenTURNS class UnionEvent or IntersectionEvent
            The system event cannot be a mixture of union and intersection of events.

        Returns
        -------
        new_collection : list of each Openturns class ThresholdEvent that constitutes the system event
        """
        collection = system_event.getEventCollection()
        new_collection = []
        for event in collection:
            new_collection.append(self._checkDomainEvent(event))
        return new_collection
            
    
    def _checkEvent(self,event):
        """
        Checks the validity of the given event.

        Parameters
        ----------
        event : OpenTURNS class ot.ThresholdEvent or ot.DomainEvent or ot.UniondEvent or ot.IntersectionEvent

        Returns
        -------
        new_collection : individual ThresholdEvent or list of ThresholdEvent
        """
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
        """
        Checks the validity of the given initial search direction.

        Parameters
        ----------
        alpha : list of scalars
            Its dimension must be equal to the number of random variables. This vector determines the search direction in the standard space.

        Returns
        -------
        normalized_alpha : the vector alpha after normalization
        """
        if len(alpha) != self._dim:
            raise ValueError("The dimension of alpha should be " +str(self._dim))
        else:
            normalized_alpha = alpha/np.linalg.norm(alpha)
            return [normalized_alpha]
        
    def _checkSolver(self,rootSolver):
        """
        Checks the validity of the given solver for estimating the roots along each line.

        Parameters
        ----------
        rootSolver : OpenTURNS class SafeAndSlow or MediumSafe or RiskyAndFast
            Three possibilities.

        Returns
        -------
        rootSolver
        """
        if rootSolver.getClassName() not in ['MediumSafe','RiskyAndFast','SafeAndSlow']:
            raise TypeError("The solver is not one of the required OpenTURNS classes (ot.MediumSafe(),ot.RiskyAndFast() or ot.SafeAndSlow())")
        return rootSolver
        
    
    def _checkActiveLS(self,activeLS):
        """
        Checks the validity of the parameter activeLS that determines if the alpha direction may be updated during the process.

        Parameters
        ----------
        activeLS : bool 
            True or False.

        Returns
        -------
        activeLS
        """
        if type(activeLS) is not bool:
            raise TypeError("activeLS should be True or False")
        else:
            return activeLS
        
    def _checkOppositeDirection(self,oppositeDirection):
        """
        Checks the validity of the parameter oppositeDirection that determines if roots should also be searched following the opposite direction of alpha.
        Parameters
        ----------
        oppositeDirection : bool (Default value is False)
            True or False.

        Returns
        -------
        oppositeDirection
        """
        if type(oppositeDirection) is not bool:
            raise TypeError("activeLS should be True or False")
        else:
            return oppositeDirection
        
    def _checkMinCoV(self,minCoV):
        """
        Checks the validity of the parameter minCoV which is a stopping criteria.

        Parameters
        ----------
        minCoV : positive non negative float (Default value is 0.05)

        Returns
        -------
        minCoV
        """
        if type(minCoV) is not float or minCoV <= 0:
            raise ValueError("minCoV should be a float value strictly greater than 0")
        else:
            return minCoV
        
    def _checkMaxLines(self,maxLines):
        """
        Checks the validity of the parameter maxLines which is a stopping criteria (the maximum number of lines along which roots are searched).

        Parameters
        ----------
        maxLines : positive non negative integer (Default value is 1000)

        Returns
        -------
        minCoV
        """
        if type(maxLines) is not int or maxLines <= 0:
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
        inputs = [self._currentLine[i]+self._sign*c[0]*self._alpha[-1][i] for i in range(len(self._alpha[-1]))]
        res = self._currentStandFunction(tuple([i for i in inputs]))
        return res
    
    def getSolver(self):
        return self._rootSolver
    
    def setSolver(self,rootSolver):
        self._rootSolver = self._checkSolver(rootSolver)
            
    def setSeed(self,seed=1234):
        if type(seed) is not int or seed <= 0:
            raise ValueError("seed should be a positive integer")
        else:
            self._fixedSeed = True
            return seed
            

    def _sampleLines(self,N):
        """Sample the lines."""
        if self._fixedSeed:
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
        self._maximumLines = self._checkMaxLines(maxLines)
        
    def getMinCov(self):
        return self._minCoV
        
    def setMinCov(self,minCoV):
        self._minCoV = self._checkMinCoV(minCoV)
        
    def getOppositeDirection(self):
        return self._oppositeDirection
        
    def setOppositeDirection(self,oppositeDirection):
        self._oppositeDirection = self._checkOppositeDirection(oppositeDirection)
        
    def _evaluatePf(self,roots,event):
        if None in roots or len(roots) == 0:
            Pf = 0.
        else:
            roots_pos_temp = np.sort([i for i in roots if i >= 0])
            roots_pos = []
            for i in roots_pos_temp:
                if i not in roots_pos:
                    roots_pos.append(i)
            roots_neg_temp = np.sort([i for i in roots if i <= 0])
            roots_neg = []
            for i in roots_neg_temp:
                if i not in roots_neg:
                    roots_neg.append(i)
            if 0 in roots:
                if roots != [0]*len(roots):
                    if list(roots_pos) != [0]:
                        Pf_pos = sum([(-1)**-k*ot.Normal().computeCDF(-abs(roots_pos[k])) for k in range(len(roots_pos))])
                    else:
                        Pf_pos = 1
                    if list(roots_neg) != [0]:
                        Pf_neg = sum([(-1)**k*ot.Normal().computeCDF(-abs(roots_neg[-1-k])) for k in range(len(roots_neg))])
                    else:
                        Pf_neg = 1
                    Pf = 1-Pf_pos+Pf_neg
                else:
                    Pf = 1
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
        self._sign = 1.
        pythonFunc = ot.PythonFunction(1,1,self._evaluate_line_function)
        pythonFunc = ot.MemoizeFunction(pythonFunc)
        try:
            roots = list(self._rootSolver.solve(pythonFunc,self._thresholds[index]))
            functionCalls = len(pythonFunc.getInputHistory())
            if self._oppositeDirection:
                self._sign = -1.
                pythonFunc_opp = ot.PythonFunction(1,1,self._evaluate_line_function)
                pythonFunc_opp = ot.MemoizeFunction(pythonFunc_opp)
                roots_opposite = list(self._rootSolver.solve(pythonFunc_opp,self._thresholds[index]))
                roots_opposite = [-i for i in roots_opposite]
                roots+=roots_opposite
                functionCalls += len(pythonFunc_opp.getInputHistory())
            if self._activeLS and len(roots) > 0:
                uCandidate = [self._currentLine[i] + roots[0]*self._alpha[-1][i] for i in range(len(self._alpha[-1]))]
                if np.linalg.norm(uCandidate) < np.linalg.norm(self._ustar[-1]):
                    self._ustar.append(uCandidate)
                    self._alpha.append([i / np.linalg.norm(self._ustar[-1]) for i in self._ustar[-1]])
        except:
            missedRoot = True
            roots = [None]
            functionCalls = len(pythonFunc.getInputHistory())
        root_points = [[self._currentLine[i]+self._sign*j*self._alpha[-1][i] for i in range(len(self._alpha[-1]))] for j in roots]
        return [roots,functionCalls,missedRoot,root_points]
        
    def run(self,reset=False):
        """Sample lines and find corresponding roots."""
        if self._activeLS:
            self._activeLS = False
            try:
                roots = self._find_roots([0.] * self._dim,0)[0][0]
                self._ustar = [[roots*self._alpha[-1][i] for i in range(len(self._alpha[-1]))]]
            except:
                self._ustar = [[1000] * self._dim]
                #print("WARNING: No root was found along the line of direction alpha passing through the center of the space, activeLS is set to False")
            self._activeLS = True
        
        length_alpha = len(self._alpha)
        # my_func = ot.PythonFunction(self._dim,1,self._find_roots,n_cpus=self._batchSize)
        lines = self._sampleLines(self._maximumLines)
        if reset:
            self._solvedLines = 0
            self._linePf = []
            self._CoV = [1000]
            self._roots = []
            self._rootPoints = []
            self._missedRoots = []
            self._pf = []
            self._totalFunctionCalls = []
            self._alpha = self._alpha[:1]
        while self._solvedLines < self._maximumLines and self._CoV[-1] > self._minCoV:
            res = [self._find_roots(lines[self._solvedLines],i) for i in range(len(self._eventCollection))]
            self._roots.append([i[0] for i in res])
            self._rootPoints.append([i[-1] for i in res])
            self._totalFunctionCalls.append([i[1] for i in res])
            self._missedRoots.append([i[2] for i in res])
            self._solvedLines += self._batchSize
            self._linePf.append([self._evaluatePf(list(self._roots[-1][i]),self._eventCollection[i]) for i in range(len(self._eventCollection))])
            self._pf.append([np.mean(np.array(self._linePf)[:,i]) for i in range(len(self._eventCollection))])
            if len(self._eventCollection) == 1 and self._pf[-1][0]>0:
                if len(self._pf) > 1:
                    Var = 1 / (len(self._pf)*(len(self._pf)-1)) * sum([(i[0]-self._pf[-1][0])**2 for i in self._linePf])
                    self._CoV.append(np.sqrt(Var)/self._pf[-1])
            if length_alpha < len(self._alpha):
                lines = self._sampleLines(self._maximumLines)
                length_alpha = len(self._alpha)
        
    def getResults(self):
        if len(self._eventCollection) == 1:
            dico = {'Pf_MarginalEvent': self._pf, 'CoV': self._CoV, 'roots': self._roots, 'Pf_line': self._linePf, 'alpha': self._alpha, 'lineFunctionCalls': self._totalFunctionCalls, 'rootPoints': self._rootPoints, 'missedRoots': self._missedRoots}
        else:
            print("WARNING: each given failure probability is linked to its corresponding marginal event, it does not correspond to the failure probability of the system event.")
            dico = {'Pf_MarginalEvent': self._pf, 'CoV': None, 'roots': self._roots, 'Pf_line': self._linePf, 'alpha': self._alpha, 'lineFunctionCalls': self._totalFunctionCalls, 'rootPoints': self._rootPoints, 'missedRoots': self._missedRoots}
        return dico
    
