#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 08:23:19 2023

@author: c65779
"""
import openturns as ot
import scipy.optimize as opt
import numpy as np
### Problème avec DomainEvent : 'DomainEvent' object has no attribute 'getLevel' + 'getOperator'. Pareil pour DomainEvent.getDomain() 

class RandomSets():
    """
    Description.
    """
    def __init__(self,copula,listType,listParameters):
        _ = self._checkVariablesDefinition()
        _ = self._checkCopula()
        self._listType = listType
        self._listParameters = listParameters
       
    def _checkCopula(self):
        check = None             

    def _checkVariablesDefinition(self):
        for i in range(len(self._listType)):
            if type(self._listType[i]) is not str:
                raise TypeError("Variable "+str(i) + 'must be a string from the list: ' + str(['Probability','P-box','Convex','FuzzyTriangle','FuzzyTrapeze','FuzzyNormal']))
            if self._listType[i] not in ['Probability','P-box','Convex','FuzzyTriangle','FuzzyTrapeze','FuzzyNormal']:
                raise TypeError("Variable "+str(i) + 'named ' + str(self._listType[i]) + ' is not in the list: ' + str(['Probability','P-box','Convex','FuzzyTriangle','FuzzyTrapeze','FuzzyNormal']))
            if self._listType[i] == 'Probability' and 'openturns.dist' not in type(self._listParameters[i]):
                raise TypeError("Variable "+str(i) + 'type is Probability but its parameter is not an openturns distribution')
            if self._listType[i] == 'P-box':
                if len(self._listParameters[i])!=2:
                    raise TypeError("Variable "+str(i) + 'type is P-box but the corresponding list is not of dimension 2')
                for j in self._listParameters[i]:
                    if 'openturns.dist' not in type(j):
                        raise TypeError("Variable "+str(i) + 'type is P-box but at least one parameter is not an openturns distribution')


    def _fuzzyTriangular(self,alpha,param):
        I = [alpha*(param[1]-param[0])+param[0], -alpha*(param[2]-param[1])+param[2]]
        return I

    def _fuzzyTrapezoidal(self,alpha,param):
        I = [alpha*(param[1]-param[0])+param[0], -alpha*(param[3]-param[2])+param[3]]
        return I

    def _fuzzyNormal(self,alpha,param):
        x1 = 0.5*(2*param[0]-np.sqrt(-8*param[1]**2*np.log(alpha)))
        x2 = 0.5*(2*param[0]+np.sqrt(-8*param[1]**2*np.log(alpha)))
        return [np.max([param[2],x1]),np.min([param[3],x2])]
    
    def _pbox(self,alpha,cdfs):
        res = []
        for i in cdfs:
            res.append(i.computeQuantile(alpha)[0])
        I = [min(res),max(res)]
        return I
        
    def _convexParallelepiped(self,alpha,Zc,Zw,rho):
        nb_var = len(Zc)
        w = [1/sum([np.abs(rho[i][j]) for j in range(nb_var)]) for i in range(nb_var)]
        Z = [Zw[i]*w[i]*sum([rho[i][k]*alpha[k] for k in range(nb_var)]) + Zc[i] for i in range(nb_var)]
        return Z
    
    def generateSample(self):
        generate = None
        
        