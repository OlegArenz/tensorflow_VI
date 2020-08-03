#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 17:14:01 2019

@author: mzhong
"""

from abc import ABC, abstractmethod

class Distribution(ABC):
    # Abstract Base Class for defining a distribution
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def density(self):
        # Compute the density function
        pass
    
    @abstractmethod
    def log_density(self):
        # Compute the log density function
        pass
    
    @abstractmethod
    def sample(self):
        # draw a sample from this distribution
        pass