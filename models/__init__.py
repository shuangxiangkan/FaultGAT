"""
Models package for Fault Diagnosis with GNN.

This package contains neural network models for fault diagnosis:
- FaultGAT: Graph Attention Network for PMC model
- RNNIFDCom_PMC: RNN-based Intermittent Fault Diagnosis Communication for PMC model
"""

from .FaultGAT import FaultGAT
from .RNNIFDCom_PMC import RNNIFDCom_PMC

__all__ = ['FaultGAT', 'RNNIFDCom_PMC'] 