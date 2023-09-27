# -*- coding: utf-8 -*-
"""
Full pipeline: pre-processing + 
"""
from src.pre_process import PreProcessor

class Pipeline:
    """ Pipeline to run all steps """
    def __init__(self):
        self.pre_process = PreProcessor()


