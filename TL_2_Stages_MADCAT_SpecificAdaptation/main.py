#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 11:43:59 2017

@author: aradillas
"""
import sys
sys.path.append('/home/ahmed/Desktop/sana/RCNN/HTRTF/Modules')

 
from Structure_006_TL_2_Stages_1 import stage1
from Structure_006_TL_2_Stages_2 import stage2
from Structure_006_TL_2_Stages_2_test import test
 

def main():
	stage1()
	stage2()
	test()
			
    
if __name__ == '__main__':
    main()

