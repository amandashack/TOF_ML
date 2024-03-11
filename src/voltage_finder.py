# -*- coding: utf-8 -*-

"""
Created on Thu Jul  6 15:42:46 2023

@author: robaid
"""

import os
import numpy as np
from control_methods import *


def calculateVoltage_NelderMeade(retardationValue):


	#setup fast adjust voltages
	voltageFront = 0
	voltageBack = -1*retardationValue
	voltageMidOne = 0.11248 * (voltageFront - voltageBack) + voltageBack
	voltageMidTwo = 0.1354 * (voltageFront - voltageBack) + voltageBack
	# #run for NM
	voltageArray = voltageArrayGeneratorWrapperNM(voltageFront, voltageBack, voltageMidOne, voltageMidTwo)
	

	return voltageArray


def calculateVoltage_OneoverR(retardationValue):


	#setup fast adjust voltages
	voltageFront = 0
	voltageBack = -1*retardationValue


	# #run for NM
	voltageArray = voltageArrayGeneratorWrapperOneOverR(voltageFront, voltageBack)
	

	return voltageArray


if __name__ == "__main__":
	va = calculateVoltage_NelderMeade(200)
	print(va)