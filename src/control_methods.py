# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 10:38:31 2023

@author: lauren
"""

import os
import re
import numpy as np


###########define file locations here

# define the directory for the text file that will serve as the output log.  This file is created by simion when
# simion is done running, and can then be loaded in and parsed with python to convert the simion output
# into a more useful array.  This file may be accessed by multiple different functions,
# and it is helpful to establish it in the main script's scope.
baseDir = "C:/Users/proxi/Documents/SimionRunFiles"
outputFile = baseDir + "/flightOutputLog.txt"
#denote where the potential array file is
potArrLoc = baseDir + "/copiedArray.PA0"
#"C:/Users/andre/Documents/GitHub/kamalovSimionLearning/runFiles/LCL5005-010580_TOF_HighRes.PA0"
#select the .REC file to dictate recording options
recordingFile = baseDir + "/recordingOptions.rec"
#select the .IOB file to dictate the interactive bench to use for this simion call
iobFileLoc = baseDir + "/workbench.iob"
#specify the directory for the .fly2 file.  The fly2 file defines the particles that will be flown in the simulation.
fly2FileLoc = baseDir + "/ionsToFly.fly2"
# specify the location for the .lua file which sets the scale parameter
luaFileLoc = baseDir + "/set_scale.lua"

#%%


def generate_lognormal_energies(num_particles, median, sigma):
    # median and sigma are the parameters for the lognormal distribution
    # Note: np.exp(median) is the scale parameter (exp(mean of the underlying normal distribution))
    return np.random.lognormal(mean=np.log(median), sigma=sigma, size=num_particles)


# generate a .fly2 file type.  file name must be provided as directory string INCLUDING the .FLY2 post-fix
def generate_fly2File_lognorm(filenameToWriteTo, numParticles=500, medianEnergy=10, energySigma=1):
	# Generate lognormal distributed energies
	energies = generate_lognormal_energies(numParticles, medianEnergy, energySigma)
	# check if .fly2 file with this name already exists.
	os.chdir("../simulations/simionSimulationFiles")
	fileExists = os.path.isfile(filenameToWriteTo)
	# delete previous copy, if there is one
	if fileExists:
		os.remove(filenameToWriteTo)

	# open up file to write to
	with open(filenameToWriteTo, "w") as fileOut:
		# Write out the .fly2 scripts
		fileOut.write("particles {\n")
		fileOut.write("  coordinates = 0,\n")
		fileOut.write("  standard_beam {\n")
		fileOut.write("    n = " + str(numParticles) + ",\n")
		fileOut.write("    tob = 0,\n")
		fileOut.write("    mass = 0.000548579903,\n")
		fileOut.write("    charge = -1,\n")
		fileOut.write("    ke = " + str(energies) + "\n")
		fileOut.write("    az =  uniform_distribution {\n")
		fileOut.write("      min = -2.5,\n")
		fileOut.write("      max = 2.5\n")
		fileOut.write("    },\n")
		fileOut.write("    el =  uniform_distribution {\n")
		fileOut.write("      min = -5,\n")
		fileOut.write("      max = 5\n")
		fileOut.write("    },\n")
		fileOut.write("    cwf = 1,\n")
		fileOut.write("    color = 0,\n")
		fileOut.write("    position =  sphere_distribution {\n")
		fileOut.write("      center = vector(244, 0, 0),\n")
		fileOut.write("      radius = 0,\n")
		fileOut.write("      fill = true")
		fileOut.write("    }\n")
		fileOut.write("  }\n")
		fileOut.write("}")


# generate a .fly2 file type.  file name must be provided as directory string INCLUDING the .FLY2 post-fix
def generate_fly2File(filenameToWriteTo, numParticles=500, meanEnergy=10, energySTD=0):
	#check if .fly2 file with this name already exists.
	os.chdir("/simulations/simionSimulationFiles")
	fileExists = os.path.isfile(filenameToWriteTo)
	#delete previous copy, if there is one
	if fileExists == True:
		os.remove(filenameToWriteTo)

	#open up file to write to
	fileOut = open(filenameToWriteTo, "w")

	#write out the .fly2 scripts
	fileOut.write("particles {\n")
	fileOut.write("  coordinates = 0,\n")
	fileOut.write("  standard_beam {\n")
	fileOut.write("    n = " + str(numParticles) + ",\n")
	fileOut.write("    tob = 0,\n")
	fileOut.write("    mass = 0.000548579903,\n")
	fileOut.write("    charge = -1,\n")
	fileOut.write("    ke =  gaussian_distribution {\n")
	fileOut.write("      mean = " + str(meanEnergy) + ",\n")
	fileOut.write("      stdev = " + str(energySTD) + "\n")
	fileOut.write("    },\n")
	fileOut.write("    az =  uniform_distribution {\n")
	fileOut.write("      min = -2.5,\n")
	fileOut.write("      max = 2.5\n")
	fileOut.write("    },\n")
	fileOut.write("    el =  uniform_distribution {\n")
	fileOut.write("      min = -5,\n")
	fileOut.write("      max = 5\n")
	fileOut.write("    },\n")
	fileOut.write("    cwf = 1,\n")
	fileOut.write("    color = 0,\n")
	fileOut.write("    position =  sphere_distribution {\n")
	fileOut.write("      center = vector(244, 0, 0),\n")
	fileOut.write("      radius = 0,\n")
	fileOut.write("      fill = true")
	fileOut.write("    }\n")
	fileOut.write("  }\n")
	fileOut.write("}")

	#close file
	fileOut.close()


# method fastAdj performs a fast adjustment of the voltage array for the PA0 file that is referenced in potArrayFile.
def fastAdj(voltageArray, potArrayFile):
	# pa0 file directory
	potArrLoc = potArrayFile
	# initialize string that will be supplied as argument of voltage values
	voltArgString = str()
	# convert supplied potential voltages into a string
	for i in range(voltageArray.size):
		# setup values for electrode number and voltage
		electrodeNumber = i + 1
		voltage = voltageArray[i]
		if((i+1) == voltageArray.size):
			stringToAdd = str(electrodeNumber) + "=" + str(voltage)
		else:
			stringToAdd = str(electrodeNumber) + "=" + str(voltage) + ","
		# add current electrode's parameters to string
		voltArgString = voltArgString + stringToAdd

	# go to simion's working directory and call simion
	os.chdir(r"C:\Users\proxi\Downloads\Simion_8-1-20230825T223627Z-001\Simion_8-1")
	os.system("simion.exe --nogui fastadj " + potArrLoc + " " + voltArgString)



# helper method that makes the call to run simion to fly particles.  This method links up all the required file
# directories into a single call to simion, which will then fly the simulation and create an outputFile
# fly2FileDir contains the definition of the particles that should be simulated.
# This file is made with the method 'makeFLY2File()'
# outputFile is the directory to which the output log will be saved.  This log is a summary of the simulation
# results, and is made by simion as simion runs the simulation
# recordingFile is a special file that has the recording options.  This is a file that can only be made in simion -
# it is a binary file that has a bunch of flags to tell the program what to record during simulation.
# the iobFileLoc is directory to the .IOB file, which is the ion bench file.  I am not fully sure I understand what
# this is, but I think it is a file that links the potential arrays to the simulation.
def runSimion(fly2FileDir, luaFile, outputFile, recordingFile, iobFileLoc):
	# Check if outputFile exists and delete it if it does
	if os.path.isfile(outputFile):
		os.remove(outputFile)

	# Command parts may need to be individually quoted if they contain spaces
	luaCommand = f"--lua @{luaFile}"
	flyCommand = f"fly --recording-output=\"{outputFile}\" --recording=\"{recordingFile}\" " \
				 f"--particles=\"{fly2FileDir}\" --restore-potentials=0 \"{iobFileLoc}\""

	# Construct the full command
	fullCommand = f"simion.exe --nogui {luaCommand} {flyCommand}"

	# Change to the SIMION working directory
	os.chdir(r"C:\Users\proxi\Downloads\Simion_8-1-20230825T223627Z-001\Simion_8-1")

	# Execute the command
	os.system(fullCommand)
	# delete the temporary files
	os.chdir(baseDir)
	os.system('del *.tmp')


# a helper method to convert simion's output text log into python variables.  Note that this depends on the format
# of the log file, which is controlled by a simion generated .REC file.  This current method assumes that the .REC is
# set to only record ion splats, has a minimal amount of header, and the recorded values are a comma-delimited
# string of numbers that are ended with a new line character.
def postProcessSimionOutputLog(logDirectory):
	# denote the regular expression to search for using re.findall
	# this reg expression is pretty good at describing anything that could be considered a number.
	regExMatch = "(\-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?)"


	# specify the number of unique values recorded in the output log per ion event.
	# This is controlled with the .REC file
	numValues = 11
	# specify pattern to search the text log for.  Since this variant assumes that the .REC file specifies to
	# spit out only a line of numbers for each instance, can use a repeating pattern
	builtUpPattern = ""
	for i in range(numValues - 1):
		builtUpPattern += regExMatch + ","
	# for final number, add regExMatch and end of line character
	builtUpPattern += regExMatch + "\n"

	# I now have the string that I will be searching for within the file.
	# open up and read the file into a singular string
	openedFile = open(logDirectory, "r")
	fullFileString = openedFile.read()
	# find the instances that meet the required expression.
	allFoundInstances = re.findall(builtUpPattern, fullFileString)
	numHits = len(allFoundInstances)

	# initialize array that will hold the results
	arrayExtractedData = np.zeros((numValues, numHits))
	for i in range(numHits):
		instanceNow = allFoundInstances[i]
		for j in range(numValues):
			# convert extracted string into a float value
			arrayExtractedData[j, i] = float(instanceNow[j])

	# close out the opened file
	openedFile.close()

	# return array of extracted values.
	return arrayExtractedData


# convert a simion generated flight log to a more useful python array.  Requires used to supply complete directory
# for text log file.  Note that this verion may assume the flight log has some text to function.  A better way is
# to change the recording options to minimize amount of text that is printed to the log
def postProcessSimionOutputLog_withText(logDirectory):
	# denote the regular expression to search for using re.findall
	# this reg expression is pretty good at describing anything that could be considered a number.
	regExMatch = "(\-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?)"
	# it is useful to be able to build up the regular expression that is searched for with smaller conglomerates.
	# Those tidbits are defined here, and the conglomerate is joined later
	tidbitIonHit = "Ion\(" + regExMatch + "\) Event\(Hit Electrode\) "
	tidbitTOF = "TOF\(" + regExMatch + " usec\) "
	tidbitX = "X\(" + regExMatch + " mm\) "
	tidbitY = "Y\(" + regExMatch + " mm\) "
	tidbitZ = "Z\(" + regExMatch + " mm\) "
	tidbitVt = "Vt\(" + regExMatch + " mm/usec\) "
	tidbitVx = "Vx\(" + regExMatch + " mm/usec\) "
	tidbitVy = "Vy\(" + regExMatch + " mm/usec\) "
	tidbitVz = "Vz\(" + regExMatch + " mm/usec\) "
	tidbitKeError = "KE_Error\(" + regExMatch + " eV\)"

	# build up the pattern that will be searched for, using the individual tidbits
	builtUpPattern = tidbitIonHit + tidbitTOF + tidbitX + tidbitY + tidbitZ + "\n" + tidbitVt + tidbitVx + \
					 tidbitVy + tidbitVz + "\n" + tidbitKeError
	numValues = 10

	# I now have the string that I will be searching for within the file.
	# open up and read the file into a singular string
	# os.chdir("C:/Users/andre/Documents/SimionRunFiles")
	openedFile = open(logDirectory, "r")
	fullFileString = openedFile.read()
	# find the instances that meet the required expression.
	allFoundInstances = re.findall(builtUpPattern, fullFileString)
	numHits = len(allFoundInstances)

	# initialize array that will hold the results
	arrayExtractedData = np.zeros((numValues, numHits))
	for i in range(numHits):
		instanceNow = allFoundInstances[i]
		for j in range(numValues):
			# convert extracted string into a float value
			arrayExtractedData[j, i] = float(instanceNow[j])

	# close out the opened file
	openedFile.close()

	# return array of extracted values.
	return arrayExtractedData

# return the node voltages for the 1/R configuration, given a user supplied front and rear voltages.
# Has hard-coded resistor values that match those determined by Averell Gaston.
def calculateOneOverRVoltageBridge(voltageFront, voltageBack):
	# resistor values (in MegaOhms) for 1/R as selected by Ave
	resistorValues = np.array([1.84, \
					1.14, \
					1.09, \
					1.04, \
					0.993, \
					0.950, \
					0.910, \
					0.873, \
					0.837, \
					0.805, \
					0.773, \
					0.744, \
					0.716, \
					0.690, \
					0.666, \
					0.642, \
					0.620, \
					0.599, \
					0.579, \
					0.560, \
					0.541, \
					0.524, \
					0.508, \
					0.492, \
					0.477, \
					0.396])

	# setup an array that will hold the relative voltage values for each blade/lead.
	# This will be done by building up a resistor bridge
	numResistors = len(resistorValues)
	numVoltages = numResistors + 1#the number of voltages is the number of resistors plus one,
	# since I want voltages at leads , which are bridged by resistors.
	voltageValuesRelative = np.zeros(numVoltages)
	# to keep track of where I am on the resistor bridge, keep sums of resistors counted up so far
	resistanceSummedSoFar = 0
	resistanceTotal = np.sum(resistorValues)
	# fill in last value of voltage array:
	voltageValuesRelative[-1] = resistanceSummedSoFar / resistanceTotal
	# loop through the rest voltageValues array and fill it up from end to start.
	# use range to go from largest index left, to 0 in steps of -1.
	for i in range(numResistors - 1, -1, -1):
		resistanceSummedSoFar += resistorValues[i]
		voltageValuesRelative[i] = resistanceSummedSoFar /resistanceTotal

	# calculate the actual voltage values based on the relative distribution.
	voltageDifference = voltageFront - voltageBack
	voltages = voltageDifference*voltageValuesRelative + voltageBack

	return voltages


# return the voltages for the NM configuration, given a user supplied front, blade22, blade25, and rear voltages.
# Has hard-coded resistor values that match those determined by Averell Gaston.
def calculateNMVoltageBridge(voltageFront, voltageBack, midOneVoltage, midTwoVoltage):
	# denote the locations along the voltage node of the mid voltage sources.
	# They are counted assuming the first value is numbered "0"
	midOneLeadLocation = 22
	midTwoLeadLocation = 25
	frontLocation = 0

	# resistor values (in MegaOhms) for NM as selected by Ave
	resistorValues = np.array([3.08, \
					2.43, \
					1.75, \
					1.25, \
					0.897, \
					0.664, \
					0.528, \
					0.446, \
					0.457, \
					0.483, \
					0.528, \
					0.577, \
					0.621, \
					0.650, \
					0.657, \
					0.639, \
					0.593, \
					0.520, \
					0.422, \
					0.305, \
					0.178, \
					0.048, \
					0.072, \
					0.167, \
					0.220, \
					2.13])

	# setup an array that will hold the relative voltage values for each blade/lead.
	# This will be done by building up a resistor bridge
	numResistors = len(resistorValues)
	numVoltages = numResistors + 1 # the number of voltages is the number of resistors plus one,
	# since I want voltages at leads , which are bridged by resistors.
	voltages = np.zeros(numVoltages)

	# to calculate NM voltages, need to treat it as a series of voltage divider bridges,
	# depending on where the leads are located.
	#take care of first monotonic series, between front mesh and midOne
	# perform setup
	voltageValuesRelativeNow = np.zeros(midOneLeadLocation + 1 - frontLocation)
	resistanceSum = 0
	# handle case of voltage lead prior to any resistors being encountered
	voltageValuesRelativeNow[0] = resistanceSum
	# run through the loop of having the resistors
	for i in range(frontLocation, midOneLeadLocation):
		resistanceSum += resistorValues[i]
		voltageValuesRelativeNow[i + 1 - frontLocation] = resistanceSum
	# completed the loop.  normalize by total resistance crossed:
	voltageValuesRelativeNow = voltageValuesRelativeNow/resistanceSum
	# change normalized offset to represent real voltages, depending on those supplied
	voltageDifference = (midOneVoltage - voltageFront)
	voltages[frontLocation:(midOneLeadLocation + 1)] = voltageDifference*voltageValuesRelativeNow + voltageFront



	# take care of second monotonic series, between midOne and midTwo
	# perform setup
	voltageValuesRelativeNow = np.zeros(midTwoLeadLocation + 1 - midOneLeadLocation)
	resistanceSum = 0
	# handle case of voltage lead prior to any resistors being encountered
	voltageValuesRelativeNow[0] = resistanceSum
	# run through the loop of having the resistors
	for i in range(midOneLeadLocation, midTwoLeadLocation):
		resistanceSum += resistorValues[i]
		voltageValuesRelativeNow[i + 1 - midOneLeadLocation] = resistanceSum
	# completed the loop.  normalize by total resistance crossed:
	voltageValuesRelativeNow = voltageValuesRelativeNow/resistanceSum
	# change normalized offset to represent real voltages, depending on those supplied
	voltageDifference = (midTwoVoltage - midOneVoltage)
	voltages[midOneLeadLocation:(midTwoLeadLocation + 1)] = voltageDifference*voltageValuesRelativeNow + midOneVoltage



	backLocation = numResistors
	# take care of final monotonic series, between midTwo and back mesh
	# perform setup
	voltageValuesRelativeNow = np.zeros(backLocation + 1 - midTwoLeadLocation)
	resistanceSum = 0
	# handle case of voltage lead prior to any resistors being encountered
	voltageValuesRelativeNow[0] = resistanceSum
	# run through the loop of having the resistors
	for i in range(midTwoLeadLocation, backLocation):
		resistanceSum += resistorValues[i]
		voltageValuesRelativeNow[i + 1 - midTwoLeadLocation] = resistanceSum
	# completed the loop.  normalize by total resistance crossed:
	voltageValuesRelativeNow = voltageValuesRelativeNow/resistanceSum
	# change normalized offset to represent real voltages, depending on those supplied
	voltageDifference = (voltageBack - midTwoVoltage)
	voltages[midTwoLeadLocation:(backLocation + 1)] = voltageDifference*voltageValuesRelativeNow + midTwoVoltage
	return voltages


# this is a helper method that looks at all the electrons hits in runResults, and selects which ones to keep.
# The selection can be done, for example, by only looking at hits that splat at the flight-axis position of
# the MCP front.  Filtering is highly specific to setup geometry or user preference.  This method should be
# modified frequently in accordance to user desires.
def resultFilter(runResults):
	# recall the number of hits that were extracted from the hit log file
	numSplats = runResults.shape[1]
	# look up number of values printed to the log for each splat
	numValues = runResults.shape[0]
	# initialize a counter for the number of splats that pass the filter condition and were added to the good results array
	numGoodSplats = 0
	for i in range(numSplats):
		# load up specific hit to analyze here
		splatNow = runResults[:, i]
		# check the filtering condition here.  It is currently set to filter splats (ie, hits)
		# that end at the front surface of the MCP.
		if(int(splatNow[4]) == 8134):
			#filter condition passed
			#array appending will require the vector to be converted to array dimensions.  This needs to be done for numpy syntax reasons.
			splatNow = np.expand_dims(splatNow, 1)
			#check if the good splat array has been initialized yet
			if(numGoodSplats == 0):
				#it has not yet been started.  start it up
				goodSplatArray = np.array(splatNow)
				numGoodSplats += 1
			else:
				#array has previously been started.  append to previously initiated array
				goodSplatArray = np.append(goodSplatArray, splatNow, 1)
				numGoodSplats += 1
		else:
			#filter condition failed
			#do nothing
			pass

	if(numGoodSplats > 0):
		return goodSplatArray, numGoodSplats
	else:
		return [], numGoodSplats


# this method is here to have a compact method that generates the voltage array that will be used for simulation.
# If the simion files change, this method must be updated to reflect that.
# used for NM voltage distribution on lens blades
def voltageArrayGeneratorWrapperNM(voltageFront, voltageBack, midOneVoltage, midTwoVoltage):
	# current (10/9/2020) version of file has 37 electrodes
	voltageArray = np.zeros(37)

	# call a helper method to get voltage settings of NM lens stack, given a front and back mesh voltage
	lensVoltages = calculateNMVoltageBridge(voltageFront, voltageBack, midOneVoltage, midTwoVoltage)

	voltageArray[1:26] = lensVoltages[1:26]#control lens stack voltages
	voltageArray[27] = lensVoltages[26] #set back mesh voltage to be same as final electric lens voltage
	voltageArray[26] = lensVoltages[0]  #set front mesh voltage to be same as first lens voltage
	voltageArray[[29, 30, 34, 36]] = lensVoltages[0] #set potential for nose and surrounding material
	voltageArray[0] = lensVoltages[0] #set the voltage of outer construct of model - material that should be on outside of flight path
	voltageArray[28] = voltageArray[27] #control voltage of MCP mesh to be same as voltage of back mesh
	voltageArray[31:33] = voltageArray[28] #control inner surface of flight tube to be same voltage as MCP mesh
	voltageArray[33] = voltageArray[28] + 300 #set MCP front voltage to be 300 V above the MCP mesh
	voltageArray[35] = lensVoltages[0] #control voltage of mesh near electron generation point, to define voltage
	# near area where electrons are created
	return voltageArray


# this method is here to have a compact method that generates the voltage array that will be used for simulation.
# If the simion files change, this method must be updated to reflect that.
# used for 1/R voltage distribution on lens blades
def voltageArrayGeneratorWrapperOneOverR(voltageFront, voltageBack):
	# current (10/9/2020) version of file has 37 electrodes
	voltageArray = np.zeros(37)

	# call a helper method to get voltage settings of 1/R lens stack, given a front and back mesh voltage
	lensVoltages = calculateOneOverRVoltageBridge(voltageFront, voltageBack)

	voltageArray[1:26] = lensVoltages[1:26] # control lens stack voltages
	voltageArray[27] = lensVoltages[26] # set back mesh voltage to be same as final electric lens voltage
	voltageArray[26] = lensVoltages[0]  # set front mesh voltage to be same as first lens voltage
	voltageArray[[29, 30, 34, 36]] = lensVoltages[0] # set potential for nose and surrounding material
	voltageArray[0] = lensVoltages[0] # set the voltage of outer construct of model - material that should be on
	# outside of flight path
	voltageArray[28] = voltageArray[27] # control voltage of MCP mesh to be same as voltage of back mesh
	voltageArray[31:33] = voltageArray[28] # control inner surface of flight tube to be same voltage as MCP mesh
	voltageArray[33] = voltageArray[28] + 300 # set MCP front voltage to be 300 V above the MCP mesh
	voltageArray[35] = lensVoltages[0] # control voltage of mesh near electron generation point, to define voltage near
	# area where electrons are created

	return voltageArray


# run the simulation for a number of different energies, as listed in energiesToRun array.
# User can stipulate the number of particles to run, and energy spread to allow here.
# It should be possible to expand more user control down the road.
def runMultipleEnergies(voltageArray, energiesToRun, numParticles=500, energySTD=0, runType=""):
	# adjust the simulated detector to use specified voltages for the electrodes that are defined in the .PA# file.
	# potArrLoc specifies the location the .PA0 file, which is a refined version of the PA#?  IDK how this works tbh
	fastAdj(voltageArray, potArrLoc)
	# declare an empty list to track the found time of flights associated with specific energy values.
	simulatedToF_list = []
	# check to see if energiesToRun can be iterated through
	if(energiesToRun.size > 1):
		# can iterate - there is in fact, only one energy value
		# run the individual analysis for each energy configuration
		for E in energiesToRun:
			print("Currently running energy: " + str(E))
			# for each energy, specify the .fy2 file to tell simion what electrons to fly
			generate_fly2File_lognorm(fly2FileLoc, numParticles=numParticles, medianEnergy=E, energySigma=1)
			# run simion
			runSimion(fly2FileLoc, outputFile, recordingFile, iobFileLoc)
			# retrieve results from simion's .txt log
			runResults = postProcessSimionOutputLog(outputFile)


# This is a bit of an outdated methof.  It was initially built to help plot the results of a completed run,
# as represented through a filtered set of hits in filteredHitsList.  The method has since been repurposed to
# calculate the weighted average of the time of flight is for a run.  Doing this calculation could be done a LOT
# more fficiently and better.  However, this method has stuck around somehow.
def calculateToF_forSingleRun(filteredHitsList):
	# parameters for histogram plot
	bins = 800001
	rangeMin = 0
	rangeMax = 2

	timeAxis = np.linspace(rangeMin, rangeMax, bins)
	histogram = np.zeros(timeAxis.size)
	stepSize = (rangeMax - rangeMin)/(bins-1)
	if len(filteredHitsList) > 0:
		timeSplatsNow = filteredHitsList[1, :]
		# add each individual time of splat to the histogram
		for j in timeSplatsNow:
			binToAddTo = (j/20 - rangeMin)/stepSize # the origin of the 20 is regrettable, but important.
			# Simion can generate .PA models based on autocad files.  This was done to generate my sample.
			# During this generation, I used a value of '20' for scale, thinking this would increase the spatial
			# resolution for the model (which it does), and not realizing that this would blow my model to be 20
			# times the actual size.  So what was meant to be a ~0.4 meter long tube, is simulated as a 20-meter-
			# long tube.  This was at first considered disastrous.  Thankfully, it is okay - because all calculations
			# are done in potential, which is linear in space, the model's behaviour scales linearly with this '20'
			# scale.  This was verified.  This scale factor can be safely removed, by stretching time by a linear
			# scale factor of 1/20.  That is the origin of this value.  It must be kept for the model as supplied.
			# Sorry.
			if binToAddTo <= bins:
				histogram[int(binToAddTo)] += 1

	# try to find the time value most associated with this energy.
	if np.sum(histogram > 0):
		avgTimeHistogram = np.average(timeAxis, weights=histogram)
	else:
		avgTimeHistogram = 0

	return avgTimeHistogram*1e-6