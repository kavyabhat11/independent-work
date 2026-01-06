from __future__ import print_function
import os
import copy
import time
import pickle
from music21 import *
import DT
import datetime
from pprint import pprint

_USEANALYSISOFFSETS = False		# compare on an eighth-note basis or when chord changes happen? eighth notes are better for keyboard textures
_PRINTINDIVIDUAL = False

accuracies = []
printComparison = False
confusionMatrix = {}

"""
Compare two RN analyses.

TODO:
	would be great to give some flexibility on just where the key changes, to capture the pivot chord idea
	
"""

def gather_stats(source="BACH", source2 = 'BACH CHORALES HMM CLEAN', show = False, clean = True, saveFile = False, ignoreSixFour = False, ignoreInversions = False):
	global counts, locations, currentSize, accuracies, analyzed_file1, analyzed_file2, printComparison, keyErrors, chordErrors, totalCorrectEighths
	global globalSource, globalSave, outputFile, inversionErrors, globalIgnoreSixFour, globalIgnoreInversions, wrongKeyRightChord, confusionMatrix, confusions
	print(f'Comparing {source} to {source2}')
	recordResults = False
	if show == False: 
		printComparison = False
	else:
		printComparison = True
	globalIgnoreInversions = ignoreInversions
	globalIgnoreSixFour = ignoreSixFour
	keyErrors = 0
	chordErrors = 0
	totalCorrectEighths = 0
	inversionErrors = 0
	wrongKeyRightChord = 0
	globalSave = saveFile
	confusionMatrix = {}
	pcts = []
	startT = time.time()
	fileStr = ''
	accuracies = []
	wrongKey = []
	wrongChord = []
	confusions = {}
	globalSource = source.upper()
	if globalSave:
		s = datetime.datetime.fromtimestamp(time.time())
		fileName = globalSource + 'accuracies' + str(s.year) + '-' + str(s.month) + '-' + str(s.day) + '-' + str(s.hour) + '.' + str(s.minute) + '.txt'
		outputFile = open(DT._DATAPATH + fileName, 'wb')
	if globalSource in DT._PATHS:
		filePath = DT._PATHS[globalSource][0]
		fileStr = globalSource
		files = sorted(os.listdir(filePath))
		recordResults = True
		counter = 1
		for myFile in files:
			if myFile[-3:] == 'txt':
				try:
					analyzed_file2 = converter.parse(DT._PATHS[source2][0] + myFile, format='romantext')
				except:
					continue
				analyzed_file1 = converter.parse(filePath + myFile, format='romantext')
				if _PRINTINDIVIDUAL: print_out(myFile + ' ', end = '')
				pct, wK, wC = dochorale(myFile)
				pcts.append(pct)
				wrongKey.append(wK)
				wrongChord.append(wC)
				accuracies.append([myFile, pct])
				#confusions[myFile] = confusionLocations
	minutes = (time.time() - startT)/60.
	print_out('Total analysis took ' + str(int(minutes)) + ' minutes and ' + str(int((minutes - int(minutes)) * 60)) + ' seconds')
	s1 = f'Average correctness {round(1.0 * sum(pcts)/len(pcts), 1)}% (per chorale) {round(100*totalCorrectEighths/(chordErrors + keyErrors + totalCorrectEighths), 1)}% (per chord)'
	s2 = 'Correct key %.2f' % (100. - 1.0 * sum(wrongKey)/len(wrongKey)) + ' Correct chord (given correct key) %.2f' % (100. - 1.0 * sum(wrongChord)/len(wrongChord)) + ' (piece average)'
	s3 = 'Correct key %.1f Correct chord (given correct key) %.1f (chord average)' % (100. - (100.*keyErrors/(chordErrors + keyErrors + totalCorrectEighths)), 100. - (100.*chordErrors/(chordErrors + totalCorrectEighths)))
	sOut = '\r\n'.join([s1, s2, s3])
	print_out(sOut)
	print("Wrong key but right chord", wrongKeyRightChord, 'out of', keyErrors, f"key errors ({int(round(100*wrongKeyRightChord/keyErrors, 0))}%)")
	if globalSave:
		for thing in accuracies: 
			accuracies = sorted(accuracies, key = lambda x: x[1])
			outputFile.write(str(thing) + '\n')
		outputFile.close()

def two_files(file1 = 'op127m1', file2 = False, show = True):
	global counts, locations, currentSize, accuracies, analyzed_file1, analyzed_file2, printComparison, keyErrors, chordErrors, totalCorrectEighths, inversionErrors, inversionErrorPct
	global globalSave, globalIgnoreSixFour, globalIgnoreInversions, wrongKeyRightChord
	myPath = DT._PATHS['Tim'][0]
	globalSave = False
	if show == False: 
		printComparison = False
	else:
		printComparison = True
	if not file2:
		file2 = myPath + file1 + 'ORIG.txt'
	else:
		file2 = myPath + file1 
	file1 = myPath + file1 + '.txt'
	recordResults = False
	keyErrors = 0
	chordErrors = 0
	totalCorrectEighths = 0
	inversionErrors = 0
	pcts = []
	wrongKey = []
	wrongChord = []
	startT = time.time()
	fileStr = ''
	accuracies = []
	analyzed_file2 = converter.parse(file2, format='romantext')
	analyzed_file1 = converter.parse(file1, format='romantext')
	pct, wK, wC = dochorale()
	pcts.append(pct)
	wrongKey.append(wK)
	wrongChord.append(wC)
	minutes = (time.time() - startT)/60.
	print('Total analysis took', int(minutes), 'minutes and', int((minutes - int(minutes)) * 60), 'seconds')
	s1 = 'Average correctness (per chorale) %.2f' % (1.0 * sum(pcts)/len(pcts))
	s2 = 'Wrong key %.2f' % (1.0 * sum(wrongKey)/len(wrongKey)) + ' Wrong chord %.2f' % (1.0 * sum(wrongChord)/len(wrongChord)) + ' Correct ' + str(totalCorrectEighths)
	s3 = 'Correct key %.1f percent, Correct chord (given correct key) %.1f percent' % (100. - (100.*keyErrors/(chordErrors + keyErrors + totalCorrectEighths)), 100. - (100.*chordErrors/(chordErrors + totalCorrectEighths)))
	sOut = '\r\n'.join([s1, s2, s3])
	print("inversion error percentage", round(inversionErrorPct, 1))
	print(sOut)
	return pct

def dochorale(pieceName = ''):
	global counts, locations, output, pTreb, pBass, chorale, analyzed_file1, analyzed_file2, printComparison, keyErrors, chordErrors, totalCorrectEighths
	global confusionMatrix, inversionErrorPct, inversionErrors, globalIgnoreSixFour, globalIgnoreInversions, wrongKeyRightChord, confusions
	anal1 = analyzed_file1.flat.getElementsByClass(['RomanNumeral', 'RomanTextUnprocessedToken']).stream()
	anal2 = analyzed_file2.flat.getElementsByClass(['RomanNumeral', 'RomanTextUnprocessedToken']).stream()
	
	totalOffsets = sorted({x.offset for x in anal1}.union({x.offset for x in anal2}))
	finalOffset = max(totalOffsets)
	
	if not _USEANALYSISOFFSETS:
		totalOffsets = [x/2. for x in range(0, int(finalOffset * 2) + 1)]
	
	correctEighths = 0
	wrongEighths = 0
	localKeyErrors = 0
	localChordErrors = 0
	localWrongKeyRightChord = 0
	lastRN1 = False
	lastRN2 = False
	
	for timePoint in totalOffsets:
		rn1 = anal1.getElementsByOffset(timePoint, timePoint, includeEndBoundary = True, mustBeginInSpan = False, classList = ['RomanNumeral'])
		rn2 = anal2.getElementsByOffset(timePoint, timePoint, includeEndBoundary = True, mustBeginInSpan = False, classList = ['RomanNumeral'])
		if not rn1 or not rn2: continue
		rn1 = rn1[0]
		rn2 = rn2[0]
		pitchNames1 = [x.name for x in rn1.pitches]
		pitchNames2 = [x.name for x in rn2.pitches]
		
		#if pitchNames1 == pitchNames2 and DT.parse_figure(rn1.figure)[0] == DT.parse_figure(rn2.figure)[0]:
		if globalIgnoreSixFour and rn2.figure.count('6/4'): 
			rn2 = lastRN2
		if rn1.figure == rn2.figure:
			correctEighths += 1
			s = 'C'
		else:
			pf1 = DT.parse_figure(rn1.figure)
			pf2 = DT.parse_figure(rn2.figure)
			if rn1.key.mode == rn2.key.mode and rn1.key.tonic == rn2.key.tonic:
				if pf1[0] == pf2[0] and pf1[2] == pf2[2] and pf1[3] == pf2[3]:
					if globalIgnoreInversions:
						correctEighths += 1
						continue
					else:
						inversionErrors += 1
				localChordErrors += 1
				wrongEighths += 1	
				confusionLabel = rn2.figure + ' for ' + rn1.figure
				confusionMatrix[confusionLabel] = confusionMatrix.setdefault(confusionLabel, 0) + 1
				confusions.setdefault(confusionLabel, []).append([pieceName, timePoint])
			else:
				"""rn3 = anal1.getElementsByOffset(int(timePoint) - 1, int(timePoint) - 1, includeEndBoundary = True, mustBeginInSpan = False, classList = ['RomanNumeral'])
				rn4 = anal1.getElementsByOffset(int(timePoint) + 1, int(timePoint) + 1, includeEndBoundary = True, mustBeginInSpan = False, classList = ['RomanNumeral'])
				if rn3 and rn4: 
					rn3 = rn3[0]
					rn4 = rn4[0]"""
				if sorted(pitchNames1) == sorted(pitchNames2):
					localWrongKeyRightChord += 1
				wrongEighths += 1
				localKeyErrors += 1
			s = 'I'
		lastRN1 = rn1
		lastRN2 = rn2
		#print rn1.pitches, rn2.pitches, rn1.pitches == rn2.pitches
		#print DT.parse_figure(rn1.figure)[0], DT.parse_figure(rn2.figure)[0], DT.parse_figure(rn1.figure)[0] == DT.parse_figure(rn2.figure)[0]
		if printComparison: 
			print_out(' '.join([str(x) for x in ['%i %.2f' % DT.get_measure_number_and_beat_from_offset(analyzed_file1, timePoint), s, keystring(rn1), rn1.figure, keystring(rn2), rn2.figure, pitchNames1, pitchNames2]]))
	pct = 100. * correctEighths/(wrongEighths + correctEighths)
	wrongKey = (100. * localKeyErrors / (wrongEighths + correctEighths))
	wrongChord = (100. * localChordErrors / (wrongEighths + correctEighths))
	inversionErrorPct = (100. * inversionErrors / (wrongEighths + correctEighths))
	totalCorrectEighths += correctEighths
	#print correctEighths, wrongEighths, '%.2f' % pct
	if _PRINTINDIVIDUAL:
		print_out('Wrong Key %.2f' % wrongKey + ' Wrong Chord %.2f' % wrongChord)
	keyErrors += localKeyErrors
	chordErrors += localChordErrors
	wrongKeyRightChord += localWrongKeyRightChord
	return [pct, wrongKey, wrongChord]


def confusion(n = 10):
	global confusionMatrix
	c = sorted(confusionMatrix.items(), key = lambda x: -x[1])
	for i in c[:n]:
		print(i)

def show_confusion(i):
	global confusions
	confusionItems = sorted(confusions.items(), key = lambda x:-len(x[1]))
	pprint(confusionItems[i])


def keystring(RN):
	outStr = RN.key.tonic.name + ':'
	if RN.key.mode == 'minor': outStr = outStr.lower()
	return outStr

def print_out(storedString, end = '\n'):
	if globalSave:
		outputFile.write(storedString + '\n')
	print(storedString, end = end)

def print_report(top_n_confusions=10):
    """
    Pretty-print all evaluation statistics after running dochorale().
    Requires dochorale() to have been run already, so that the globals
    are populated.
    """

    print("\n" + "="*60)
    print("                  ROMAN NUMERAL COMPARISON REPORT")
    print("="*60)

    # ----- Basic correctness -----
    total_eighths = totalCorrectEighths + keyErrors + chordErrors
    overall_accuracy = 100 * totalCorrectEighths / total_eighths if total_eighths else 0
    key_accuracy = 100 * (1 - keyErrors / total_eighths) if total_eighths else 0
    chord_accuracy = 100 * (1 - chordErrors / total_eighths) if total_eighths else 0

    print(f"Overall Accuracy:                {overall_accuracy:5.2f}%")
    print(f"Key Accuracy:                    {key_accuracy:5.2f}%")
    print(f"Chord Accuracy (given key):      {chord_accuracy:5.2f}%")
    print(f"Inversion Error %:               {inversionErrorPct:5.2f}%")
    print(f"Wrong-key-but-right-chord:       {wrongKeyRightChord} cases")
    print(f"Total Eighth Notes Compared:     {total_eighths}")

    print("-"*60)

    # ----- Confusion breakdown -----
    if confusionMatrix:
        print(f"Top {top_n_confusions} Confusions:")
        sorted_conf = sorted(confusionMatrix.items(), key=lambda x: -x[1])
        for label, count in sorted_conf[:top_n_confusions]:
            print(f"  {label:<25}  {count} times")
    else:
        print("No confusions recorded.")

    print("="*60 + "\n")
