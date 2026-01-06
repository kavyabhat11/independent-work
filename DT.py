




from __future__ import print_function
import copy
import random
import itertools
import math
import time
import subprocess, os
from music21 import *
import pickle

"""
================================================================================================================================

A centralized collection of utilities used by different programs.  Contains some redundancies.

SETUP: 

	You need to change the specific directories _MAINPATH and _PALCORPUS.  _ENGLISHWORDS is optional.  
	Many programs expect you to have folders _MAINPATH/Music/ and _MAINPATH/Music Data/

	You can do set things up manually or automatically 
		1. open a terminal window
		2. change  directory to the location of DT.py
		3. type python setup.py . (to make that your _MAINPATH) 
			or python setup.py DESIREDPATH
			This will set the relevant directories in this file and create _MAINPATH/Music/ and _MAINPATH/Music Data/ directories if they are not there.

	For each composer you should have _MAINPATH/Music/COMPOSERNAME/.  Inside the _COMPOSERNAME directory you will have a collection of files

		SCOREFILE1.xml (or krn or mxl)
		SCOREFILE1.txt (in romantext format)

	Many programs will iterate through the composer directories looking at score and/or analysis files.

	You are being given access to this code on the condition you not make fun of my programming! 
		I was learning python as I wrote this stuff and there is a lot in here that I would do differently now.

================================================================================================================================

"""

_MAINPATH = '/Users/dmitri/Source Code/Python/'
_PALCORPUS = '/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/music21/corpus/palestrina/'
_ENGLISHWORDS = '/usr/share/dict/words'			# optional, not needed

"""
================================================================
				Basic classes and data structures	
================================================================
"""


birthyears = {'Dufay': 1397,
'Ockeghem': 1410,
'Josquin': 1450, 
'Encina': 1468, 
'Frottola': 1470,
'Dalza': 1478,
'Sermisy': 1490, 
'Willaert': 1490, 
'Willaert2': 1490, 
'Gombert': 1495, 
'Tallis': 1505, 
'Tallis2': 1505, 
'Clemens': 1510, 
'Goudimel': 1514,
'Rore': 1515, 
'Palestrina': 1526, 
'Lassus': 1532, 
'Victoria': 1548, 
'Cavalieri': 1550, 
'Gastoldi': 1554,
'Morley': 1557, 
'Monteverdi': 1567, 
'Schutz': 1585, 
'Castello': 1590, 
'Merula': 1594, 
'Cavalli': 1602, 
'Lully': 1632, 
'Corelli': 1653, 
'Bach': 1685, 
'Haydn': 1732, 
'Mozart': 1756, 
'Beethoven': 1770, 
'Beethoven Quartets': 1770, 
'Mendelssohn': 1809,
'Chopin': 1810, 
'Brahms': 1833, 
'Rock': 1940, 
'Bill': 1945} 

comps = ['Dufay', 'Ockeghem', 'Josquin', 'Encina', 'Frottola', 'Dalza', 'Gombert', 'Willaert', 'Willaert2', 'Sermisy', 'Tallis', 'Tallis2', 'Clemens', 'Rore', 'Goudimel', 'Palestrina','Lassus','Victoria',  'Cavalieri', 'Gastoldi', 'Morley', 'Monteverdi', 'Schutz', 'Castello', 'Merula','Cavalli', 'Lully', 'Corelli', 'Bach', 'Haydn', 'Mozart', 'Beethoven', 'Beethoven Quartets', 'Mendelssohn','Chopin', 'Brahms', 'Rock', 'Bill']

comps = sorted(comps, key = lambda x: birthyears[x])

_NEWMAN = '/Users/dmitri/Source Code/Python/Newman/DATA/'		# DELETE THIS 

_MUSICPATH = _MAINPATH + 'Music/'
_DATAPATH = _MAINPATH + 'Music Data/'

_BACHANALYSIS = _MUSICPATH + 'Bach Chorales/'
_MORLEYSCORES = _MUSICPATH + 'Morley/'
_PALESTRINASCORES = _MUSICPATH + 'Palestrina/'					
_REDUCEDCHORALEPATH = _MUSICPATH + 'Bach Chorales REDUCED/'
_XMLCHORALES = _MUSICPATH + 'Bach Chorales XML/'
_PDFPATH = _MUSICPATH + 'Bach Chorales (Riemenschneider PDF)/'

composerPaths = {'BACH': [ _BACHANALYSIS, _XMLCHORALES], 
				'BACH CHORALES REDUCED': [_BACHANALYSIS, _REDUCEDCHORALEPATH], 
				'BACH CHORALES RAWMACHINE': [ _MUSICPATH + 'Bach Chorales RAWMACHINE/', _XMLCHORALES], 
				'BACH CHORALES CLEANMACHINE': [ _MUSICPATH + 'Bach Chorales CLEANMACHINE/', _XMLCHORALES],
				'BACH CHORALES HMM': [ _MUSICPATH + 'Bach CHORALES HMM/', _XMLCHORALES], 
				'BACH CHORALES HMM CLEAN': [ _MUSICPATH + 'Bach Chorales HMM Clean/', _XMLCHORALES], 
				'BACH CHORALES HMM RAW': [ _MUSICPATH + 'Bach Chorales HMM Raw/', _XMLCHORALES], 
				'BACH CHORALES HMM RAW CLEAN': [ _MUSICPATH + 'Bach Chorales HMM Raw Clean/', _XMLCHORALES], 
				'BEETHOVEN QUARTETS IAN': [_MUSICPATH + 'Beethoven Quartets Ian/', _MUSICPATH + 'Beethoven Quartets/'], 
				'PALESTRINA REDUCED': [_PALESTRINASCORES, _MUSICPATH + 'Palestrina Reduced/'], 
				'PALESTRINA IONIAN': [_MUSICPATH+'Palestrina Ionian/', _PALESTRINASCORES], 
				'PALRAW': [None, _PALCORPUS],
				'MORLEY REDUCED': [_MUSICPATH + 'Morley Reduced/', _MORLEYSCORES], 
				'MONTEVERDI REDUCED': [_MUSICPATH + 'Monteverdi Reduced/', _MUSICPATH + 'Monteverdi/'], 
				'JOSQUIN REDUCED': [_MUSICPATH + 'Josquin Reduced/',  _MUSICPATH + 'Josquin/'], 
				'SH1M': [_MUSICPATH + 'Sacred Harp/group-1/major/',_MUSICPATH + 'Sacred Harp/group-1/major/']
				}

class ComposerList(dict):
	
	"""Used for file management: returns a default directory path (Music/X) if the path is not overridden in composerPaths"""
	
	def __contains__(self, item):
		item = item.upper()
		for key in self:
			if key is item:
				return True
		else:
			return True
			
	def __getitem__(self, item):
		global composerPaths
		item = item.upper()
		if item in composerPaths:
			return composerPaths[item]
		else:
			return [_MUSICPATH + item[0] + item[1:].lower() + '/', _MUSICPATH + item[0] + item[1:].lower() + '/']
			
class VLLabel():
	
	"""label voiceleadings"""
	
	def __init__(self, startPos = [60, 64, 67], endPos = [67, 64, 60], transpositionRegion = False, inversionRegion = False, modulus = 12):
		self.lastNotes = startPos
		self.currentNotes = endPos
		self.modulus = modulus
		self.transpositionRegion = transpositionRegion
		self.invert = inversionRegion
		self.identify_voiceleading()
	
	def identify_voiceleading(self):
		
		self.get_crossing_times()
		
		self.set1 = [x % self.modulus for x in self.lastNotes]
		self.gnf1 = geometrical_normal_form_local(self.lastNotes, invert = self.invert, modulus = self.modulus)
		self.chordNames1 = self.get_chord_element_names(self.set1, self.gnf1)
		self.crossingLabels = self.calculate_crossings()
		
		self.gnf2 = geometrical_normal_form_local(self.currentNotes, invert = self.invert, modulus = self.modulus)
		self.normal_form_VL = [self.gnf2[i] - self.gnf1[i] for i in range(len(self.lastNotes))]
		self.newSet = any(self.normal_form_VL)
		
		self.adjustedChord1 = [self.lastNotes[i] + self.normal_form_VL[self.chordNames1[i]] for i in range(len(self.lastNotes))]
		self.avgSumDiff = (sum(self.currentNotes) - sum(self.adjustedChord1))/len(self.lastNotes)
		
		self.transposedChord1 = [self.adjustedChord1[i] + self.avgSumDiff for i in range(len(self.lastNotes))]
		
		self.scalarTransp = self.find_scalar_transposition(self.transposedChord1, self.currentNotes)
		self.VLstring = self.formulate_VL_string()
	
	def formulate_VL_string(self):
		
		vlString = ''
		if self.newSet:
			vlString = f"P{[self.normal_form_VL[self.chordNames1[i]] for i in range(len(self.lastNotes))]} ".replace('[', '(').replace(']', ')')
		if self.crossingLabels:
			if self.transpositionRegion:
				cString = [f't{x}C0' for x in self.transposition_region_crossings()]
				cString = ''.join(cString)[:-2]								# strip off last c0
				vlString += cString
			else:
				vlString += ''.join(['C'+str(x) for x in self.crossingLabels])				
		if self.scalarTransp and not self.transpositionRegion:
			vlString += f't{self.scalarTransp}'
		if self.avgSumDiff:
			vlString += f'T{self.avgSumDiff}'.replace('.0', '')
		return vlString
		
	def transposition_region_crossings(self):
		tList = []
		lastTransposition = 0
		for c in self.crossingLabels:
			tList.append((lastTransposition + c)%len(self.lastNotes))
			lastTransposition = -c
		tList.append((lastTransposition + self.scalarTransp) % len(self.lastNotes))
		return tList
	
	def find_scalar_transposition(self, startChord, endChord):
		endPCs = sorted([x % self.modulus for x in endChord])
		startPCs = sorted([x % self.modulus for x in startChord])
		avgSum = self.modulus/len(self.lastNotes)
		
		for i in range(len(self.lastNotes)):
			if all([startPCs[i] == endPCs[i] for i in range(len(self.lastNotes))]):
				return i
			startPCs = sorted([(x - avgSum)%self.modulus for x in startPCs])
		
		print("Problem finding scalar transposition")
		return None	
		
	def calculate_crossings(self):
		
		"""todo: translate C2 into t2c0t-2, etc., then consolidate adjacent t labels"""
		"""chordNames is temporary since we relabel as we go"""
	
		chordNames = {x[0]:x[1] for x in self.chordNames1.items()}
		lastCrossing = None
		crossingLabels = []
	
		for t in sorted(self.crossingTimes.keys()):
			#print(round(t, 2), crossingTimes[t])
			for v1, v2, distance in self.crossingTimes[t]:
				voice1 = chordNames[v1]
				voice2 = chordNames[v2]
			
				if len(self.lastNotes) != 2:
					if voice2 == (voice1 + 1) % len(self.lastNotes):
						lastCrossing = (-voice1)%len(self.lastNotes)
					elif voice1 == (voice2 + 1)%len(self.lastNotes):
						lastCrossing = (-voice2)%len(self.lastNotes)
					else:
						print('Problem labeling crossings:', voice1, voice2)
						continue
				else:
					if lastCrossing == None:
						if distance > self.modulus/2:
							lastCrossing = 1
						else:
							lastCrossing = 0
					else:
						lastCrossing = (lastCrossing + 1)%2
				crossingLabels.append(lastCrossing)
				#print(f"C{lastCrossing}")
					
				chordNames[v1] = voice2
				chordNames[v2] = voice1
			
		return crossingLabels
	
		
	def get_crossing_times(self):
		self.moveVector = [self.currentNotes[i] - self.lastNotes[i] for i in range(len(self.lastNotes))]
		self.crossingTimes = {}
	
		for firstVoice in range(len(self.lastNotes) - 1):
			for secondVoice in range(firstVoice + 1, len(self.lastNotes)):
				relativeVelocity = self.moveVector[secondVoice] - self.moveVector[firstVoice]
				distance = (self.lastNotes[secondVoice] - self.lastNotes[firstVoice]) % self.modulus
				if relativeVelocity > 0:
					distance = self.modulus - distance
				elif relativeVelocity == 0:
					continue
				vel = abs(relativeVelocity)
				crossingTime = distance/vel
				while crossingTime < 1:
					self.crossingTimes.setdefault(crossingTime, []).append((firstVoice, secondVoice, distance))
					crossingTime += self.modulus/vel
		
	def get_chord_element_names(self, PCs, gnf):
		"""identifies chord elements by their normal forms; fold back into nr.py?"""
		chordNames = {}
	
		transp = None
	
		for i in range(self.modulus):
			working = [(x + i) % self.modulus for x in gnf]
			if all([x in PCs for x in working]):
				transp = i
				break
	
		if transp is None:
			index = None
			for i in range(self.modulus):
				working = [(i - x) % self.modulus for x in gnf]
				if all([x in PCs for x in working]):
					index = i
					break
			
			if index is None:
				print("Problem labeling chord elements,", myChord)
				return {}
					
		for i, pc in enumerate(PCs):
			target = working.index(pc)
			chordNames[i] = target
			working[target] = None
	
		return chordNames
	
	def __str__(self):
		return self.VLstring
		
_PATHS = ComposerList()

_FIVEVOICECHORALES = [150]
_DUPLICATECHORALES = [9, 23, 53, 86, 91, 93, 120, 126, 144, 177, 195, 198, 201, 235, 254, 256, 295, 302, 305, 313, 328, 354]

_UNKNOWNCHORDS = {'It':'#4 6 1', 'Ger':'#4 6 1 3', 'Fr': '2 #4 6 1',
'Nmaj': 'b2 4 6 1', 'V9[b9]': '5 #7 2 4 6','bVIImaj':'b7 2 4 6','It6/ii': 'b7 2 b6'} 	 #It6/ii is only for chorale 216, kludgey

_DIATONICMAJOR = ['I', 'I6', 'I6/4', 'Imaj7', 'Imaj6/5', 'Imaj4/3', 'Imaj2', 
'ii', 'ii6', 'ii6/4', 'ii7', 'ii6/5', 'ii4/3', 'ii2', 'iii', 'iii6', 'iii6/4', 'iii7', 'iii6/5', 'iii4/3', 'iii2', 
'IV', 'IV6', 'IV6/4', 'IVmaj7', 'IVmaj6/5', 'IVmaj4/3', 'IVmaj2', 'V', 'V6', 'V6/4', 'V7', 'V6/5', 'V4/3', 'V2', 
'vi', 'vi6', 'vi6/4', 'vi7', 'vi6/5', 'vi4/3', 'vi2', 'viio', 'viio6', 'viio6/4', 'vii/o7', 'vii/o6/5', 'vii/o4/3', 'vii/o2']
#'viio7', 'viio6/5', 'viio4/3', 'viio2']	# these really aren't diatonic.  Why include them?
_DIATONICMINOR = ['i', 'i6', 'i6/4', 'i7', 'i6/5', 'i4/3', 'i2', 'iio', 'iio6', 'iio6/4', 'ii/o7', 'ii/o6/5', 'ii/o4/3', 'ii/o2', 'ii', 'ii6', 'ii6/4', 'ii7', 'ii6/5', 'ii4/3', 'ii2', 'III', 'III6', 'III6/4', 'IIImaj7', 'IIImaj6/5', 'IIImaj4/3', 'IIImaj2', 'III+', 'III+6', 'III+6/4', 'III+maj7', 'III+maj6/5', 'III+maj4/3', 'III+maj2', 'iv', 'iv6', 'iv6/4', 'iv7', 'iv6/5', 'iv4/3', 'iv2', 'IV', 'IV6', 'IV6/4', 'IV7', 'IV6/5', 'IV4/3', 'IV2', 'v', 'v6', 'v6/4', 'v7', 'v6/5', 'v4/3', 'v2', 'V', 'V6', 'V6/4', 'V7', 'V6/5', 'V4/3', 'V2', 'VI', 'VI6', 'VI6/4', 'VImaj7', 'VImaj6/5', 'VImaj4/3', 'VImaj2', 'vio', 'vio6', 'vio6/4', 'vi/o7', 'vi/o6/5', 'vi/o4/3', 'vi/o2', 'VII', 'VII6', 'VII6/4', 'VII7', 'VII6/5', 'VII4/3', 'VII2', 'viio', 'viio6', 'viio6/4', 'viio7', 'viio6/5', 'viio4/3', 'viio2']


_DIATONIC = [_DIATONICMAJOR, _DIATONICMINOR]

"""
================================================================
				Routines to open and load pieces
================================================================
"""

"""The most versatile of all of these"""
def piece(s, comp = ''):
	global c
	theFiles = ''
	if not comp:
		for comp in os.listdir(_MUSICPATH):
			#print comp, _MUSICPATH + comp
			try:
				myPath = _MUSICPATH + comp + '/'
				theFiles = os.listdir(myPath)
				theFiles = [x for x in theFiles if (x.upper().startswith(s.upper()) and x.upper().count('XML') + x.upper().count('KRN') > 0)]
				#if comp == 'Mozart': print theFiles
			except:
				pass
			if theFiles:
				break
	else:
		myPath = _PATHS[comp][1]
		theFiles = os.listdir(myPath)
	if theFiles:	
		c = converter.parse(myPath+theFiles[0])
		c.show()
		return c
	print ("not found")

palestrinaData = []

def palestrina_data():
	global palestrinaData
	with open(_DATAPATH + 'PALESTRINACORPUSDATA.pkl', 'rb') as myFile:
		palestrinaData = pickle.load(myFile)
	return palestrinaData

def find_mass(s, index = 0):
	global palestrinaData
	if not palestrinaData:
		palestrina_data()
	for i, v in palestrinaData.items():
		if v[index].count(s):
			print(i, v)

def get_palestrina(s, show = True):
	global output
	output = corpus.parse(s)
	if show:
		output.show()

def get_chorale(n):						# open the PDF of the original Riemenschneider
	subprocess.call(('open', _PDFPATH + 'riemenschneider' + str(n).zfill(3) +'.pdf'))

def get_analysis(n):						# open the analysis file
	if type(n) is int:
		subprocess.call(('open', _BACHANALYSIS + 'riemenschneider' + str(n).zfill(3) +'.txt'))
	else:
		subprocess.call(('open', n))

def get_both(n):
	get_chorale(n)
	get_analysis(n)

"""
UNIX AND WINDOWS USERES MAY NEED PLATFORM-INDEPENDENT CODE TO OPEN FILES:

import subprocess, os, sys
	if sys.platform.startswith('darwin'):
	    subprocess.call(('open', filepath))
	elif os.name == 'nt':
	    os.startfile(filepath)
	elif os.name == 'posix':
	    subprocess.call(('xdg-open', filepath))"""

def translate_chorales(n):
	s = str(n)
	c = corpus.chorales.ChoraleListRKBWV().byBWV
	for key in c:
		if key[0:len(s)] == s:
			print (key, c[key])

def beethoven_quartet(*args):
	if len(args) == 2:
		i, j = args
		n = 0
	elif len(args) == 3:
		i, n, j = args
	else:
		return
	pieces = {74: 10, 95: 11, 127: 12, 130: 13, 131: 14, 132: 15, 135: 16}
	if i == 18:
		n = n + 0
	elif i == 59:
		n = n + 6
	else:
		n = pieces[i]
	s = 'op' + str(i)+'_no' + str(n) + '_mov' + str(j)
	os.system("open -t " + s + '.txt')
	cmd = "/Applications/MuseScore\\ 3.app/Contents/MacOS/mscore " + s + '.mscx'
	os.system(cmd)
	return

def mazurka(s):
	return converter.parse('http://kern.ccarh.org/cgi-bin/ksdata?file=mazurka' + s + '.krn&l=users/craig/classical/chopin/mazurka&format=kern')

def maz(s):
	converter.parse('http://kern.ccarh.org/cgi-bin/ksdata?file=mazurka' + s + '.krn&l=users/craig/classical/chopin/mazurka&format=kern').write('musicxml', _MUSICPATH + 'Chopin/mazurka' + s + '.xml')

def invention(i):
	converter.parse('http://kern.ccarh.org/cgi-bin/ksdata?location=osu/classical/bach/inventions&file=inven' + str(i).zfill(2) + '.krn&format=kern').write('musicxml', _MUSICPATH + 'BachInventions/no' + str(i) + '.xml')

def josquin(fName = None):
	import requests
	downloadedFiles = []
	outName = _PATHS['Josquin New'][0]
	fNames = sorted(os.listdir(_PATHS['Josquincomplete'][0]))
	fNames = [x.replace('.xml', '').replace('.krn', '') for x in fNames if x.count('krn') or x.count('xml')]
	if fName:
		fNames = [fName]
	prefix = 'https://josquin.stanford.edu/data?a=musicxml&f='
	for f in fNames:
		if f not in downloadedFiles:
			print(f)
			downloadName = prefix+f
			r = requests.get(downloadName, allow_redirects = True)
			open(outName + f + '.xml', 'wb').write(r.content)
				
def ockeghem():
	import requests
	downloadedFiles = []
	outName = _PATHS['Ockeghem New'][0]
	fNames = sorted(os.listdir(_PATHS['ockeghem'][0]))
	fNames = [x.replace('.xml', '').replace('.krn', '') for x in fNames if x.count('krn') or x.count('xml')]
	prefix = 'https://josquin.stanford.edu/data?a=musicxml&f='
	for f in fNames:
		if f not in downloadedFiles:
			print(f)
			downloadName = prefix+f
			r = requests.get(downloadName, allow_redirects = True)
			open(outName + f + '.xml', 'wb').write(r.content)
	
	

"""
================================================================
				Math stuff
================================================================
"""

def add_vectors(a, b):
	return [a[i] + b[i] for i in range(len(a))]

def euclidean_distance(l1, l2):
	return pow(sum([(l2[i] - l1[i])**2 for i in range(len(l1))]), .5)

def setclass_distance(l1, l2):
	l = len(l1)
	l1 = zeroplane(l1)
	l2 = zeroplane(l2)
	dist = [l2[i] - l1[i] for i in range(l)]
	return pow(sum([x**2 for x in dist]), .5)
	
def zeroplane(x):
	return [i - 1.0*sum(x)/len(x) for i in x]

def frequency_to_midi(f):
	return 69 + 12.0 * math.log(f/440., 2)
	
ftom = frequency_to_midi
	
def midi_to_frequency(m):
	return 440.*pow(2, (m - 69)/12.)
	#return 69 + 12.0 * math.log(f/440., 2)

mtof = midi_to_frequency

def interval_to_ratio(m):
	return pow(2, m/12.)
	#return 69 + 12.0 * math.log(f/440., 2)

def digit(number, digit, base = 10):
	return int(number/(base**(digit - 1))) % base
	
def get_digits(number, digits, base = 10):
	return [int(number/(base**(digit - 1))) % base for digit in range(digits, 0, -1)]

def linear_map(value, firstRange, secondRange):
	pct = (value - firstRange[0])/(firstRange[1] - firstRange[0])
	output = secondRange[0] + pct*(secondRange[1] - secondRange[0])
	return output

def get_linear_map(firstRange, secondRange, returnFunction = True):
	zeroValue = linear_map(0, firstRange, secondRange)
	oneValue = linear_map(1, firstRange, secondRange)
	if returnFunction:
		return lambda x: (oneValue - zeroValue)*x + zeroValue
	return oneValue - zeroValue, zeroValue					# y = mx + b

def scale(x = 0, inputRange = [0, 1], outputRange = [0, 1], power = 1):
	
	""" 
	copied from the Max/MSP scale object; power gives the curve
		> 1 = concave down
		1 = linear
		2 = concave up
	
	DTgraph.graph([DT.scale(i/n, power = e) for i in range(0, n + 1)])
	
	"""
	
	in_low, in_high = inputRange
	out_low, out_high = outputRange
	inSpan = in_high-in_low
	inPos = (x - in_low)/inSpan
	if inPos > 0:
		return (out_low + (out_high-out_low) * ((x-in_low)/inSpan)**power) 	
	elif inPos < 0:
		return -(out_low + (out_high-out_low) * ((((-x+in_low)/inSpan))**(power)))
	return out_low

"""
	random numbers weighted toward the lower end of the range
	as startpoint increases, the distribution flattens out
	
	I have included a commented-out function that will print the distribution;
	try DTgraph.graph(get_distribution(f))
	
	can probably do this with scale
	
"""
def reciprocal_distribution(low = 0, high = 100, startPoint = 1):
	lowerval = math.log(low + startPoint)
	upperval = math.log(high + startPoint)
	return math.exp(random.uniform(lowerval, upperval)) - startPoint

def reciprocal_distribution_function(low = 0, high = 100, startPoint = 1):
	lowerval = math.log(low + startPoint)
	upperval = math.log(high + startPoint)
	return lambda: math.exp(random.uniform(lowerval, upperval)) - startPoint

def get_distribution(myFunc, bins = 100, trials = 100000, printOut = True):
	rawList = []
	for i in range(trials):
		rawList.append(myFunc())
	high = max(rawList)
	low  = min(rawList)
	results = {}
	binSize = (high - low)/bins
	for r in rawList:
		intR = int((r - low)/binSize)
		results[intR] = results.get(intR, 0) + 1
	total = sum(results.values())
	finalList = [results.get(i, 0)*100./total for i in range(bins)]
	if printOut:
		for i, v in enumerate(finalList):
			print('Bin', i, f'{round(v, 1)}%')
	return finalList

def weighted_choice(d):
	choices = d.items()
	total = sum(w for c, w in choices)
	if not total:
		return 0
	r = random.uniform(0, total)
	upto = 0
	for c, w in choices:
		if upto + w >= r:
			return c
		upto += w
	assert False, "Shouldn't get here"

def dict_entropy(myDict):			# also works on lists containing [key, value] pairs
	if type(myDict) is list:
		tempDict = {}
		for item in myDict:
			tempDict[item[0]] = item[1]
		myDict = tempDict
	tempSum = 1.0 * sum(myDict.values())
	entropy = 0
	for myProb in myDict.values():
		if myProb == 0: continue
		myProb = myProb/tempSum
		entropy -= myProb*math.log(myProb, 2)
	return entropy

def mutual_information(myDict, splitString = ' -> '):				# myDict should have keys of the form "firstUnit[splitStr]secondUnit"
	marginalProb = {}												# also works on lists containing [key, value] pairs
	if type(myDict) is list:
		tempDict = {}
		for item in myDict:
			tempDict[item[0]] = item[1]
		myDict = tempDict
	for item in myDict:
		firstChord, secondChord = item.split(splitString)
		marginalProb[firstChord] = marginalProb.setdefault(firstChord, 0) + myDict[item]
		marginalProb[secondChord] = marginalProb.setdefault(secondChord, 0) + myDict[item]
	marginalSum = 1.0 * sum(marginalProb.values())
	for theChord in marginalProb:
		marginalProb[theChord] = marginalProb[theChord] / marginalSum
	tempSum = 1.0 * sum(myDict.values())
	mutualInformation = 0
	for myProg in myDict:
		theChords = myProg.split(splitString)
		myProb = myDict[myProg]/tempSum
		if myDict[myProg] == 0: continue
		mutualInformation += myProb*math.log(myProb / (marginalProb[theChords[0]] * marginalProb[theChords[1]]), 2)
	return mutualInformation

"""manipulating permutations"""

def get_permutation(s): 
	return [[int(x) - 1 for x in y.replace('(', '')] for y in s.split(')') if y]

def permute(cycle = '(12)(36)(45)', s = ['C', 'c', 'E', 'gs', 'Af', 'e']):
	if type(cycle) is str: cycle = get_permutation(cycle)
	out = s[:]
	#print(cycle, out)
	for perm in cycle:
		#print(' ', perm)
		for i, pos in enumerate(perm):
			newPos = perm[(i+1) % len(perm)]
			out[newPos] = s[pos]
	return out

def identify_permutation(list1, list2, returnString = False):
	cycles = []
	foundObjects = []
	for index, item in enumerate(list1):
		if item in foundObjects: continue
		currentCycle = []
		while index not in currentCycle:
			if item not in list2:
				return False
			currentCycle.append(index)
			foundObjects.append(item)
			newIndex = list2.index(item)
			if newIndex in currentCycle:
				cycles.append(currentCycle)
				currentCycle = []
				break
			item = list1[newIndex]
			index = newIndex
	if not returnString:
		return cycles
	newC = [''.join([str(x+1) for x in y]) for y in cycles]
	return '(' + ')('.join(newC) + ')'

"""
================================================================
				Formatting routines
================================================================
"""

class Timer():
	"""simple routine to time longer programs"""
	
	def __init__(self, *args, **kwargs):
		self.reset()
	
	def reset(self):
		self.startTime = time.time()
		
	def get_time(self):
		curTime = time.time()
		self.hours = int((curTime - self.startTime)/3600)
		self.minutes = int((curTime - self.startTime) / 60) % 60
		self.seconds = int(curTime - self.startTime) % 60
		#print(self.hours, self.minutes, self.seconds)
		
	def string_time(self):
		self.get_time()
		if self.hours:
			return f"{self.hours} hours, {self.minutes} minutes, and {self.seconds} seconds"
		elif self.minutes:
			return f"{self.minutes} minutes and {self.seconds} seconds"
		return f"{self.seconds} seconds"
	
	def short_string_time(self):
		self.get_time()
		if self.hours:
			return f"{self.hours}h {self.minutes}m {self.seconds}s"
		elif self.minutes:
			return f"{self.minutes}m {self.seconds}s"
		return f"{self.seconds}s"
	
	def print_out(self, prefix = 'Analysis took '):
		self.get_time()
		print(prefix + f"{int(self.minutes + .5)} minutes and {self.seconds} seconds.")

def sort_dict(myDict):
	return sorted(myDict.items(), key=lambda x: -x[1])

def sum_dict(d):
	return sum(d.values())

def print_dict(myDict, pct = False, top = 10, cutoff = 0, digits = 1):
	theSum = sum(myDict.values())
	newList = []
	print(theSum)
	if pct:
		for i, item in enumerate(sort_dict(myDict)):
			pctStr = str(round((100.0 * item[1])/theSum, digits))
			if pctStr == '0.0': continue
			if pctStr.endswith('.0'): pctStr = pctStr.replace('.0', '')
			if cutoff and i > cutoff: break
			print(str(str(item[0]) + ' ' + pctStr))
			newList.append([item[0], pctStr])
	else:
		for i, item in enumerate(sort_dict(myDict)):
			if cutoff and i > cutoff: break
			print (item[0], item[1])
			newList.append(item)
	try:
		if top:
			total = 0
			for i in range(0, min(len(myDict), top + 1)):
				if type(newList[i][1]) is str:
					total += float(newList[i][1])
				else:
					total += newList[i][1]
			print (total)
	except:
		pass

"""Convert chord or list of pitches to a readable string"""

def nice_name(listOfPitches):
	if type(listOfPitches) is not list:								# for chords
		listOfPitches = listOfPitches.pitches
	return ' '.join([x.nameWithOctave for x in listOfPitches])

"""convert readable string to list of pitches"""
def pitchlist(myList):
	if type(myList) == str: myList = myList.split()
	return [pitch.Pitch(x) for x in myList]

def float_string(f, decimals = 2): 
	return ('%.' + str(decimals) + 'f') % f

def key_name(myKey):
	basicName = myKey.tonic.name.replace('-', 'b')
	if myKey.mode == 'minor':
		basicName = basicName.lower()
	return basicName

def scale_degree_string_from_pitch(myPitch, myKey):
	degreePair = myKey.getScaleDegreeAndAccidentalFromPitch(myPitch)
	sdStr = str(degreePair[0])
	if degreePair[1]:
		accidentalSym = '+'
		i = int(degreePair[1].alter)
		if i < 0:
			accidentalSym = '-'
			i = i * -1
		sdStr = (accidentalSym * i) + sdStr
	return sdStr

def unpack_scale_degree(degStr):
	if type(degStr) == int:
		return degStr, 0
	modifier = degStr.count('#') - degStr.count('b') - degStr.count('-')
	degStr = degStr.lstrip('#b-')
	return int(degStr), modifier

def pc_from_scale_degree(degStr, myKey):
	sd = unpack_scale_degree(degStr)
	return (myKey.pitches[(sd[0] - 1) % (len(myKey.pitches) - 1)].midi + sd[1]) % 12

letterPCs = {'C': 0, 'D': 2, 'E':4, 'F':5, 'G':7, 'A':9, 'B':11}
def parse_lettername(s, lastOctaveSymbol = -1):
	s = s.upper()
	letterName = None
	for i, c in enumerate(s):
		letterName = letterPCs.get(c, None)
		if letterName != None:
			break
	if letterName is None:
		return parse_number(s)
	newS = s[i+1:]
	octaveSymbols = ''.join([x for x in newS if x.isdigit()])
	if not octaveSymbols:
		octave = lastOctaveSymbol
	else:
		octave = int(octaveSymbols)
	modifier = -1*(newS.count('B') + newS.count('-')) + (newS.count("#"))
	return letterName + modifier + (octave + 1)*12
	
def parse_number(s):
	if s.count('.') > 0:
		firstDot = s.index('.')
		newS = s[:firstDot] + '.' + s[firstDot+1:].replace('.','')
		return float(''.join([x for x in newS if x.isdigit() or x in '.-'])) 
	elif s.count('/') > 0:
		firstSlash = s.index('/')
		return float(s[:firstSlash])/float(s[firstSlash+1:].replace('/',''))
	else:
		digitChars = ''.join([x for x in s if x.isdigit() or x == '-'])
		if digitChars:
			return int(digitChars)
		return None
		
"""Save clipboard data as Max coll file"""

def maxcoll():
	if 'xerox' not in globals():
		globals()['xerox'] = __import__('xerox')
	outDict = {}
	s = xerox.paste()
	theLines = s.split('\r')
	for l in theLines:
		if not l:
			continue
		i = l.index(',')
		address = l[:i]
		value = l[i+1:-1].split()
		if all([x.isdigit() for x in address]): address = int(address)
		outList = []
		for item in value:
			if not item:
				continue
			if all([x.isdigit() for x in item]): item = int(item)
			outList.append(item)
		outDict[address] = outList
	return outDict

"""
================================================================
				Theory stuff
================================================================	
"""

# A simple chromatic spelling algorithm that works pretty well
# see "spelling.py" in the Theory directory

class ChromaticSpeller():
	lowercaseLetters = 'cdefgab'
	
	#pc: [((lettername, alteration), cost)]
	possibleSpellings = {	0: [((6, 1), 7), ((0, 0), 0)], 1: [((0, 1), 2), ((1, -1), 4)], 2: [((1, 0), 0)], 3: [((1, 1), 4), ((2, -1), 2)], 
							4: [((2, 0), 0), ((3, -1), 7)], 5: [((2, 1), 6), ((3, 0), 0)], 6: [((3, 1), 1), ((4, -1), 5)], 7: [((4, 0), 0)], 
							8: [((4, 1), 3), ((5, -1), 3)], 9: [((5, 0), 0)], 10: [((5, 1), 5), ((6, -1), 1)], 11: [((6, 0), 0), ((0, -1), 6)]}
	
	# excludes respelled white notes
	sharpSpellings_black = {	0: ((0, 0), 0), 1: ((0, 1), 2), 2: ((1, 0), 0), 3: ((1, 1), 4), 4: ((2, 0), 0), 5: ((3, 0), 0), 
								6: ((3, 1), 1), 7: ((4, 0), 0), 8: ((4, 1), 3), 9: ((5, 0), 0), 10: ((5, 1), 5), 11: ((6, 0), 0)}

	flatSpellings_black = {	0: ((0, 0), 0), 1: ((1, -1), 4), 2: ((1, 0), 0), 3: ((2, -1), 2), 4: ((2, 0), 0), 5: ((3, 0), 0), 
							6: ((4, -1), 5), 7: ((4, 0), 0), 8: ((5, -1), 3), 9: ((5, 0), 0), 10: ((6, -1), 1), 11: ((6, 0), 0)}

	# respelled white notes
	sharpSpellings_white = {0: ((6, 1), 7), 5: ((2, 1), 6)}
	
	flatSpellings_white = {4: ((3, -1), 7), 11: ((0, -1), 6)}
				
	alterations = {	(0, 6): 1, (0, 0): 0, (1, 0): 1, (1, 1): -1, (2, 1): 0, (3, 1): 1, (3, 2): -1, (4, 2): 0, (4, 3): -1, (5, 2): 1, (5, 3): 0, 
					(6, 3): 1, (6, 4): -1, (7, 4): 0, (8, 4): 1, (8, 5): -1, (9, 5): 0, (10, 5): 1, (10, 6): -1, (11, 6): 0, (11, 0): -1}

	# keys sorted by greatest difference between the two possible spellings, values sorted by cheapest spelling
	spellingsByCost = {	0: [((0, 0), 0), ((6, 1), 7)], 4: [((2, 0), 0), ((3, -1), 7)], 5: [((3, 0), 0), ((2, 1), 6)], 11: [((6, 0), 0), ((0, -1), 6)], 
						6: [((3, 1), 1), ((4, -1), 5)], 10: [((6, -1), 1), ((5, 1), 5)], 1: [((0, 1), 2), ((1, -1), 4)], 
						3: [((2, -1), 2), ((1, 1), 4)], 8: [((4, 1), 3), ((5, -1), 3)]}

	costs = {(6, 1): 7, (0, 1): 2, (1, -1): 4, (1, 1): 4, (2, -1): 2, (3, -1): 7, (2, 1): 6, (3, 1): 1, (4, -1): 5, (4, 1): 3, (5, -1): 3, (5, 1): 5, (6, -1): 1, (0, -1): 6}
		
	fNotes = set([2, 7, 9])				# notes with one spelling
	problemDists = {(1, 0): 1, (3, 1): 0.25, (2, 2): 2, (4, 3): 2, (4, 1): 2, (5, 2): 2, (7, 5): 2, (7, 3): 2} #, (8, 4): 2}	# intervals to avoid (chromaticDist, letterDist)
	
	defaultVexflowSpelling = {0: 'c', 1: 'c#', 2: 'd', 3: 'eb', 4: 'e', 5: 'f', 6: 'f#', 7: 'g', 8: 'g#', 9: 'a', 10: 'bb', 11: 'b'}
	
	def __init__(self, pitches = [], respell = False, spellingType = None, transpositions = [], naturals = False, defaultDirection = None):
		self.mixingAccidentalsPenalty = 1
		self.respellWhiteNotePenalty = 2
		
		self.defaultDirection = defaultDirection
		self.respell = respell
		
		self.courtesyNaturals = int(naturals)
		self.spellingDict = {}
		
		if pitches:
			self.get_spelling(pitches)
		
	def get_spelling(self, PClist):
		
		self.PCs = set([x % 12 for x in PClist])
		self.sortedPCs = sorted(self.PCs)
		
		self.intervals = []
		self.possibleLetters = []
		
		for i, pc in enumerate(self.sortedPCs):
			self.possibleLetters.append([pc, set([x[0][0] for x in ChromaticSpeller.possibleSpellings[pc]])])
		
		self.currentSpelling = {x:ChromaticSpeller.possibleSpellings[x][0][0] for x in self.PCs.intersection(ChromaticSpeller.fNotes)}
		self.currentCost = 0
		
		if len(self.PCs.intersection({4, 5})) == 2:
			self.currentSpelling[4] = (2, 0)
			self.currentSpelling[5] = (3, 0)
		if len(self.PCs.intersection({0, 11})) == 2:
			self.currentSpelling[0] = (0, 0)
			self.currentSpelling[11] = (6, 0)
		if len(self.PCs.intersection({2, 3, 4})) == 3:					# TODO: replace with better treatment of chromatic clusters!!!
			self.currentSpelling[2] = (1, 0)
			self.currentSpelling[4] = (2, 0)
		if len(self.PCs.intersection({9, 10, 11})) == 3:				# TODO: replace with better treatment of chromatic clusters!!!
			self.currentSpelling[9] = (5, 0)
			self.currentSpelling[11] = (6, 0)
			
		self.update_possibilities()
		
		notesToSpell = [x for x in self.sortedPCs if x not in self.currentSpelling]
		minSpelling = self.minimum_spelling(notesToSpell)
		
		if minSpelling:
			self.problemScore = self.check_for_problem_intervals(minSpelling)
			if self.problemScore: 
				newSpelling = self.fix_problem_intervals(minSpelling)
				if newSpelling:
					minSpelling = newSpelling
			self.spellingDict = minSpelling
			self.make_vexflow_dict()
		else:
			print("FAILED TO SPELL!", self.PCs)
	
	def minimum_spelling(self, notesToSpell):
		tempSpelling = {x[0]:x[1] for x in self.currentSpelling.items()}
		currentLetters = {x[0] for x in tempSpelling.values()}
		for pc, spellings in ChromaticSpeller.spellingsByCost.items():
			if pc not in notesToSpell: continue			
			success = False
			for spelling in spellings:
				spelling, cost = spelling
				if spelling[0] not in currentLetters:
					tempSpelling[pc] = spelling
					self.currentCost += cost
					currentLetters.add(spelling[0])
					success = True
					break
			if not success:
				spelling, cost = spellings[0]
				tempSpelling[pc] = spelling
				self.currentCost += cost
		return tempSpelling
	
	def check_for_problem_intervals(self, tempSpelling):
		problemScore = 0
		for i, p in enumerate(self.sortedPCs):
			lowerNote = self.sortedPCs[i-1]
			upperNote = self.sortedPCs[(i+1)%len(self.sortedPCs)]
			lowerLetter = tempSpelling[lowerNote][0]
			currentLetter = tempSpelling[p][0]
			chromaticDistance = (p - lowerNote)%12
			letterDistance = (currentLetter - lowerLetter)%7
			d = (chromaticDistance, letterDistance)
			if not (lowerNote == (p - 1) % 12 and upperNote == (p + 1) % 12):						# don't count notes in chromatic clusters
				problemScore += ChromaticSpeller.problemDists.get(d, 0)									
		return problemScore
	
	def fix_problem_intervals(self, tempSpelling):
		"""first respell black notes only"""
		sharpSpelling, sharpCost = self.respell_spelling(tempSpelling, ChromaticSpeller.sharpSpellings_black)
		sharpProblem = self.check_for_problem_intervals(sharpSpelling)
		flatSpelling, flatCost = self.respell_spelling(tempSpelling, ChromaticSpeller.flatSpellings_black)
		flatProblem = self.check_for_problem_intervals(flatSpelling)
		
		"""if that doesn't work, try white notes as well"""					
		if sharpProblem and flatProblem:
			#print('here')
			sharpSpelling2, sharpCost2 = self.respell_spelling(sharpSpelling, ChromaticSpeller.sharpSpellings_white)
			sharpProblem2 = self.check_for_problem_intervals(sharpSpelling2)
			if sharpProblem2 + self.respellWhiteNotePenalty < self.problemScore: 
				sharpSpelling = sharpSpelling2
				sharpCost += sharpCost2
				sharpProblem = sharpProblem2
			flatSpelling2, flatCost2 = self.respell_spelling(flatSpelling, ChromaticSpeller.flatSpellings_white)
			flatProblem2 = self.check_for_problem_intervals(flatSpelling2)
			if flatProblem2 + self.respellWhiteNotePenalty < self.problemScore: 
				flatSpelling = flatSpelling2
				flatCost += flatCost2
				flatProblem = flatProblem2
		
		"""print(sharpSpelling)
		print('  ', sharpCost, sharpProblem)
		
		print(flatSpelling)
		print('  ', flatCost, flatProblem)"""
		
		# TODO: allow for more fine-grained control here, user-specifiable function?
		
		if (sharpProblem < flatProblem or sharpProblem == flatProblem and self.defaultDirection == 1) and sharpProblem < self.problemScore:
			return sharpSpelling
		elif (flatProblem < sharpProblem or flatProblem == sharpProblem and not self.defaultDirection) and flatProblem < self.problemScore:
			return flatSpelling
		else:
			if self.currentCost < sharpCost and self.currentCost < flatCost or (self.currentCost == sharpCost == flatCost and self.defaultDirection == None):
				return None
			if sharpCost < flatCost or (sharpCost == flatCost and self.defaultDirection == 1):
				return sharpSpelling
			else:
				return flatSpelling
	
	def respell_spelling(self, tempSpelling, spellingDict):
		cost = 0
		d = {}
		for p, data in tempSpelling.items():
			spell, c = spellingDict.get(p, [data, 0])
			d[p] = spell
			cost += c
		return d, cost
	
	def spelling_as_string(self, spelling, spellingString = 'CDEFGAB'):
		sortedSpelling = sorted(spelling.items())
		final = []
		for pc, spelling in sortedSpelling:
			final.append(self.tuple_to_string(spelling, spellingString))
		return ' '.join(final)
	
	def tuple_to_string(self, spellingTuple, spellingString = 'CDEFGAB'):
		letter = spellingString[spellingTuple[0]]
		if not spellingTuple[1]:
			acci = 'n'*self.courtesyNaturals
		elif spellingTuple[1] > 0:
			acci = '#'*spellingTuple[1]
		elif spellingTuple[1] < 0:
			acci = 'b'*-spellingTuple[1]
		return letter + acci
	
	def print_spelling(self, spelling):
		print(self.spelling_as_string(spelling))
	
	def update_possibilities(self):
		redo = False
		usedLetters = set([x[0] for x in self.currentSpelling.values()])
		for i, data in enumerate(self.possibleLetters):
			pc, s = data
			if pc not in self.currentSpelling:
				for l in usedLetters:
					s.discard(l)
				if len(s) == 1:
					letter = list(s)[0]
					spelling = (letter, ChromaticSpeller.alterations[(pc, letter)])
					self.currentSpelling[pc] = spelling
					self.currentCost += ChromaticSpeller.costs.get(spelling, 0)		# TODO: FIX BY KEEPING COSTS ALL ALONG
					redo = True
		
		if redo:
			self.update_possibilities()	
	
	def make_vexflow_dict(self):
		self.vexflowDict = {x[0]:self.tuple_to_string(x[1], spellingString = ChromaticSpeller.lowercaseLetters) for x in self.spellingDict.items()}
	
	def spell_pitches_vexflow(self, midiPitches, respell = False):
		if not midiPitches:
			return midiPitches
		if respell or self.respell or (not self.spellingDict):
			self.get_spelling(midiPitches)
		out = []
		for p in midiPitches:
			
			octave = int(p/12) - 1
			pc = p % 12
			spelledNote = self.vexflowDict.get(pc, ChromaticSpeller.defaultVexflowSpelling[pc])			# default spellings should not be needed, but are
			
			if spelledNote.count("b#"):
				octave -= 1
			elif spelledNote.count('cb'):
				octave += 1
				
			spelledNote = spelledNote + '/' + str(octave)
			out.append(spelledNote)
			
		return out

def get_sets(scaleSize, setSize, invert = False):
	return set([tuple(normal_form(k, invert = invert, modulus = scaleSize)) for k in tuple(itertools.combinations(range(scaleSize), setSize))])

def pc_distance(first, second, modulus = 12):
	return min((first - second) % modulus, (second - first) % modulus)

def simple_vl_size(first, second, modulus = 12):
	return sum([pc_distance(first[i], second[i], modulus = modulus) for i in range(len(first))])

class BassFinder():
	
	defaultPreferences = {7: 75, 4: 30, 3: 25, 10: 5, 1: -10, 5: -5}
	
	def __init__(self, midiChord = [], intervalPreferences = None):
		if not intervalPreferences:
			self.intervalPreferences = self.fix_dict(BassFinder.defaultPreferences)
		else:
			self.intervalPreferences = self.fix_dict(intervalPreferences)
		if midiChord:
			self.lastBass = self.get_bass(midiChord)
	
	def fix_dict(self, d):
		adjustment = min(d.values())
		if adjustment > 0:
			return d
		adjustment = -adjustment
		newD = {i:d.get(i, 0)+adjustment for i in range(12)}
		return newD
	
	def get_bass(self, midiChord):
		bassHisto = {}
		for n in midiChord:
			totalGoodness = 0
			for m in midiChord:
				totalGoodness += self.intervalPreferences.get((m-n)%12, 0)
			if totalGoodness > 0:
				bassHisto[n%12] = totalGoodness
		if bassHisto:
			return weighted_choice(bassHisto)
		else:
			return False
	
def minimum_vl(first, second, sort = True, modulus = 12):		# do better for chords with different lengths
	minimum_vl.fullList = []
	if len(second) > len(first):
		newNotes = []
		for i in range(0, len(second) - len(first)):
			newNotes.append(first[random.randrange(0, len(first))])
		first = first + newNotes
	elif len(second) < len(first):
		for i in range(0, len(first) - len(second)):
			first.pop(random.randrange(0, len(first)))
	firstPCs = sorted([p % modulus for p in first])
	secondPCs = sorted([p % modulus for p in second])
	secondPCs = secondPCs[1:] + [secondPCs[0] + modulus]
	currentBest = []
	currentBestSize = 100000000									# very large number
	for i in range(0, len(firstPCs) + 1):
		newSize = simple_vl_size(firstPCs, secondPCs, modulus = modulus)
		#print(firstPCs, secondPCs, newSize)
		newPaths = [[firstPCs[i], secondPCs[i] - firstPCs[i]] for i in range(len(firstPCs))]
		minimum_vl.fullList.append([newPaths, newSize])
		if newSize < currentBestSize:
			currentBestSize = newSize
			currentBest = newPaths
		secondPCs = [secondPCs[-1] - modulus] + secondPCs[:-1]
	minimum_vl.size = currentBestSize
	if sort:
		minimum_vl.fullList = sorted(minimum_vl.fullList, key = lambda x: x[1])
	return currentBest

def interval_vector(mySet, modulus = 12):
	output = [0] * int(modulus/2)
	for i in range(len(mySet) - 1):
		for j in range(i + 1, len(mySet)):
			x = pc_distance(mySet[i], mySet[j], modulus = modulus) - 1
			output[x] = output[x] + 1
	return output

def interchord_interval_vector(firstSet, secondSet, modulus = 12):
	output = [0] * (modulus/2)
	for i in range(len(firstSet)):
		for j in range(len(secondSet)):
			x = pc_distance(firstSet[i], secondSet[j], modulus = modulus) - 1
			output[x] = output[x] + 1
	return output

def retrograde_chain(pitchList = [60, 64, 67], counterMax = 0):
	print(0, pitchList)
	curNotes = pitchList[:]
	curNotes = [sum(curNotes[-2:]) - x for x in curNotes[::-1]]
	counter = 0
	while counter < counterMax or (not all ([(curNotes[i])%12 == (pitchList[i])%12 for i in range(len(pitchList))])):
		counter += 1
		print (counter, curNotes)
		curNotes = [sum(curNotes[-2:]) - x for x in curNotes[::-1]]
	counter += 1
	print (counter, curNotes)

def has_subset(s, subset, modulus = 12):
	count = 0
	for i in range(0, modulus):
		if all([(x+i)%modulus in s for x in subset]):
			count += 1
	return count

"""def convert_pair_to_paths(firstPCs, secondPCs):
	result = []
	for i in range(0, len(firstPCs)):
		path = (secondPCs[i] - firstPCs[i]) % 12
		if path > 6: 
			path -= 12
		result.append([firstPCs[i], path])
	return result"""

def voicelead(inPitches, outPCs, topN = 1, modulus = 12):
	inPCs = sorted([p % modulus for p in inPitches])
	paths = minimum_vl(inPCs, outPCs, modulus = modulus)					# PATHS may not have the same length as inPCs
	if topN != 1:
		myRange = min(len(minimum_vl.fullList), topN)
		paths = minimum_vl.fullList[random.randrange(0, myRange)][0]
	output = []
	for path in paths:
		for inPitch in inPitches:
			if (inPitch % modulus) == path[0]:
				output.append(inPitch + path[1])
				break
	return output

def geometrical_normal_form_local(inList, invert = False, modulus = 12, extraRotations = 0, voiceleadingRegion = False, permutationRegion = False, digits = None):				
	
	"""
	
	geometrical normal form is different from standard normal form insofar as it puts the smallest interval in the first position
	
	"""
	#print('in', inList)
	
	theLen = len(inList)
	
	if voiceleadingRegion:
		listOfPCs = sorted([k % modulus if k > modulus else k for k in inList])
		while sum(listOfPCs) > modulus:
			listOfPCs = [listOfPCs[-1] - modulus] + listOfPCs[:-1]
		return listOfPCs
	
	listOfPCs = sorted([(k - inList[0]) % modulus for k in inList])
	if permutationRegion:
		return listOfPCs
		
	listOfInts = [listOfPCs[i] - listOfPCs[i-1] for i in range(1, theLen)]
	
	if digits is not None:
		listOfInts = [round(x, digits) for x in listOfInts]
		
	listOfInts.append(modulus - listOfPCs[-1])
	currentBest = listOfInts[:]
	
	newChallenger = currentBest
	for i in range(1, theLen):
		newChallenger = newChallenger[1:] + [newChallenger[0]]
		if newChallenger < currentBest:
			currentBest = newChallenger
			
	if invert:	
		newChallenger = [currentBest[0]] + currentBest[-1:0:-1]
		if newChallenger < currentBest:
			currentBest = newChallenger
			
		for i in range(1, theLen):
			newChallenger = newChallenger[1:] + [newChallenger[0]]
			if newChallenger < currentBest:
				currentBest = newChallenger
	
	if extraRotations:
		minInt = currentBest[0]
		numInts = currentBest.count(minInt)
		extraRotations = extraRotations % numInts
		i = 0
		count = 0
		while count < extraRotations:
			i = (i + 1) % len(currentBest)
			if currentBest[i] == minInt:
				count += 1
		currentBest = currentBest[i:] + currentBest[:i]
	
	outList = [0]
	for i in currentBest[:-1]:
		outList.append(outList[-1] + i)
	#print('  out', outList)
	return outList

def normal_form(inList, invert = False, removeDuplicates = False, modulus = 12):				# takes a list of midi numbers
	if removeDuplicates:
		listOfPCs = sorted(list(set([k % modulus for k in inList])))
	else:
		listOfPCs = sorted([k % modulus for k in inList])
	currentBest = [k - listOfPCs[0] for k in listOfPCs]
	newChallenger = currentBest[:]
	normal_form.transposition = -listOfPCs[0] % modulus
	normal_form.inversion = False
	for i in range(1, len(listOfPCs)):
		newChallenger = newChallenger[1:] + [newChallenger[0] + modulus]
		newChallenger = [k - newChallenger[0] for k in newChallenger]
		transp = -listOfPCs[i] % modulus
		for j in reversed(range(0, len(listOfPCs))):
			if newChallenger[j] < currentBest[j]:
				currentBest = newChallenger
				normal_form.transposition = transp
			else:
				if newChallenger[j] > currentBest[j]:
					break
	if invert:																	# THIS PROBABLY DOESN'T WORK RIGHT!!!!
		listOfPCs = sorted([(modulus - k) % modulus for k in inList])
		listOfPCs = [x - listOfPCs[0] for x in listOfPCs]
		for i in range(len(listOfPCs)):
			newChallenger = listOfPCs[-i:] + listOfPCs[:-i]
			newChallenger = sorted([(k - newChallenger[0]) % modulus for k in newChallenger])
			for j in reversed(range(0, len(listOfPCs))):
				if newChallenger[j] < currentBest[j]:
					currentBest = newChallenger
				else:
					if newChallenger[j] > currentBest[j]:
						break
	return currentBest

def vl_normal_form(inList):														# list of [PC, path] pairs
	myList = sorted([[k[0] % 12] + k[1:] for k in inList])
	currentBest = [[(k[0] - myList[0][0]) % 12] + k[1:] for k in myList]
	vl_normal_form.transposition = myList[0][0] * -1
	for i in range(1, len(myList)):
		newChallenger = myList[-i:] + myList[:-i]
		transp = newChallenger[0][0] * -1
		newChallenger = sorted([[(k[0] - newChallenger[0][0]) % 12] + k[1:] for k in newChallenger])
		for j in reversed(range(len(myList))):
			if newChallenger[j][0] < currentBest[j][0]:
				currentBest = newChallenger
				vl_normal_form.transposition = transp
			else:
				if newChallenger[j][0] > currentBest[j][0]:
					break
	return currentBest

def twelve_tone_matrix(theRow, startOnZero = True):
	if startOnZero:
		theRow = [(x - theRow[0])%12 for x in theRow]
	inversion = [-x % 12 for x in theRow]
	for i in inversion:
		for j in theRow:
			print((j+i)%12, '\t', end = '')
		print()

def scale_matrix(theScale, modulus = 12):
	output = []
	newScale = theScale + [x + modulus for x in theScale]
	scaleLen = len(theScale)
	for i in range(scaleLen):
		output.append([newScale[i+j] - newScale[j] for j in range(scaleLen)])
	return output

def interscalar_matrix(firstChord, secondChord, fixedForm = False, modulus = 12):
	if fixedForm:
		myVL = [[firstChord[i], secondChord[i] - firstChord[i]] for i in range(len(firstChord))]
	else:
		myVL = minimum_vl(firstChord, secondChord, modulus)
	theVL = [x[1] for x in myVL]
	secondChord = [sum(x) for x in myVL]
	theMatrix = [add_vectors(theVL, x) for x in scale_matrix(secondChord, modulus = modulus)]
	return theMatrix


majInts = ['P1', 'm2', 'm3', 'P4', 'P5', 'm6', 'm7']					# tonics of scales containing a given note
melMinInts = ['P1', 'm2', 'm3', 'P4', 'P5', 'M6', 'm7']					# ... aka inversions of the scale around its tonic
harmMinInts = ['P1', 'm2', 'M3', 'P4', 'P5', 'M6', 'm7']

storedScales = [majInts, melMinInts, harmMinInts]

"""Equivalent to the above:
scale_from_letternames_init(pitchlist('C4 D4 E4 F4 G4 A4 B4'), reset = True)
scale_from_letternames_init(pitchlist('C4 D4 E-4 F4 G4 A4 B4'))
scale_from_letternames_init(pitchlist('C4 D4 E-4 F4 G4 A-4 B4'))
"""

def scale_from_letternames_init(newScale, reset=False):		# converts a list of pitches into a list of intervals
	global storedScales										# inverted around tonic note, for use in scale finding
	if reset: storedScales = []
	if newScale:
		iList = ['P1'] + [interval.subtract(['P8', interval.notesToInterval(newScale[0], x)]).name for x in newScale[1:]]
		storedScales.append(iList)
	
def scale_from_letternames(pitches):
	global storedScales
	results = []
	for myScale in storedScales:
		acceptableScales = set([pitches[0].transpose(x).name for x in myScale])
		for i in range(1, len(pitches)):
			acceptableScales = acceptableScales & set([pitches[i].transpose(x).name for x in myScale])
		results.append(acceptableScales)						
	return results

sevenScales = [[0, 2, 4, 5, 7, 9, 11], [0, 2, 4, 6, 7, 9, 10], [0, 2, 3, 5, 7, 8, 11], [0, 2, 4, 5, 7, 8, 11], [0, 2, 4, 6, 8, 10], [0, 1, 4, 5, 8, 9], [0, 1, 3, 4, 6, 7, 9, 10]]
sevenScaleSymmetries = [[-1, 3], [-2, 4], [-3, 4]]

class ScaleChooser():
	
	sevenScales = [[0, 2, 4, 5, 7, 9, 11], [0, 2, 3, 5, 7, 9, 11], [0, 2, 3, 5, 7, 8, 11], [0, 2, 4, 5, 7, 8, 11], [0, 2, 4, 6, 8, 10], [0, 1, 4, 5, 8, 9], [0, 1, 3, 4, 6, 7, 9, 10]]
	standardScales = [[[0, 2, 4, 5, 7, 9, 11], 12], [[0, 2, 3, 5, 7, 9, 11], 12], [[0, 2, 3, 5, 7, 8, 11], 12], [[0, 2, 4, 5, 7, 8, 11], 12], [[0], 2], [[0, 1], 4], [[0, 1], 3]]
	
	def __init__(self, scales = None, modulus = 12, weights = [], default = None):
		self.modulus = modulus
		if not scales:
			self.inputScales = ScaleChooser.sevenScales
			self.vocabulary = ScaleChooser.standardScales
		elif type(scales) is int:
			self.inputScales = ScaleChooser.sevenScales[:scales]
			self.vocabulary = ScaleChooser.standardScales[:scales]
		else:
			self.inputScales = scales
			self.get_symmetries()
			self.vocabulary = []
			for s  in scales:
				self.vocabulary.append(self.check_for_symmetry(s))
		self.weights = weights
		self.default = default if default else list(range(modulus))
		
			
	def get_symmetries(self):
		self.symmetries = []
		for i in reversed(range(2, int(self.modulus/2) + 1)):
			if self.modulus/i == int(self.modulus/i):
				self.symmetries.append(i)
		
	def check_for_symmetry(s):
		for symInt in self.symmetries:
			if all([(x+symInt)%self.modulus in s for x in s]):
				newScale = sorted({x % symInt for x in s})
				return [newScale, symInt]
		return [s, self.modulus]
	
	def choose_scale(self, pitches):
		scaleCount = len(self.vocabulary)
		options = [scale_inclusion(pitches, *x) for x in self.vocabulary]
		weights = [self.weights[i]*len(options[i]) for i in range(len(self.vocabulary))]
		weightDict = {}
		for i, w in enumerate(weights):
			if w: 
				weightDict[i] = w
		if not weightDict:
			return self.default
		else:
			self.scaleType = weighted_choice(weightDict)
		transposition = random.choice(list(options[self.scaleType]))
		self.lastScale = [(transposition + x) % self.modulus for x in self.inputScales[self.scaleType]]
		return self.lastScale
	
	def fix_lattice(self):
		"""
		for manually setting a scale while also using quadruple.py
			if you manually set the scale to acoustic, you want the program to go back to the diatonic lattice
		"""
		diatonicScale = self.lastScale[:]
		latticeMoves = []
		if self.scaleType == 1:
			for i in [5, 6]:
				diatonicScale[i] = (diatonicScale[i] - 1) % 12
			latticeMoves = [0, 2]
		elif self.scaleType == 2:
			for i in [1, 6]:
				diatonicScale[i] = (diatonicScale[i] - 1) % 12
			latticeMoves = [1, 2]
		elif self.scaleType == 3:
			for i in [2, 6]:
				diatonicScale[i] = (diatonicScale[i] - 1) % 12
			latticeMoves = [0, 3]
		return diatonicScale, latticeMoves
	
	@property
	def weights(self):
		return self.scaleWeights
	
	@weights.setter
	def weights(self, weights):
		self.scaleWeights = weights + [1] * (len(self.vocabulary) - len(weights))
	
def scale_chooser(pitches, weights = [], scalarVocabulary = sevenScales, symmetries = sevenScaleSymmetries):		# choose one of the seven scales randomly
	scaleCount = len(scalarVocabulary)				# yes, I know it is seven ... but maybe I want to generalize it someday
	options = scale_finder(pitches, scalarVocabulary)
	for i, symmetryFactor in [[-1, 3], [-2, 4], [-3, 4]]:
		options[i] = set([x % symmetryFactor for x in options[i]])
	l = len(weights)
	weights += [1] * (scaleCount - len(weights))
	weights = [weights[i]*len(options[i]) for i in range(scaleCount)]
	weightDict = {}
	for i, w in enumerate(weights):
		if w: 
			weightDict[i] = w
	if not weightDict:
		return False
	else:
		scaleNumber = weighted_choice(weightDict)
	transposition = random.choice(list(options[scaleNumber]))
	finalScale = [(transposition + x) % 12 for x in sevenScales[scaleNumber]]
	return sorted(finalScale)

def scale_finder(pitches, scales = [[0, 2, 4, 5, 7, 9, 11], [0, 2, 3, 5, 7, 9, 11], [0, 2, 3, 5, 7, 8, 11]]):
	results = []												# returns list of [scale#, possibility pairs]
	for s in scales:
		results.append(scale_inclusion(pitches, s))
	return results

def seven_scales(notes):
	results = scale_finder(notes, sevenScales)
	for i, modulus in [[4, 2], [5, 4], [6, 3]]:
		results[i] = {x % modulus for x in results[i]}
	return results

def ranked_choice_of_seven_scales(notes, choiceIndex = range(7)):
	"""choose from the seven scales in ranked order"""
	results = seven_scales(notes)
	newScale = False
	for i in choiceIndex:
		possibilities = results[i]
		if not possibilities: continue
		root = random.choice(list(possibilities))
		newScale = [(x + root) % 12 for x in sevenScales[i]]
		break
	if newScale: 
		return newScale
	return False
		
def scale_inclusion(pitches, scale, modulus = 12):
	"""
	for T-symmetrical scales you need to choose a modulus to account for redundancy
		octatonic -> modulus = 3
		hexatonic -> modulus = 4
		whole-tone -> modulus = 2
	
	"""
	scale = [modulus - x for x in scale]
	if type(pitches[0]) is int:
		pcs = pitches
	else:
		pcs = [p.pitchClass for p in pitches]
	acceptableScales = set(range(modulus))
	for p in pcs:
		acceptableScales = acceptableScales & set([(x + p) % modulus for x in scale])
	return acceptableScales
	
def find_scalar_and_chromatic_transposition(myVL, modulus = 12):
	myChord = [x[0] for x in myVL]
	myPaths = [x[1] for x in myVL]
	theLen = len(myChord)
	theMatrix = scale_matrix(myChord, modulus)
	for i, theRow in enumerate(theMatrix):
		chromaticTransp = [myPaths[x] - theRow[x] for x in range(theLen)]
		if chromaticTransp == ([chromaticTransp[0]] * theLen):
			return i, chromaticTransp[0]
	return False

def scalar_and_chromatic_transposition(myChord, scalarTransp = 0, chromaticTransp = 0, modulus = 12, pathsOnly = True):
	theLen = len(myChord)
	theMatrix = scale_matrix(myChord, modulus)
	if pathsOnly:
		return [theMatrix[scalarTransp][i] + chromaticTransp for i in range(len(myChord))]
	else:
		return [myChord[i] + theMatrix[scalarTransp][i] + chromaticTransp for i in range(len(myChord))]

def is_quarter_tone(inList, epsilon = .00001):
	return all([(x == int(x) or abs((x - int(x)) - .5) < epsilon) for x in inList])

def note_to_scalarMIDIdegree(myNote):			# convert note to lettername, middle C = 60
	"""o = myNote.pitch.octave
	degree = 'CDEFGAB'.index(myNote.name[0])
	return 32 + (7*o) + degree"""
	return myNote.pitch.diatonicNoteNum + 31

"""legacy routine, replaced by ScaleObject"""
def scalar_MIDI_notes(theScale):				# allows you to use something like standard MIDI numbers to stay inside a scale
	modulus = len(theScale)
	pitchScale = {}
	inversePitchScale = {}
	lastNote = min(theScale)
	j = 0
	i = theScale.index(lastNote)
	while lastNote < 127:
		i = (i + 1) % modulus
		j += 1
		newPC = theScale[i] + (lastNote / 12) * 12
		if newPC < lastNote:
			newPC += 12
		pitchScale[j] = newPC
		lastNote = newPC
	s = sorted(pitchScale.items(), key = lambda x: abs(60 - x[1]))
	offset = 60 - s[0][0]
	newDict = {}
	for item in pitchScale:
		newDict[item + offset] = pitchScale[item]
	pitchScale = newDict
	inversePitchScale = {v: k for k, v in pitchScale.items()}
	return pitchScale, inversePitchScale	

# simple beat-finding routine
def quantize_rhythm(durationList = [], tactus = False, tolerance = .035, smallestAllowableNote = .020, subdivisions = [1, 2]):		# change subdivisions to [1, 2]
	"""
	tactus is a pre-existing beat; otherwise it uses the smallest value in the pattern
	tolerance is the maximum average deviation from precision
	smallestAllowableNote weeds out very small note values (treated as parts of chords)
	subdivisions 

	"""
	
	tempDurations = []
	newDurations  = []
	overflow = 0
	for d in durationList:
		if d < smallestAllowableNote:
			overflow += d
			newDurations.append(0)
		else:
			tempDurations.append(d + overflow)
			overflow = 0
			newDurations.append(d)
	
	if not tempDurations:
		return False
	if not tactus:
		minDur = min(tempDurations)
		smallValues = [x for x in tempDurations if x < minDur+tolerance]
		tactus = sum(smallValues)/len(smallValues)
	
	totalDuration = sum(tempDurations)
	numBeats = int(0.5 + totalDuration/tactus)
	originalBaseDur = totalDuration/numBeats
	
	for subdivision in subdivisions:
		baseDur = originalBaseDur/subdivision
		quantizedDurations = []
		failures = 0
		errors = []
	
		for testDur in newDurations:
			if testDur == 0:
				quantizedDurations.append(0)
				continue
			ratio = testDur/baseDur
			qRatio = int(.5 + ratio)
			error =  testDur - qRatio*baseDur
			errors.append(error)
			quantizedDurations.append(qRatio)
		
		meanErr = sum(errors)/len(errors)
		
		if abs(meanErr) <= tolerance: 
			return baseDur, quantizedDurations
			
	return False

"""
================================================================
				Flatten a hierarchical list
================================================================
"""

"""
	the ignore function describes lists that should be preserved	

"""
	
def default_ignore_function(item):
	if type(item[0]) is str:
		return True
	return False
	
def flatten_note_list(l, ignoreFunction = default_ignore_function):
	
	flatList = []
	noteGroups = []
	
	for item in l:
		curLen = len(flatList)
		if (type(item) is not list) or ignoreFunction(item):
			noteGroups.append(curLen)
			flatList.append(item)
		else:
			item = flatten_list(item, ignoreFunction = ignoreFunction)
			noteGroups.append(slice(curLen, curLen + len(item)))
			flatList += item
			
	return flatList, noteGroups

def flatten_list(l, ignoreFunction = default_ignore_function):
	out = []
	for item in l:
		if (type(item) is not list) or ignoreFunction(item):
			out.append(item)
		else:
			out += flatten_list(item)
	return out

"""
================================================================
			Group a split-up list by parentheses
================================================================
"""


def parse_parentheses(sSplit):
	for openItem, closeItem in ['{}', '()']:
		sSplit = parse_paren_unit(sSplit, openItem, closeItem)
	return sSplit

def parse_paren_unit(sSplit, openItem = '{', closeItem = '}'):
	out = []
	targetList = out
	depth = 0
	
	for i, s in enumerate(sSplit):	
		delta = s.count(openItem) - s.count(closeItem)
		if depth == 0 and delta > 0:
			targetList = []
			
		targetList.append(s)
		
		if depth > 0 and delta == -depth:
			out.append(' '.join(targetList))
			targetList = out
			
		depth = depth + delta
	
	return out

"""
================================================================
				Routines for processing Roman Numerals
================================================================
"""
	
"""The following kludgey routine identifies unknown RNs"""

def fix_roman(RN):		
	global seventh
	seventh = None
	fix_roman.seventh = None
	rawFigure = stripInv(RN.figure)														# deal with secondary dominants someday
	if rawFigure in _UNKNOWNCHORDS:														# also make major-key analogues for all my chords
		scaleDegrees = _UNKNOWNCHORDS[rawFigure]
		if scaleDegrees.count(' ') > 2:
			seventh = parseDegrees(scaleDegrees.split()[3], RN.key)
			fix_roman.seventh = parseDegrees(scaleDegrees.split()[3], RN.key)
		return parseDegrees(scaleDegrees, RN.key, RN.inversion())				
	else:
		if RN.seventh: 
			seventh = RN.seventh.pitchClass
			fix_roman.seventh = RN.seventh.pitchClass 
		return RN.pitchClasses

acciFix = ['A#']

def parseDegrees(chordStr, myKey, rotation = 0):
#	rotationDict = {'6':1, '6/4': 2, '7':0, '6/5':1, '4/3':2, '2':3}
#	rotation = rotationDict[rotStr]
	PCs = []
	myScale = myKey.pitches
	degrees = chordStr.split()
	for degStr in degrees:
		degree = int(degStr[-1:]) - 1
		myNote = myScale[degree]
		alteration = (-1 * degStr.count('b')) + degStr.count('#')
		PCs.append((myNote.pitchClass + alteration) % 12)
	return PCs[rotation:] + PCs[:rotation]

def figure_in_new_key(figureInOldKey, oldKeyName, newKeyName):
	newKey = key.Key(newKeyName)
	return figure_from_chord(roman.RomanNumeral(figureInOldKey, oldKeyName).pitches, newKey)

def parse_figure(chordFig):
	if not chordFig: return ['', '', '', '']
	if chordFig[-1] == '/': chordFig = chordFig[:-1]		#REMOVE THIS AFTER FIXING DATA!!!!
	outList = ['', '', '', '']			# base figure, inversion, tonicization, alteration
	index = 0
	for i in range(len(chordFig)):
		if index == 0 and chordFig[i].isdigit():
			index = 1
		if chordFig[i] == '/':
			if chordFig[i+1] != 'o' and not chordFig[i+1].isdigit():
				index = 2
		if chordFig[i] == '[': 
			index = 3
			continue
		outList[index] += chordFig[i]
	if outList[2]: outList[2] = outList[2].lstrip('/')
	if outList[3]: outList[3] = outList[3].rstrip(']')
	return outList

def unparse_figure(root, inv = '', secondary = '', alteration = ''):
	fig = root + inv
	if secondary:
		fig = fig + '/' + secondary
	if alteration:
		fig = fig + '[' + alteration + ']'
	return fig

def stripInv(chordStr):
	if chordStr[-1:] == '7' or chordStr[-1:] == '2' or chordStr[-1:] == '6': return chordStr[:-1]
	if chordStr[-3:] == '6/5' or chordStr[-3:] == '4/3' or chordStr[-3:] == '6/4': return chordStr[:-3]
	if chordStr == 'It6/ii' or chordStr == 'V9[b9]': return chordStr
	return chordStr

def remove_inversions(progStr, separateSixFour = True):
	remove_inversions.isDiatonic = True
	prog = progStr.split(' -> ')
	for i in range(len(prog)):
		keyStr = ''
		if prog[i].count(': ') > 0:
			keyStr, prog[i] = prog[i].split(': ')
			keyStr += ': '
		if prog[i][-1] == '/': prog[i] = prog[i][:-1]
		newFig = parse_figure(prog[i])
		if separateSixFour:
			if newFig[1] != '6/4': newFig[1] = ''
		else:
			newFig[1] = ''
		if newFig[2]: 
			remove_inversions.isDiatonic = False
			newFig[2] = '/' + newFig[2]
		newFig[0] = newFig[0].replace('maj', '').replace('/o','o')
		prog[i] = keyStr + ''.join(newFig)
	newProg = []
	newProg.append(prog[0])
	for i in range(1, len(prog)):
		if prog[i] != newProg[-1]: newProg.append(prog[i])
	return ' -> '.join(newProg)

romanNumeralMeanings = [["VII", 7], ["VI", 6],  ["IV", 4], ["V", 5], ["III", 3], ["II", 2],  ["I", 1]]
def getroman(s):
	sUP = s.upper()
	for r in romanNumeralMeanings:
		if sUP.count(r[0]) > 0: return r[1]
	return False

def fix_flats(l):
	if len(l) > 1 and l[-1] == 'b':
		return l[:-1] + '-'
	return l

class Transposer():
	"""transposes letter names with just regular python"""
	
	semitoneTranspDict = {'m2': 1, 'M2': 2, 'm3': 3, 'M3': 4, 'P4': 5, 'A4': 6, 'd5': 6, 'D5': 6, 'P5': 7, 'm6': 8, 'M6': 9, 'm7': 10, 'M7': 11}
	letterNames = 'CDEFGAB'
	letterPCs = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
	#CmajorScale = [0, 2, 4, 5, 7, 9, 11]
	#for i in range(7):
	#	letterPCs[letterNames[i]] = CmajorScale[i]
	
	def __init__(self, transp = '', octaveShift = 0):
		self.set_transposition(transp, octaveShift)
		
	def set_transposition(self, transp = None, octaveShift = 0):
		if not transp or len(transp) < 2:
			self.letterShift = 0
			self.semitoneShift = 0
			self.sign = 1
		else:
			if transp[0] == '-':
				self.sign = -1
				transp = transp[1:]
			else:
				self.sign = 1
			self.letterShift = self.sign * (int(transp[-1]) - 1)
			stShift = Transposer.semitoneTranspDict.get(transp, False)
			if not stShift:
				print("Can't get that transposition", transp)
				self.error = True
				return
			self.semitoneShift = self.sign * stShift
		self.transpositionDict = {s:self.calculate_letter(s, octaveShift) for s in Transposer.letterNames}
		
	def set_transposition_manual(self, letterShift = 0, semitoneShift = 0, sign = 0, octaveShift = 0):
		"""for exotic transpositions"""
		self.letterShift = letterShift
		self.semitoneShift = semitoneShift
		self.sign = sign
		self.transpositionDict = {s:self.calculate_letter(s, octaveShift) for s in Transposer.letterNames}
		
	def calculate_letter(self, baseLetter, octaveShift = 0):
		
		oldPC = Transposer.letterPCs[baseLetter]
		baseIndex = Transposer.letterNames.index(baseLetter)
		
		newIndex = (baseIndex + self.letterShift) % 7
		newLetter = Transposer.letterNames[newIndex]
		newPC = Transposer.letterPCs[newLetter]
		
		PCdelta = newPC - oldPC
		if PCdelta > 0 and self.semitoneShift < 0:
			PCdelta = PCdelta - 12
		elif PCdelta < 0 and self.semitoneShift > 0:
			PCdelta = PCdelta + 12
		
		letterDiff = newIndex - baseIndex
		if PCdelta > 0 and letterDiff < 0:
			octaveShift += 1
		if PCdelta < 0 and letterDiff > 0:
			octaveShift -= 1
		
		modifierChange = self.semitoneShift - PCdelta
		return [newLetter, modifierChange, octaveShift]
		
	def transpose_vexflow(self, letterName):
		baseLetter, *modifiers = [x for x in letterName]
		modifier = -modifiers.count('b') - modifiers.count('-') + modifiers.count('#')
		newLetter, modifierChange, octaveShift = self.transpositionDict[baseLetter.upper()]
		newModifier = (self.sign * modifierChange) + modifier
		
		newLetter = newLetter.lower()	
		octaveStr = '/' + str(int(letterName[-1]) + octaveShift)
		
		if newModifier <= 0:
			return newLetter + -newModifier * 'b' + octaveStr
		return newLetter + newModifier * '#' + octaveStr
	
	def transpose_letter(self, letterName):
		baseLetter, *modifiers = [x for x in letterName]
		modifier = -modifiers.count('b') - modifiers.count('-') + modifiers.count('#')
		newLetter, modifierChange, octaveShift = self.transpositionDict[baseLetter.upper()]
		newModifier = (self.sign * modifierChange) + modifier
		
		octaveStr = str(int(letterName[-1]) + octaveShift) if letterName[-1].isdigit() else ''
			
		if newModifier <= 0:
			return newLetter + -newModifier * 'b' + octaveStr
			
		return newLetter + newModifier * '#' + octaveStr
	
	def transpose_letter_or_number(self, theInput):
		if theInput[0].isdigit():
			return (int(theInput) + self.semitoneShift) % 12
		else:
			return transpose_letter(theInput)

def transpose_letter(letterName, transp, fixFlats = False):
	if letterName[0] == '?': return False
	if fixFlats:
		letterName = fix_flats(letterName)
	if letterName[0].islower():
		return pitch.Pitch(letterName).transpose(transp).name.lower()
	else:
		return pitch.Pitch(letterName).transpose(transp).name
		
def transpose_bass(chorale, t = '-P8'):
	chorale.parts[len(chorale.parts) - 1].transpose(t, inPlace = True)

romans = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
triad_invs = ['','6', '6/4']
seventh_invs = ['7','6/5', '4/3', '2']
ninth_invs = ['9','7/6/5', '6/5/4', '4/3/2', '7/6/2']

_SHARPACCIDENTAL = pitch.Accidental("sharp")

def figure_from_chord(myChord, myKey, secondaries = True, minorSeven = True, useMajorDominantInMinor = True):
	"""fast, cheap method for getting an RN figure from a chord and a key
			- no error checking for multiple scale degrees
			- it does not inform you of missing triad notes
			- no labels for crazy chords like [C, E--, G]
			- right now there's an error in the case of bare sevenths like A, G, returning a ninth rather than a seventh
		but it runs quickly!
	"""
	global triad_invs, seventh_invs, ninth_invs, romans
	alterationsAndPCs = [[],[],[],[],[],[],[]]					# indexed by scale degree
	degreeSet = set([])											# set of scale degrees
	figure_from_chord.missingDegrees = []
	
	if type(myChord) == list or type(myChord) == tuple:
		bass = (myKey.getScaleDegreeAndAccidentalFromPitch(min(myChord))[0] - 1) % 7
		for p in myChord: 
			sd = myKey.getScaleDegreeAndAccidentalFromPitch(p)			
			alterationsAndPCs[sd[0] - 1] = [sd[1], p.pitchClass]
			degreeSet = degreeSet | set([(sd[0] - 1) % 7])
	else:
		bass = (myKey.getScaleDegreeAndAccidentalFromPitch(myChord.bass())[0] - 1) % 7
		for p in myChord: 
			sd = myKey.getScaleDegreeAndAccidentalFromPitch(p.pitch)
			"""TO CHECK FOR DUPLICATE SDs:
			if alterationsAndPCs[sd[0] - 1]: print 'ERROR'"""
			alterationsAndPCs[sd[0] - 1] = [sd[1], p.pitch.pitchClass]
			degreeSet = degreeSet | set([(sd[0] - 1) % 7])
	foundNumeral = False
	
	"""We find the root by stacking a ninth chord on top of each note in the chord; if the result
	contains our input scale degrees, we are set."""
	
	for d in degreeSet:
		tryChord = set([(x + d) % 7 for x in [0, 2, 4, 6, 1]])	# stack of thirds, expressed as scale degrees
		if degreeSet <= tryChord:			
			foundNumeral = True									# d is the root scale degree
			break
	
	if degreeSet == set([d, d+1]): d = (d + 1) % 7				# fixes ninth chord error
	
	if foundNumeral:
		findBass = [(x + d) % 7 for x in [0, 2, 4, 6, 1]]
		alteration = alterationsAndPCs[d][0]
		if alteration: 
			alteration = alteration.modifier.replace('-', 'b')	# DT format, not music21-approved; just comment it out
		else: 
			alteration = ''
		
		"""	NOW WE CLEAN THE PITCHES IN OUR CHORD, adding in any missing degrees
			late breaking addition: dominants missing a third now default to V in minor, rather than v 
		"""
		#print('a', d, alterationsAndPCs)
		missingThird = (d + 2) % 7
		if not alterationsAndPCs[missingThird]: 
			if useMajorDominantInMinor and myKey.mode == 'minor':
				if missingThird == 6:
					alterationsAndPCs[missingThird] = [0, (myKey.pitches[missingThird].pitchClass + 1)%12]
				elif missingThird == 5 and alterationsAndPCs[d%7][0] == _SHARPACCIDENTAL:
					alterationsAndPCs[missingThird] = [0, (myKey.pitches[missingThird].pitchClass + 1) % 12]
				else:
					alterationsAndPCs[missingThird] = [0, myKey.pitches[missingThird].pitchClass]
			else:
				alterationsAndPCs[missingThird] = [0, myKey.pitches[missingThird].pitchClass]
			figure_from_chord.missingDegrees.append(3)
		#print('b', d, alterationsAndPCs)
		
		
		missingFifth = (d + 4) % 7
		if not alterationsAndPCs[missingFifth]: 
			if myKey.mode == 'minor' and alterationsAndPCs[missingThird][0] == _SHARPACCIDENTAL and d == 1:
				alterationsAndPCs[(d + 4) % 7] = [0, myKey.pitches[(d + 4) % 7].pitchClass + 1]
			else:
				alterationsAndPCs[(d + 4) % 7] = [0, myKey.pitches[(d + 4) % 7].pitchClass]
			figure_from_chord.missingDegrees.append(5)
			
		if degreeSet <= set([(x + d) % 7 for x in [0, 2, 4]]):						# chord is a triad
			inv = triad_invs[findBass.index(bass)]									# findBass gives the position of the bass in the stack of thirds
			nf = [(alterationsAndPCs[x][1] - alterationsAndPCs[d][1]) % 12 for x in findBass[0:3]]	# stack of thirds normal form
			outStr = ''
			tonicization = ''
			if nf == [0, 4, 7]: outStr = romans[d]
			elif nf == [0, 3, 7]: outStr = romans[d].lower()
			elif nf == [0, 3, 6]: outStr = romans[d].lower() + 'o'
			elif nf == [0, 4, 8]: outStr = romans[d] + '+'
			elif nf == [0, 2, 6] and alteration == '#': 
				if d == 3:
					alteration = ''
					outStr = 'It'
				else:
					outStr = romans[d].lower() + 'o'
					tonicization = '[b3]'	# not a tonicization, but a suffix
			outStr = alteration + outStr 
			if myKey.mode == 'minor':							# correct minor key triads
				if outStr == "#vio": outStr = 'vio'
				elif outStr == "#viio": outStr = 'viio'
			if secondaries:
				if myKey.mode == 'major':
					if outStr == 'II': 
						outStr = 'V'
						tonicization = '/V'
					elif outStr == '#io': 
						outStr = 'viio'
						tonicization = '/ii'
					elif outStr == '#iio': 
						outStr = 'viio'
						tonicization = '/iii'
					elif outStr == '#ivo': 
						outStr = 'viio'
						tonicization = '/V'
					elif outStr == '#vo': 
						outStr = 'viio'
						tonicization = '/vi'
					elif outStr == 'III': 
						outStr = 'V'
						tonicization = '/vi'
					elif outStr == 'VI': 
						outStr = 'V'
						tonicization = '/ii'
					elif outStr == 'VII':
						outStr = 'V'
						tonicization = '/iii'
				elif myKey.mode == 'minor':
					if outStr == 'II': 
						outStr = 'V'
						tonicization = '/V'
					elif outStr == '#iiio': 
						outStr = 'viio'
						tonicization = '/iv'
					elif outStr == '#ivo': 
						outStr = 'viio'
						tonicization = '/V'
					elif outStr == 'vo': 
						outStr = 'viio'
						tonicization = '/VI'
					elif outStr == 'vio': 
						outStr = 'viio'
						tonicization = '/VII'
					elif outStr == 'VII' and (not minorSeven): 
						outStr = 'V'
						tonicization = '/III'
			return outStr + inv + tonicization
		elif degreeSet <= set([(x + d) % 7 for x in [0, 2, 4, 6]]): 		# seventh chords, basically the same
			inv = seventh_invs[findBass.index(bass)]
			nf = [(alterationsAndPCs[x][1] - alterationsAndPCs[d][1]) % 12 for x in findBass[0:4]]
			outStr = ''
			tonicization = ''
			if nf == [0, 4, 7, 10]: outStr = romans[d]
			elif nf == [0, 3, 7, 10]: outStr = romans[d].lower()
			elif nf == [0, 3, 6, 9]: outStr = romans[d].lower() + 'o'
			elif nf == [0, 3, 6, 10]: outStr = romans[d].lower() + '/o'
			elif nf == [0, 4, 7, 11]: outStr = romans[d] + 'maj'
			elif nf == [0, 4, 8, 11]: outStr = romans[d] + '+maj'
			elif nf == [0, 4, 8, 10]: outStr = romans[d] + '+'
			elif nf == [0, 3, 7, 11]: outStr = romans[d].lower() + 'maj'
			elif nf == [0, 4, 6, 10]: outStr = romans[d] + '[b5]'
			elif nf == [0, 2, 6, 9] and alteration == '#': 
				if d == 3:
					outStr = 'Ger'
					alteration = ''
				else:
					outStr = romans[d].lower() + 'o'
					tonicization = '[b3]'	# not a tonicization, but a suffix
			outStr = alteration + outStr 
			if myKey.mode == 'minor':
				if outStr == "#vi/o": outStr = 'vi/o'
				elif outStr == "#viio": outStr = 'viio'
				elif outStr == '#vii/o': outStr = 'vii/o'
			if secondaries:
				if myKey.mode == 'major':
					if outStr == 'II': 
						outStr = 'V'
						tonicization += '/V'
					elif outStr == 'I':
						outStr = 'V'
						tonicization += '/IV'
					elif outStr == 'III': 
						outStr = 'V'
						tonicization += '/vi'
					elif outStr == 'VI': 
						outStr = 'V'
						tonicization += '/ii'
					elif outStr == '#io':
						outStr = 'viio'
						tonicization += '/ii'
					elif outStr == '#ivo':
						outStr = 'viio'
						tonicization += '/V'
					elif outStr == '#iv/o':
						outStr = 'vii/o'
						tonicization += '/V'
					elif outStr == '#vo':
						outStr = 'viio'
						tonicization += '/vi'
					elif outStr == 'VII':
						outStr = 'V'
						tonicization += '/iii'
				elif myKey.mode == 'minor':
					if outStr == 'II': 
						outStr = 'V'
						tonicization += '/V'
					elif outStr == 'I': 
						outStr = 'V'
						tonicization += '/iv'
					elif outStr == 'III':
						outStr = 'V'
						tonicization += '/VI'
					elif outStr == '#ivo':
						outStr = 'viio'
						tonicization += '/V'
					elif outStr == '#iiio':
						outStr = 'viio'
						tonicization += '/iv'
					elif outStr == 'v/o':
						outStr = 'ii/o'
						tonicization += '/iv'
					elif outStr == 'VII' and (not minorSeven): 
						outStr = 'V'
						tonicization += '/III'
			return outStr + inv + tonicization
		elif degreeSet <= set([(x + d) % 7 for x in [0, 2, 4, 6, 1]]):		# now ninths
			if not alterationsAndPCs[(d + 6) % 7]: 
				alterationsAndPCs[(d + 6) % 7] = [0, myKey.pitches[(d + 6) % 7].pitchClass]
				figure_from_chord.missingDegrees.append(7)
			inv = ninth_invs[findBass.index(bass)]
			nf = [(alterationsAndPCs[x][1] - alterationsAndPCs[d][1]) % 12 for x in findBass[0:6]]
			outStr = ''
			if nf[0:4] == [0, 4, 7, 10]: outStr = romans[d]
			elif nf[0:4] == [0, 3, 7, 10]: outStr = romans[d].lower()
			elif nf[0:4] == [0, 3, 6, 10]: outStr = romans[d].lower() + '/o'
			elif nf[0:4] == [0, 4, 7, 11]: outStr = romans[d] + 'maj'
			elif nf[0:4] == [0, 4, 8, 11]: outStr = romans[d] + '+maj'
			elif nf[0:4] == [0, 4, 8, 10]: outStr = romans[d] + '+'
			elif nf[0:4] == [0, 3, 7, 11]: outStr = romans[d].lower() + 'maj'
			elif nf[0:4] == [0, 4, 6, 10]: outStr = romans[d] + '[b5]'
			if nf[4] == 1: outStr += '[b9]'
			if nf[4] == 3: outStr += '[#9]'
			outStr = alteration + outStr 
			if myKey.mode == 'minor':
				if outStr[0:5] == "#vi/o": outStr = 'vi/o' + outStr[5:]
				elif outStr[0:5] == "#viio": outStr = 'viio'+ outStr[5:]
			return outStr + inv
	return False

"""
================================================================
				Remove repeats and multiple endings
================================================================
	
This is the bane of computational analysis; very hard to do reliably; I have not succeeded
"""

def remove_repeats(myStream):							# this assumes a stream with repeat measures numbered 56a, 57a, 58a, etc.
	remove_repeats.removedMeasures = []
	for myPart in myStream.parts:
		myMeasures = myPart.getElementsByClass('Measure')
		mySpanners = myPart.getElementsByClass('Spanner')
		for i in reversed(range(len(myMeasures))):
			if myMeasures[i].numberSuffix == 'a':
					remove_repeats.removedMeasures.append([myMeasures[i].number, myMeasures[i].numberSuffix])
					for s in mySpanners:
						if myMeasures[i] in s:
							myPart.remove(s)
					myPart.remove(myMeasures[i], shiftOffsets=True)
			elif myMeasures[i].numberSuffix == 'b':
				myMeasures[i].numberSuffix = ''
	return myStream

def remove_repeats_guided(myStream):					# this assumes a file with the measures numbered properly
	targetMeasures = [x[0] for x in remove_repeats.removedMeasures]
	targetMeasures.reverse()
	print ("target", targetMeasures)
	for myPart in myStream.parts:
		myMeasures = myPart.getElementsByClass('Measure')
		mySpanners = myPart.getElementsByClass('Spanner')
		for i in range(len(myMeasures)):
			if myMeasures[i].number in targetMeasures:
				for s in mySpanners:
					if myMeasures[i] in s:
						myPart.remove(s)
				myPart.remove(myMeasures[i], shiftOffsets=True)
				targetMeasures.remove(i)								#remove only the first appearance of the measure number
				if not targetMeasures: break
	return myStream

def remove_repeats_guided2(myStream):					# this assumes a file with the measures numbered sequentially
	safeTargetMeasures = [x[0] for x in remove_repeats.removedMeasures]
	safeTargetMeasures.reverse()
	print("target", safeTargetMeasures)
	for myPart in myStream.parts:
		targetMeasures = safeTargetMeasures[:]
		mySpanners = myPart.getElementsByClass('Spanner')
		measureCorrection = 0
		i = 0
		while i < (len(myPart)):
			#print i, myPart[i]
			if 'Measure'  in myPart[i].classes and myPart[i].number != 0:
				#print "     ", myPart[i].number, myPart[i].number - measureCorrection
				newNumber = myPart[i].number - measureCorrection
				myPart[i].number = newNumber
				if myPart[i].number in targetMeasures:
					for s in mySpanners:
						if myPart[i] in s:
							myPart.remove(s)
					#print "removing", myPart[i], myPart[i].number
					myPart.remove(myPart[i], shiftOffsets=True)
					#print "postremove", myPart[i]
					measureCorrection += 1
					#print targetMeasures
					if newNumber + 1 in targetMeasures:				# a bit of voodoo here.  If we are removing 56 and 57, this is because they are both under the same repeat
						targetMeasures.remove(newNumber + 1)
					else:
						targetMeasures.remove(newNumber)
					#print targetMeasures
					continue			
			i += 1								# increment the pointer only when not removing the measure currently stored at i
	return myStream

"""
================================================================
A bunch of tools for displaying sections of scores, used by many different programs
================================================================
"""

def read_chorale(n, useReduced = False, useDuplicates = False):												# eventually convert everything to XML
	if (not useDuplicates) and (n in _DUPLICATECHORALES): return False
	if useReduced:
		tempChorale = converter.parse(_REDUCEDCHORALEPATH + 'simpchor' + str(n).zfill(3) + '.xml')
	else:
		tempChorale = converter.parse(_XMLCHORALES + 'riemenschneider' + str(n).zfill(3) + '.xml')	
	tempChorale2 = stream.Score()
	for i in range(len(tempChorale)):
		if 'Part' in tempChorale[i].classes:
			tempChorale2.insert(0, tempChorale[i])
	for i in range(len(tempChorale2.parts)):
		tempChorale2.parts[i].id = i																		# part ID can be useful later
	return tempChorale2

def append(chorale, output):																				# simpler version of append_measures			
	neededParts = len(chorale.parts) - len(output.parts)
	#print "NEEDEDPARTS", neededParts
	if neededParts > 0 and len(output.parts) > 0 and len(output.parts[0].getElementsByClass('Measure')) > 0:
		myTemplate = output.parts[0].template(fillWithRests=False)									# output must have parts
	else:
		myTemplate = stream.Part()
	for i in range(neededParts):
		output.insert(0, copy.deepcopy(myTemplate))
	myTemplate = chorale.parts[0].template(fillWithRests=False)	
	for i in range(-1 * neededParts):
		chorale.insert(0, copy.deepcopy(myTemplate))
	for voice in range(0, len(chorale.parts)):
		measures = chorale.parts[voice].getElementsByClass('Measure')
		for m in measures:
			output[voice].append(m)
	return output

def append_measures(chorale, output, mList, choraleNo = 0, textNote = '', addRoman = True):
	neededParts = len(chorale.parts) - len(output.parts)
	if neededParts > 0 and len(output.parts) > 0 and len(output.parts[0].getElementsByClass(measures)) > 0:
		myTemplate = output.template(fillWithRests=True)
	else:
		myTemplate = stream.Part()
	for i in range(neededParts):
		output.insert(0, copy.deepcopy(myTemplate))
	if type(mList) is int:
		mStart = mList
		mEnd = mStart
	else:
		mStart = mList[0]
		if len(mList) > 1: 
			mEnd = mList[1]
		else:
			mEnd = mStart
	if choraleNo != 0 and addRoman is not False: 
		add_roman(chorale,choraleNo, mStart, mEnd)
	newMeas = get_measures(chorale, mStart, mEnd)
	header1 = str(choraleNo)+'m'+str(mStart)
	insert_text(newMeas.parts[0][0], header1, 55, 0)
	if textNote != '': insert_text(newMeas.parts[0][0], textNote, 85, 0)
	for voice in range(0, len(newMeas.parts)):
		measures = newMeas.parts[voice].getElementsByClass('Measure')
		for m in measures:
			output[voice].append(m)
	return output
	
def get_measures(chorale, mStart, mEnd = -1):					# need to fix this so that it gets unnumber continuation measures
	if mEnd == -1: mEnd = mStart
	output = stream.Score()
	if mStart <= 1: 
		startOffset = 0							# pickup measures included automatically at the start
	else:
		startOffset = chorale.parts[-1].elementOffset(chorale.parts[-1].measure(mStart))
	#print mStart, startOffset
	if not chorale.parts[-1].measure(mEnd):
		mEnd = mEnd - 1
	endOffset = chorale.parts[-1].elementOffset(chorale.parts[-1].measure(mEnd))
	if mEnd == mStart:
		try:
			tryMeasure = chorale.parts[-1].getElementAfterElement(chorale.parts[-1].measure(mStart), classList = ['Measure'])
			if tryMeasure.number == 0: endOffset = tryMeasure.offset
		except:
			pass
	for voice in range(0, len(chorale.parts)):
		theMeasures = chorale.parts[voice].getElementsByOffset(startOffset, endOffset, includeEndBoundary = True, classList = ['Measure'])
		if voice == 0 : 
			#print '1', startOffset
			#print 2, chorale.parts[voice].flat.getElementAtOrBefore(startOffset, classList = ['KeySignature'])
			try:
				kSig = copy.deepcopy(chorale.parts[voice].flat.getElementAtOrBefore(startOffset, classList = ['KeySignature']))
				tSig = copy.deepcopy(chorale.parts[voice].flat.getElementAtOrBefore(startOffset, classList = ['TimeSignature']))
			except:
				kSig = key.KeySignature(0)
				tSig = meter.TimeSignature('4/4')
		try:
			myClef = chorale.parts[voice].flat.getElementAtOrBefore(startOffset, classList = ['Clef'])
		except:
			myClef = clef.TrebleClef()
#		if kSig: kSig.show('text')
		tempPart = stream.Part()
#		count = 0
		for m in theMeasures:
			tempPart.insert(tempPart.duration.quarterLength, m)
		output.insert(0, tempPart)
		try:
			output.parts[voice][0].insert(0, kSig)				
			output.parts[voice][0].insert(0, myClef)
			output.parts[voice][0].insert(0, tSig)	
		except:
			pass
	return output

def sc(chorale, start, end = -1):
	if end == -1: end = start
	show_chorale(chorale, [start, end]).show()

def get(chorNo, startM, endM = -1):
	if endM == -1:
		show_chorale(chorNo, [startM]).show()
	else:
		show_chorale(chorNo, [startM, endM]).show()

def show_chorale(choraleNumber, measureList, useReduced = False, twoStaves = True, addRoman = True):			# TODO: errorcheck for measure#s out of range																# add braces to this
	chorale = read_chorale(choraleNumber, useReduced, useDuplicates = True)										# to use reduced, need to add this
	if type(measureList) == list:																				# better offset checking here
		startMeas = measureList[0]																				# one possibility: put offsets on
		endMeas = measureList[-1]																				# BEFORE getting the measures via get_measures!
	else:
		startMeas = endMeas = measureList
	if addRoman:
		add_roman(chorale, choraleNumber, startMeas, endMeas)
	output = get_measures(chorale, startMeas, endMeas)
	if twoStaves:
		output = two_staves(get_measures(chorale, startMeas, endMeas))
	return output

def show_piece(filePath, analysisPath, measureList, twoStaves = True, addRoman = True):					# TODO: errorcheck for measure#s out of range																# add braces to this
	chorale = converter.parse(filePath)																	# to use reduced, need to add this
	if type(measureList) == list:																		# better offset checking here
		startMeas = measureList[0]																		# one possibility: put offsets on
		endMeas = measureList[-1]																		# BEFORE getting the measures via get_measures!
	else:
		startMeas = endMeas = measureList
	if addRoman:
		add_roman(chorale, analysisPath, startMeas, endMeas)
	output = get_measures(chorale, startMeas, endMeas)
	if twoStaves:
		output = two_staves(get_measures(chorale, startMeas, endMeas))
	return output

def add_roman(chorale, anal, startMeas, endMeas):							# NOTE: assumes that you are passing a complete score in CHORALE!		
	if startMeas == 1 or startMeas == 0:
		startOffset = 0
	else:
		startOffset = anal.parts[0].elementOffset(anal.parts[0].measure(startMeas))
	try:
		endOffset = anal.parts[0].elementOffset(anal.parts[0].measure(endMeas))
	except:
		endMeas = endMeas - 1
		endOffset = anal.parts[0].elementOffset(anal.parts[0].measure(endMeas))
	theMeasures = anal.parts[0].getElementsByOffset(startOffset, endOffset, includeEndBoundary = True, classList = ['Measure'])
	theRNs = theMeasures.flat.getElementsByClass('RomanNumeral')
	oldKeyStr = '?'
	rnCount = 0
#		output.show('text')
	for RN in theRNs:															# doesn't always work!!!!
		RNoffset = RN.offset
		try:
			temp = chorale.parts[len(chorale.parts) - 1].getElementsByOffset(RNoffset, RNoffset, includeEndBoundary = True, mustBeginInSpan = False, classList = ['Measure'])
			if not temp:
				print ("problem here!!!!", startMeas, endMeas)
				continue
			m = temp[0]
			newOffset = RNoffset - m.offset
#			print RNoffset, RN.figure, m.offset, newOffset
			rnCount += 1
			vertPos = -105
			if rnCount % 2 == 0:
				vertPos = -125
			if RN.key.mode == 'minor':
				newKeyStr = RN.key.tonic.name.lower() + ': '
			else:
				newKeyStr = RN.key.tonic.name + ': '
			if newKeyStr == oldKeyStr:
				figure = RN.figure
			else:
				figure = newKeyStr + RN.figure
				oldKeyStr = newKeyStr
			insert_text(m, figure, vertPos, newOffset, justify = 'center')
		except:
			print (RNoffset, startOffset, "ERRROR!!!!?")
			print ('CHORALE')
			chorale.show('text')
			print ('RNs')
			theRNs.show('text')
			
def fix_roman_placement(chorale):
	if len(chorale.parts) < 2: return chorale
	for i in range(len(chorale.parts[0])):								# i should be measure numbers
		foundLabels = False
		for j in reversed(range(1, len(chorale.parts) - 1)):				# j is parts
			if foundLabels: break
			for item in chorale.parts[j][i]:
				if 'TextExpression' in item.classes:
					foundLabels = True
					chorale.parts[-1][i].insert(item.offset, item)
					chorale.parts[j][i].remove(item)
	return chorale

def insert_text(measure, message, position = 55, offset = 0, justify = 'center'):					# 55 is good vertical for headers
	myLabel = expressions.TextExpression(message)												# error check for offsets > measure duration!
	myLabel.style.absoluteY = position
	myLabel.style.justfy = 'center'																	# fix this
	measure.insert(offset, myLabel)

def copy_text_expressions(oldPart, newPart):
	tExpressions = oldPart.flat.getElementsByClass(['TextExpression'])
	for tExp in tExpressions:
		m = newPart.getElementsByOffset(tExp.offset, tExp.offset + .01, classList = ['Measure'], mustBeginInSpan = False)[0]
		m.insert(tExp.offset - m.offset, tExp)

def two_staves(chorale):									
	output = stream.Score()
	pTreb = stream.Part()
	pBass = stream.Part()
	if len(chorale.parts) > 4:
		newBass = chorale.parts[3:].chordify()
		copy_text_expressions(chorale.parts[-1], newBass)
		newChorale = stream.Score()
		for i in range(3):
			newChorale.insert(0, chorale.parts[i])
		newChorale.insert(0, newBass)
		chorale = newChorale
	offsetList = sorted(chorale.parts[0].measureOffsetMap().keys())
	for voice in [0, 2]:
		for properOffset in offsetList:
			v1 = stream.Voice()
			v2 = stream.Voice()
			m1 = chorale.parts[voice].getElementsByOffset(properOffset, properOffset + .05, classList = ['Measure'], mustBeginInSpan = False)[0]
			m2 = chorale.parts[voice + 1].getElementsByOffset(properOffset, properOffset + .05, classList = ['Measure'], mustBeginInSpan = False)[0]
			mInd = chorale.parts[voice].index(m1)
			ts = chorale.parts[voice][mInd].getContextByClass('TimeSignature')
			#print ts, ts.barDuration.quarterLength
			m = stream.Measure()
			mDuration = m2.duration.quarterLength
			baseOffset = m1.elements[0].offset							# Why do I need to do this at all?  And why only for m1?????
			for e in m1.elements:
				newOffset = e.offset - baseOffset						# EEEEK!   Why is this needed?????
				if 'Note' in e.classes or 'Chord' in e.classes:
					e.stemDirection = 'up'
					if e.beams: e.beams.beamsList = []			
					if e.expressions and not voice == 0: e.expressions = []				# ask Myke how to set Fermata vertical position
					v1.insert(newOffset, e)					
				elif 'Rest' in e.classes:
					v1.insert(newOffset, e)
				elif 'MetronomeMark' in e.classes:
					pass
				elif 'TextExpression' in e.classes:
					m.insert(newOffset, e)
				elif 'Clef' in e.classes:
					pass								
				else:
					m.insert(newOffset, e)									# text expressions, key signatures, clefs outside of voices
			baseOffset = m2.elements[0].offset
			for e in m2.elements:
				newOffset = e.offset - baseOffset
				if 'Note' in e.classes or 'Chord' in e.classes:
					e.stemDirection = 'down'
					if e.beams: e.beams.beamsList = []
					if e.expressions: e.expressions = []				
					v2.insert(newOffset, e)
				elif 'Rest' in e.classes:
					v2.insert(newOffset, e)
				elif 'TextExpression' in e.classes: m.insert(newOffset, e)
			try:
				m.timeSignature = ts
			except:
				pass
			m.insert(0, v1)
			m.insert(0, v2)
#			m.makeBeams(inPlace = True)	
#			m.number = m1.number
			if voice == 0:
				pTreb.insert(properOffset, m)
			else:
				pBass.insert(properOffset, m)
	output.append(pTreb)
	output.append(pBass)
	brace = layout.StaffGroup([pTreb, pBass], name='SATB', abbreviation='SATB', symbol='brace')
	brace.barTogether = True
	output.insert(0, brace)
	return output
	
"""
================================================================
			Turn offsets into measures and beats
================================================================
"""

def get_measure_number_and_beat_from_offset(myStream, myOffset, fixZeros = True, printErrors = False):
	"""This was something that either got incorporated into or replaced by a builtin music21 routine"""
	try:
		temp = myStream.beatAndMeasureFromOffset(myOffset)
	except:
		return False
	return (temp[1].number, temp[0])
	

def beat_strength(myStream, myOffset, defaultSignature = '4/2', printErrors = True):
	try:
		result = myStream.beatAndMeasureFromOffset(myOffset)
	except:
		if printErrors: 
			print("ERROR, beat_strength can't find any measure")
		return False
	measureBeat, myMeasure = result
	timeSig = myMeasure.getContextByClass('TimeSignature')
	if not timeSig:
		timeSig = meter.TimeSignature(defaultSignature)
	i = (measureBeat - 1) % timeSig.barDuration.quarterLength
	return timeSig.getAccentWeight(i, forcePositionMatch = True)