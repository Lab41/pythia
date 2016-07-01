#!/usr/bin/env python

import hashlib
import math

#This code provides a piecewise hashing of documents with variables to set the window_size for each piece, whether to split on words or characters, whether to provide sliding, and how big the sliding_window should be.

#Functions

#hashDocument:
	#Parameters:
		#text: text of document to be hashed
		#window_size: default of 25; number of characters or words to hash at a time
		#words: default of False; hash by characters = False; hash by words = True
		#sliding: default of True; slide the window and overlap the hashes or have no overlap
		#sliding_window: default of 4; ignored if sliding = False; must be less than the size of window_size
	#Description: Hashes a text document using a window_size to hash piecewise. Provide string of text document and string of piecewise hash for that text document will be returned.

#splitText:
	#Parameters:
		#text: text of document to be hashed
		#window_size: default of 25; number of characters or words to hash at a time
		#words: default of False; hash by characters = False; hash by words = True
		#sliding: default of True; slide the window and overlap the hashes or have no overlap
		#sliding_window: default of 4; ignored if sliding = False; must be less than the size of window_size
	#Description: Splits the text into the sizes needed for hashing. Returns a list of strings split based on the parameters provided

def hashDocument(text, window_size = 25, words = False, sliding = True, sliding_window = 4):
	#Error if sliding_window is bigger than or equal to window_size and sliding is turned on
	if sliding_window >= window_size and sliding:
		raise ValueError("sliding_window must be smaller than window_size;")
	returnText = ""
	#
	#Check to see if document length is less than window_size and just hash the whole doc if that's the case
	if len(text) < window_size:
		m = hashlib.md5()
		m.update(text.encode("utf-8"))
		return str(m.digest())[2:].strip().strip("'").strip()
	#Split text into distinct blocks for hashing
	textSplit = splitText(text, window_size, words, sliding, sliding_window)
	#
	#Md5 hash each part of the text and return the text as a long string of these hashes
	for item in textSplit:
		m = hashlib.md5()
		m.update(item.encode("utf-8"))
		returnText += str(m.digest())[2:].strip().strip("'").strip()
	return returnText

def splitText(text, window_size, words, sliding, sliding_window):
	textSplit = []
	if words:
		tempSplit = text.split(" ")
		if sliding:
			#Grab each window of window_size words from the text sliding sliding_window over each iteration.
			textSplit = [" ".join(tempSplit[i:i+window_size]) for i in range(0, len(tempSplit), sliding_window) if (i + window_size) < (len(tempSplit) + sliding_window)] 
		else:
			#Grab each window of window_size words from the text without overlap
			textSplit = [" ".join(tempSplit[i:i+window_size]) for i in range(0, len(tempSplit), window_size)] 
	else:
		if sliding:
			#Grab each window of window_size characters from the text sliding sliding_window over each iteration.
			textSplit = [text[i:i+window_size] for i in range(0, len(text), sliding_window) if (i + window_size) < (len(text) + sliding_window)] 
		else:
			#Grab each window of window_size characters from the text without overlap
			textSplit = [text[i:i+window_size] for i in range(0, len(text), window_size)] 
	return textSplit
