#!/usr/bin/env python
from numpy.random import choice

def sample(data, key, novelToNotNovelRatio = 1.0, over = False, replacement = False):
	'''
		This function samples a set of data with two classes based on a ratio.

		Args:
			data (list of dicts): The data to be sampled. Should be a list of dictionaries.
			key (str): The key in the dictionaries that shows which class the observation belongs. data[n][key] should equal "True" or "False" 
			novelToNotNovelRatio (double): Ratio of novel observations to non-novel observations. Defaults to 1.0
			over (bool): Boolean parameter of whether to over-sample (True) or under-sample (False). Defaults to False
			replacement (bool): Boolean parameter of whether to use replacement when sampling the data. Defaults to False. Note: replacement WILL be used to make the ratio if over-sampling is True

		Returns:
			Sampling of the data based on the parameters set when calling the function.
	'''
	
	# Get the novel and nonNovel data
	novelObs = [w for w in data if w[key] == "True"]
	nonNovelObs = [v for v in data if not v[key] == "True"]
	
	# Find the maximum and minimum lengths for number of observations
	maximum = max(len(novelObs), len(nonNovelObs))
	minimum = min(len(novelObs), len(nonNovelObs))
	
	# If one of the classes is 0, raise an error
	if minimum == 0::
		raise NameError("Length of one of the classes is zero")
	
	# Are there more novel observations or not
	novelLarge = True if len(novelObs) >= len(nonNovelObs) else False
	
	returnData = []

	# Do different things if we are oversampling or undersampling
	if over:
		if novelLarge:
			# If the novel observations are greater, set the returnData to the choice of novel observations, 
			# and select the non-novel observations based on the ratio.
			returnData = choice(novelObs, maximum, replacement)
			tempNonNovel = choice(nonNovelObs, round(maximum / novelToNotNovelRatio), True)
			for thing in tempNonNovel:
				returnData.append(thing)
		else:
			# If the novel observations are fewer, set the returnData to the choice of non-novel observations,
			# and select the novel observations based on the ratio.
			returnData = choice(nonNovelObs, maximum, replacement)
			tempNovel = choice(novelObs, round(maximum * novelToNotNovelRatio), True)
			for thing in tempNovel:
				returnData.append(thing)
	else:
		# If you will run into an error when trying to get the number of observations based on the ratio, throw an error.
		if  minimum * novelToNotNovelRatio > maxium or minimum / novelToNotNovelRatio > maximum:
			raise NameError("Ratio doesn't work with data and parameters")
		elif novelLarge:
			# If the novel observations are larger, select data from the two classes based on the minimum number and ratio
			returnData = choice(novelObs, round(minimum * novelToNotNovelRatio), replacement)
			tempNonNovel = choice(nonNovelObs, minimum, replacement)
			for thing in tempNonNovel:
				returnData.append(thing)
		else:
			# If the novel observations are fewer, select data from the two classes based on the minium number and ratio
			returnData = choice(novelObs, minimum, replacement)
			tempNonNovel = choice(nonNovelObs, round(minimum / novelToNotNovelRatio), replacement))
			for thing in tempNonNovel:
				returnData.append(thing)
	return returnData
