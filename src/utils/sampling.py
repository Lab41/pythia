#!/usr/bin/env python
import numpy.random
import sys
from src.pipelines.parse_json import count_vocab, order_vocab
from collections import defaultdict, namedtuple, OrderedDict

def sample(data, key, novelToNotNovelRatio = 1.0, over = False, replacement = False, random_state = numpy.random):
	'''
		This function samples a set of data with two classes based on a ratio.

		Args:
			data (list of dicts): The data to be sampled. Should be a list of dictionaries.
			key (str): The key in the dictionaries that shows which class the observation belongs. data[n][key] should equal "True" or "False" 
			novelToNotNovelRatio (double): Ratio of novel observations to non-novel observations. Defaults to 1.0
			over (bool): Boolean parameter of whether to over-sample (True) or under-sample (False). Defaults to False
			replacement (bool): Boolean parameter of whether to use replacement when sampling the data. Defaults to False. Note: replacement WILL be used to make the ratio if over-sampling is True

		Returns:
			returnData: List of dictionaries with sampling of the data based on the parameters set when calling the function.
			clusters: Set of cluster ids in the resampled data
			order: Dictionary holding the order of arrival for each cluster in the resampled data
			corpusdict: Ordered Dictionary holding the most commonly occurring words in the corpus
	'''
	# Get the novel and nonNovel data
	novelObs = [w for w in data if w[key] == True]
	nonNovelObs = [v for v in data if not v[key] == True]
 
	# Find the maximum and minimum lengths for number of observations
	maximum = max(len(novelObs), len(nonNovelObs))
	minimum = min(len(novelObs), len(nonNovelObs))
	
	# If one of the classes is 0, raise an error
	if minimum == 0:
		raise NameError("Length of one of the classes is zero")
	
	# Are there more novel observations or not
	novelLarge = True if len(novelObs) >= len(nonNovelObs) else False
	
	returnData = []

	# Do different things if we are oversampling or undersampling
	if over:
		if novelLarge:
			# If the novel observations are greater, set the returnData to the novel observations, 
			# and select the non-novel observations based on the ratio.
			for thing in novelObs: returnData.append(thing)
			tempNonNovel = random_state.choice(nonNovelObs, round(maximum / novelToNotNovelRatio), True)
			for thing in tempNonNovel: returnData.append(thing)
		else:
			# If the novel observations are fewer, set the returnData to the non-novel observations,
			# and select the novel observations based on the ratio.
			for thing in nonNovelObs: returnData.append(thing)		
			tempNovel = random_state.choice(novelObs, round(maximum * novelToNotNovelRatio), True)
			for thing in tempNovel: returnData.append(thing)
	else:
		# If you will run into an error when trying to get the number of observations based on the ratio, throw an error.
		if  minimum * novelToNotNovelRatio > maximum or minimum / novelToNotNovelRatio > maximum:
			raise NameError("Ratio doesn't work with data and parameters")
		elif novelLarge:
			# If the novel observations are larger, select data from the two classes based on the minimum number and ratio
			for thing in nonNovelObs: returnData.append(thing)
			tempNovel = random_state.choice(novelObs, round(minimum * novelToNotNovelRatio), replacement)
			for thing in tempNovel: returnData.append(thing)			
		else:
			# If the novel observations are fewer, select data from the two classes based on the minium number and ratio
			for thing in novelObs: returnData.append(thing)
			tempNonNovel = random_state.choice(nonNovelObs, round(minimum / novelToNotNovelRatio), replacement)
			for thing in tempNonNovel: returnData.append(thing)
			
	# Rebuild corpus metadata for clusters, arrival order and word frequency due to corpus changes after resampling
	totalTrue = 0
	totalFalse = 0
	clusters = set()
	order = defaultdict(set)
	wordcount = defaultdict(int)
	i = 0
	
	for item in returnData:
	    if item['novelty'] is True: totalTrue+=1
	    else: totalFalse+=1
	    clusters.add(item["cluster_id"])
	    order[item["cluster_id"]].add((item["order"],i))
	    wordcount = count_vocab(item["body_text"], wordcount)
	    i+=1
	    
	wordorder = order_vocab(wordcount)
	corpusdict = OrderedDict()
	corpusTuple = namedtuple('corpusTuple','count, id')
	for word in wordorder:
	    corpusdict[word] = corpusTuple(wordcount[word], wordorder[word])
	
	print("Original Novel: %d, Non-novel: %d" % (len(novelObs), len(nonNovelObs)), file=sys.stderr)
	print("Resampled Novel: %d, Non-novel: %d" % (totalTrue, totalFalse), file=sys.stderr)

	return returnData, clusters, order, corpusdict
