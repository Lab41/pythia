#!/usr/bin/env python
import sys
import logging
import numpy as np
import numpy.random
from collections import defaultdict, namedtuple, OrderedDict
from src.pipelines.parse_json import count_vocab, order_vocab

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)
def sample(data, key, novelToNotNovelRatio = 1.0, over = False, replacement = False, random_state = numpy.random):
    '''
        This function samples a set of data with two classes based on a ratio.

        Args:
            data (list of dicts): The data to be sampled. Should be a list of dictionaries.
            key (str): The key in the dictionaries that shows which class the observation belongs. 
		data[n][key] should equal "True" or "False" 
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

def label_sample(observations, label_key, replacement=False, desired_size=None, random_state = np.random, final_shuffle=True):
    """ Resample observations based on a binary key and a yield a 0.5
    proportion of cases with True responses. Provides options to 
    oversample or to downsample.
    
    Args:
        observations (list of dict): each dict should be a separate observation
        label_key (dict key): key identifying binary labels in observations
        replacement (bool): when sampling, use replacement or not?
        desired_size (int): if not None, desired size of each resampled class. If None,
            value is either: minimum class size (if replacement is False), else the maximum
            class size. If -np.Inf, sample to smaller class size.
        random_state (numpy.random.RandomState)
        final_shuffle (bool): should a final shuffle be performed on the resampled results
    """

    #key_values = [False, True]
    sorted_by_label = {}
    # Sort observations by their response value
    for case in observations:
        response_key = case[label_key]
        sorted_by_label.setdefault(response_key, []).append(case)
    logger.debug("Size of response key set {}".format(len(sorted_by_label)))
    logger.debug("Size of observations set {}".format(len(observations)))
    # What should sizes of eventual classes be?
    sizes = { key: len(x) for key, x in sorted_by_label.items() }
    min_size = np.min(list(sizes.values()))
    max_size = np.max(list(sizes.values()))
    if desired_size is None:
        if not replacement:
            desired_size = min_size
        else:
            desired_size = max_size
    if desired_size == -np.Inf:
        desired_size = min_size
    if not replacement:
        desired_size = np.min((desired_size, min_size))

    # Do actual resampling
    resampled = {k: random_state.choice(
        sorted_by_label[k], 
        size=(desired_size,), 
        replace=replacement) for k in sorted(sorted_by_label.keys()) }
    resampled = [ resampled[k] for k in sorted(resampled.keys()) ] # yields list of lists
    # unwrap nested list
    resampled = [ item for sublist in resampled for item in sublist ] 
    if final_shuffle:
        resampled = list(random_state.permutation(resampled))

    return resampled

