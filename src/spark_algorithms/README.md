##SYNOPSIS
This is example code for implementing a word2vec model, term frequency*inverse document frequency (TF-IDF) and others.
The examples shown here use the 20 news groups data which can be found in sklearn

##word2vec
Example of how to use the word2vec model and using a K Means cluster for them.
MLLib Word2Vec() expects the text to come in as a Seq[String]

```
from math import sqrt
from sklearn.datasets import fetch_20 newsgroups
import random

# get some data to experiment with
categories = ['sci.space']
remove = ('headers', 'footers', 'quotes')
news_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42, remove=remove)

# transform the data into the format desired by word2vec
# you would likely want to have additional tokenizations for your data
text = sc.parallelize(news_data.data).map(lambda row: row.split(" "))

word2vec = Word2Vec()
model = word2vec.fit(text)

# find top 40 synonyms of the word gobierno
# synonyms = model.findSynonyms('gobierno', 40)

# for word, cosine_distance in synonyms:
#    print("{}: {}".format(word, cosine_distance))

vectors = dict(model.getVectors())
vector_list = []
for key in vectors.keys():
  vector_list.append(list(vectors[key]))
vectors_RDD = sc.parallelize(vector_list)

clusters = KMeans.train(vectors_RDD, 2000, maxIterations=10,
        runs=10, initializationMode="random")

def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = vectors_RDD.map(lambda point: error(point)).reduce(lambda x, y: x + y)

cluster_val = {}
for key in vectors.keys():
  cluster_val[key] = clusters.predict(model.transform(key))
```

##TF-IDF
Term frequency * Inverse Document Frequency in Spark. It expects a list of lists where each inner list contains every
individual word of a body of text. For instance: [ [word1, word2, word3,...], [word1, word2, word3,...], ... ]. Every index
of that list is the content of a separate article.

```
# ensure the 20 news group data is in the format desired by the algorithm
text = sc.parallelize(news_data.data)

hashingTF = HashingTF()
tf = hashingTF.transform(text)
idf = IDF().fit(tf)
tfidf = idf.transform(tf)
```

##Normalizer
Normalizer scales individual samples to have unit L^p norm. This is a common operation for text classification or
clustering. For example, the dot product of two L^2 normalized TF-IDF vectors is the cosine similarity of the
vectors.
```
normalizer = Normalizer(p=float("inf"))
```

##How to split your data into training/testing and use a SVM
```
# Combine your data with their labels so it is a tuple of [(label, [features])]
data = labels.zip(normalizer.transform(tfidf))

# Convert your data into LabelPoint object for Spark
labeled_point_SV = data.map(lambda x: labelize(x))

training, testing = split_data(labeled_point_SV, seed = random.randint(1, 30))

# Train the model
model = SVM_module(training)

# Generate predictions
predictionAndLabel = predict_and_label(model, testing)

# Evaluate the model
single_accuracy = get_accuracy(predictionAndLabel, testing)
```
