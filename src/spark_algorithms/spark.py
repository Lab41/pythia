from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
from pyspark.mllib.feature import Normalizer
from pyspark.mllib.feature import Word2Vec
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint


def word2vecModel(text):
  """Computes distributed vector representation of words using a skip-gram model. The training objective of skip-gram
  is to learn word vector representations that are good at predicting its context in the same sentence.

  :parameter text: (REQUIRED) - the input data of text words/strings you'd like to use
  :return: word2vec model

  Use it as:
  .. code-block::python

      model = word2vecModel(text)
      synonyms = model.findSynonyms('random_word', 40)
  """
  word2vec = Word2Vec()
  return word2vec.fit(text)

def SVM_module(training):
  """This function returns a SVM model from your training data.

  :param training: (REQUIRED) - the training data
  :return: SVM model

  Use it as (Be sure to call split_data() to get the training data):

  >>> model = SVM_module(trainingData)
  """
  # Train a SVM model
  return SVMWithSGD.train(training, iterations=300)

def labelize(x):
  """
  This function expects a list of tuples in the format [(label, [features])]

  :param x: list of tuples
  :return: LabeledPoint object of tuples
  """
  return LabeledPoint(x[0], x[1])

def split_data(series_features, training_percent=0.9, testing_percent=0.1, seed=0):
    """This function splits the data into training and testing data and returns both as variables.
      Parameters:
        series_features  (REQUIRED             ) - list of LabeledPoint data points
        training_percent (OPTIONAL; DEFAULT=0.9) - percentage of the data you want to train on
        testing_percent  (OPTIONAL; DEFAULT=0.1) - percentage of the data you want to test on
        seed             (OPTIONAL; DEFAULT=0  ) - what random seed you want
      """
    # Splits the data randomly into training and testing data with an optional seed
    training, testing = series_features.randomSplit([training_percent, testing_percent], seed)
    return training, testing

def naive_bayes_module(training):
    """This function returns a naive bayes model from your training data.
    Parameter:
    training (REQUIRED) - the training data
    """
    # Train a Naive Bayes model
    return NaiveBayes.train(training)

def predict_and_label(model, testing):
    """This function returns a RDD of all predictions and their corresponding labels based
    on the model provided and the testing data.
    Parameter:
    model   (REQUIRED) - the model you'd like to use
    testing (REQUIRED) - the testing data you used earlier
    """
    # Give me the prediction and their corresponding label
    return testing.map(lambda p: (model.predict(p.features), p.label))

def get_accuracy(prediction_and_label, testing):
    """This function returns the accuracy of the model.
    Parameters:
    predictionAndLabel (REQUIRED) - RDD of predictions and labels
    testing            (REQUIRED) - the testing data you used earlier
    """
    # Give me the accuracy of the model
    return 1.0 * prediction_and_label.filter(lambda x: abs(x[0] - x[1]) < .001).count() / testing.count()
