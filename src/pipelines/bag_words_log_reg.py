from src.utils import normalize, performance_metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn.datasets import fetch_20newsgroups
import argparse

def run_model(train_data_text, train_labels, test_data_text, test_labels):
    '''
    Algorithm which will take in a set of training text and labels to train a bag of words model
    This model is then used with a logistic regression algorithm to predict the labels for a second set of text

    Method modified from code available at:
    https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words

    Args:
        train_data_text: Text training set.  Needs to be iterable
        train_labels: Training set labels
        test_data_text: The text to

    Returns:
        pred_labels: The predicted labels as determined by logistic regression
    '''

    #First clean the data, removing HTML, Stopwords and putting everything to be lowercase
    clean_train_text = []
    clean_test_text = []

    for raw_text in train_data_text:
        clean_train_text.append(normalize.text_to_words(raw_text))

    for raw_text in test_data_text:
        clean_test_text.append(normalize.text_to_words(raw_text))

    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    #http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = 5000)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of strings.
    train_data_features = vectorizer.fit_transform(clean_train_text)

    # Numpy arrays are easy to work with, so convert the result to an array
    train_data_features = train_data_features.toarray()

    #Also make the test data into the correct format
    #Now only transform is used as the vocabulary is learned from the training data
    test_data_features = vectorizer.transform(clean_test_text)

    # Numpy arrays are easy to work with, so convert the result to an array
    test_data_features = test_data_features.toarray()

    #Now with the data in the correct formats, use Logistic Regression to train a model
    logreg = linear_model.LogisticRegression(C=1e5)
    # we create an instance of Neighbours Classifier and fit the data.
    logreg.fit(train_data_features, train_labels)
    #Now that we have something trained we can check if it is accurate with the test set
    pred_labels = logreg.predict(test_data_features)

    perform_results = performance_metrics.get_perform_metrics(test_labels, pred_labels)
    #Perform_results is a dictionary, so we should add other pertinent information to the run
    perform_results['vector'] = 'Bag_of_Words'
    perform_results['alg'] = 'Logistic_Regression'

    return pred_labels, perform_results


# Set up command line flag handling
parser = argparse.ArgumentParser(
        description="Run a Bag of Words with Logistic Regression for Text Classification",
    )

# Run only if this script is being called directly
if __name__ == "__main__":

    args = parser.parse_args()

    #Example run uses the 20 NewsGroups data as it is very easily obtainable
    #Here we will do a test by seeing if articles are in the Hockey or Space category

    categories = [
            'rec.sport.hockey',
            'sci.space'
        ]
    remove = ('headers', 'footers', 'quotes')
    data_train = fetch_20newsgroups(subset='train', categories=categories,
                                    shuffle=True, random_state=42,
                                    remove=remove)

    data_test = fetch_20newsgroups(subset='test', categories=categories,
                                   shuffle=True, random_state=42,
                                   remove=remove)

    print("Starting Bag of Words Algorithm")

    predicted_labels, perform_results = run_model(data_train.data, data_train.target, data_test.data, data_test.target)

    print("Algorithm details and results:")
    print(perform_results)