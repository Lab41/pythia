import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
from numpy import sqrt

def get_perform_metrics(test_labels, predicted_labels):
    '''
    Simple tool to calculate some performance metrics for the

    Args:
        test_labels: the actual labels for the test data
        predicted_labels: the predicted labels for the test data

    Returns:
        results: a dictionary of metric names and values
    '''
    results = {}

    results['rmse'] = sqrt(mean_squared_error(test_labels, predicted_labels))
    results['mae'] = mean_absolute_error(test_labels,predicted_labels)
    results['accuracy'] = accuracy_score(test_labels, predicted_labels)
    results['confusion matrix'] = confusion_matrix(test_labels, predicted_labels).tolist()

    target_names = ['duplicate', 'novel']
    print(classification_report(test_labels, predicted_labels, target_names=target_names),file=sys.stderr)
    prfs = precision_recall_fscore_support(test_labels, predicted_labels)
    results['precision'] = prfs[0].tolist()
    results['recall'] = prfs[1].tolist()
    results['f score'] = prfs[2].tolist()
    results['support'] = prfs[3].tolist()


    return results
