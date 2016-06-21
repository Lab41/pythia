from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_recall_fscore_support, accuracy_score, zero_one_loss

from math import sqrt

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
    
    prfs = precision_recall_fscore_support(test_labels, predicted_labels)
    results['presision'] = prfs[0]
    results['recall'] = prfs[1]
    results['f score'] = prfs[2]
    results['support'] = prfs[3]

    return results
