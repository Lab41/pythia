from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

def get_perform_metrics(test_labels, predicted_labels):
    '''
    Simple tool to calculate some performance metrics for the

    Args:
        raw_text: Original text to clean and normalize

    Returns:
        clean_text: Cleaned text
    '''
    results = {}

    results['rmse'] = sqrt(mean_squared_error(test_labels, predicted_labels))
    results['mae'] = mean_absolute_error(test_labels,predicted_labels)

    return results
