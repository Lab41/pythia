from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_recall_fscore_support, accuracy_score, zero_one_loss

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
    results['prfs'] = precision_recall_fscore_support(test_labels, predicted_labels)
    results['accuracy'] = accuracy_score(test_labels, predicted_labels)
    results['zero-one-loss'] = zero_one_loss(test_labels, predicted_labels)

    return results
