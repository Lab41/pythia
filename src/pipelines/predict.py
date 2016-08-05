from src.utils import performance_metrics


def predicter(classifier, test_data, test_labels):
    pred_labels = classifier.predict(test_data)
    perform_results = performance_metrics.get_perform_metrics(test_labels, pred_labels)
    return pred_labels, perform_results


def main(argv):
    data, labels = predicter(argv[0], argv[1], argv[2])
    return data, labels
