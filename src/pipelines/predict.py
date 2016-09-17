import h5py
from src.utils import performance_metrics

def predicter(classifier, test_data, test_labels):
    # Handle HDF5 case
    if type(test_data) is str:
        assert test_data==test_labels
        with h5py.File(test_data) as f:
            test_data = f['data'][()]
            test_labels = f['labels'][()]
            
    pred_labels = classifier.predict(test_data)
    perform_results = performance_metrics.get_perform_metrics(test_labels, pred_labels)
    return pred_labels, perform_results

