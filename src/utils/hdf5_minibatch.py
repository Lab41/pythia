import h5py

class Hdf5Batches:
    def __init__(self, hdf5_file_path, hdf5_path="/", batch_size=128):
        self.batch_size = batch_size
        self.batch_index = 0
        self.hdf5_object = h5py.File(hdf5_file_path)
        self.hdf5_data = self.hdf5_object[hdf5_path]

    def __iter__(self):
        return self

    def __next__(self):
        next_batch = self.hdf5_data[self.batch_size * self.batch_index:
                                    self.batch_size * (self.batch_index + 1)]
        if next_batch.shape[0] == 0:
            raise StopIteration
        else:
            self.batch_index += 1
            return next_batch

class Hdf5BatchIterator:
    def __init__(self, hdf5_file_path, hdf5_path="/", batch_size=128):
        self.batch_size = batch_size
        self.hdf5_file_path = hdf5_file_path
        self.hdf5_path = hdf5_path

    def __iter__(self):
        return Hdf5Batches(self.hdf5_file_path, self.hdf5_path, self.batch_size)


