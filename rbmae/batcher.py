import numpy as np

class Batcher:
    def __init__(*args, batch_size=100):
        if len(args) == 0:
            raise ValueError('Batcher must take at least one argument')

        for typecheck in  map(lambda x: type(x) == np.ndarray, args):
            if not typecheck:
                raise TypeError('Type of all arguments must be np.ndarray')

        n_ = args[0].shape
        for n in map(lambda x: x.shape[0], args[1:]):
            if n != n_:
                raise ValueError('Shape[0] must be the same for all arguments')

        self.batch_size = batch_size

        self._n = n_
        self._args_views = args
        self._indices = np.arange(self._n)
        self._ptr = 0

        def reset(self):
            self._ptr = 0

        def shuffle(self):
            np.random.shufle(self._indices)
            self._args_views = list(map(lambda x: x[self._indices], self._args_views))

        @property
        def has_next_batch(self):
            return self._ptr < self._n

        @property
        def next_batch(self):
            start = self._ptr
            self._ptr += self.batch_size
            end = self._ptr
            return tuple(map(lambda x: x[start:end], self._args_view))



class Batcher2:
    def __init__(self, data, labels, *, batch_size=100):
        assert data.shape[0] == labels.shape[0]

        self.batch_size = batch_size

        self._n_entries = data.shape[0]
        self._data_view = data
        self._labels_view = labels
        self._indices = np.arange(self._n_entries)
        self._ptr = 0

    def reset(self):
        self._ptr = 0

    def next_batch(self):
        start = self._ptr
        self._ptr += self.batch_size
        end = self._ptr
        return self._data_view[start:end], self._labels_view[start:end]

    @property
    def has_next_batch(self):
        return self._ptr < self._n_entries

    def shuffle(self):
        np.random.shuffle(self._indices)
        self._data_view = self._data_view[self._indices]
        self._labels_view = self._labels_view[self._indices]
        self.reset()
