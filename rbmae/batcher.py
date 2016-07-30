import numpy as np


class Batcher:
    def __init__(self, data, labels=None, *, batch_size=100):
        assert 0 < len(data.shape) < 3 and 0 < len(labels.shape) < 3
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