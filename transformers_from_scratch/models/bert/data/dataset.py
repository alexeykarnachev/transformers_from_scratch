from multiprocessing import Queue

from torch.utils.data.dataset import IterableDataset


class Dataset(IterableDataset):
    def __init__(self, inp_samples_queue: Queue):
        self._inp_samples_queue = inp_samples_queue

    def __iter__(self):
        while True:
            sample = self._inp_samples_queue.get()
            yield sample
