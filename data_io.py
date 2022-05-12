import numpy as np
import pickle
import torch
import torch.utils.data
import os
from typing import List, Dict, Tuple


DEFAULT_PAD = 0


class EventData(torch.utils.data.Dataset):
    """ Event stream dataset. """
    def __init__(self, data: List[List[Dict]]):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event

        In other words, each sequence is saved as [t0, dt1, dt2, ..., dtN] + [c0, c1, c2, ..., cN],
        where each dti = ti - ti-1
        """
        # step one: convert the data to the structure List[List[List]]]
        self.time = [[elem['time_since_start'] for elem in inst] for inst in data]
        self.time_gap = [[elem['time_since_last_event'] for elem in inst] for inst in data]
        # plus 1 since there could be event type 0, but we use 0 as padding
        self.event_type = [[elem['type_event'] + 1 for elem in inst] for inst in data]
        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Tuple[List, List, List]:
        """ Each returned element is a list, which represents an event stream """
        return self.time[idx], self.time_gap[idx], self.event_type[idx]


def pad_time(insts: List[List], pad_value: int = DEFAULT_PAD) -> torch.Tensor:
    """
    Pad the instance to the max seq length in batch.
    :param: A list of event streams, each list contains a list of instances (timestamps of events)
    :param: pad_value: the value used to pad sequence
    """
    max_len = max(len(inst) for inst in insts)
    batch_seq = np.array([
        inst + [pad_value] * (max_len - len(inst))
        for inst in insts])
    return torch.tensor(batch_seq, dtype=torch.float32)


def pad_type(insts: List[List], pad_value: int = DEFAULT_PAD) -> torch.Tensor:
    """
    Pad the instance to the max seq length in batch.
    :param: A list of event streams, each list contains a list of instances (types of events)
    :param: pad_value: the value used to pad sequence
    """
    max_len = max(len(inst) for inst in insts)
    batch_seq = np.array([
        inst + [pad_value] * (max_len - len(inst))
        for inst in insts])
    return torch.tensor(batch_seq, dtype=torch.long)


def collate_fn(insts: List[List]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function, as required by PyTorch,
    which used to make the output of each sample with the same structure and size
    """
    time, time_gap, event_type = list(zip(*insts))
    time = pad_time(time)
    time_gap = pad_time(time_gap)
    event_type = pad_type(event_type)
    return time, time_gap, event_type


def get_dataloader(data: List[List[Dict]], batch_size: int, shuffle: bool = True) -> torch.utils.data.DataLoader:
    """
    Prepare dataloader.
    :param: the event streaming data with structure List[List[Dict]]
    :param: batch_size: the batch size
    :param: shuffle: shuffle the data or not.
    """
    ds = EventData(data)
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle
    )
    return dl


def load_data(name: str, dict_name: str) -> Tuple[List[List[Dict]], int]:
    """
    Load raw streaming data from a pickle file
    :param name: the name of the file
    :param dict_name: the keyword of the raw data in the file
    :return:
        1) data: the streaming data with structure List[List[Dict]]],
        each dictionary contains time_since_start, time_since_last_event, type_event
        2) The number of event types
    """
    with open(name, 'rb') as f:
        data = pickle.load(f, encoding='latin-1')
        num_types = data['dim_process']
        data = data[dict_name]
    return data, int(num_types)


def prepare_dataloader(data_folder_name: str, batch_size: int) -> Tuple[Dict[str, torch.utils.data.DataLoader], int]:
    """
    Load data and prepare dataloader.

    :param data_folder_name: a structure contains at least two items (data_folder_name, batch_size)
    :param batch_size: the batch size
    """
    print('[Info] Loading train data...')
    train_data, num_types = load_data(os.path.join(data_folder_name, 'train.pkl'), 'train')
    print('[Info] Loading dev data...')
    dev_data, _ = load_data(os.path.join(data_folder_name, 'dev.pkl'), 'dev')
    print('[Info] Loading test data...')
    test_data, _ = load_data(os.path.join(data_folder_name, 'test.pkl'), 'test')
    dataloaders = {'train': get_dataloader(train_data, batch_size, shuffle=True),
                   'val': get_dataloader(dev_data, batch_size, shuffle=False),
                   'test': get_dataloader(test_data, batch_size, shuffle=False)}
    return dataloaders, num_types


# TODO: Data augment
def random_superpose(event_type: torch.Tensor, event_time: torch.Tensor):
    """
    Randomly superpose event sequences

    :param
    """
    return None


def thinning_process(event_type: torch.Tensor, event_time: torch.Tensor, probability: torch.Tensor):
    """
    Randomly thinning event sequences to obtain positive and negative sequences
    """
    return None
