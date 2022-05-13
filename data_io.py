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


# Data augmentation operation
def sorting(event_type: torch.Tensor, event_time: torch.Tensor):
    """
    Sort each sequence in the ascending order of time
    """
    batch = event_time.shape[0]
    idx = torch.argsort(event_time, dim=1, descending=False)
    for b in range(batch):
        event_time[b, :] = event_time[b, idx[b, :]]
        event_type[b, :] = event_type[b, idx[b, :]]
    return event_type, event_time


def reorganize(event_type: torch.Tensor, event_time: torch.Tensor):
    """
    Move zeros to the end of each sequence
    """
    batch = event_time.shape[0]
    for b in range(batch):
        tmp_time = event_time[b, :]
        tmp_type = event_type[b, :]
        event_time[b, :] = torch.cat((tmp_time[tmp_type != 0], tmp_time[tmp_type == 0]), dim=0)
        event_type[b, :] = torch.cat((tmp_type[tmp_type != 0], tmp_type[tmp_type == 0]), dim=0)
    len_max = torch.max(torch.sum(event_type != 0))
    return event_type[:, :len_max], event_time[:, :len_max]


def shift_and_superpose(event_type: torch.Tensor, event_time: torch.Tensor):
    """
    Superpose event sequences with their shifted versions

    Input: event_type: batch*seq_len;
           event_time: batch*seq_len.
    Output: new event_type: batch * (2 * seq_len)
            new event_time: batch * (2 * seq_len)
    """
    batch = event_time.shape[0]
    shift = int(batch / 2)
    shift_type = torch.cat((event_type[shift:, :], event_type[:shift, :]), dim=0)
    shift_time = torch.cat((event_time[shift:, :], event_time[:shift, :]), dim=0)
    new_type = torch.cat((event_type, shift_type), dim=1)  # batch * (2seq_len)
    new_time = torch.cat((event_time, shift_time), dim=1)  # batch * (2seq_len)
    new_type, new_time = sorting(new_type, new_time)
    # idx = torch.argsort(new_time, dim=1, descending=False)
    # for b in range(batch):
    #     new_time[b, :] = new_time[b, idx[b, :]]
    #     new_type[b, :] = new_type[b, idx[b, :]]
    #     tmp_time = new_time[b, idx[b, :]]
    #     tmp_type = new_type[b, idx[b, :]]
    #     new_time[b, :] = torch.cat((tmp_time[tmp_type != 0], tmp_time[tmp_type == 0]), dim=0)
    #     new_type[b, :] = torch.cat((tmp_type[tmp_type != 0], tmp_type[tmp_type == 0]), dim=0)
    return reorganize(new_type, new_time)


def thinning_process_deterministic(event_type: torch.Tensor, event_time: torch.Tensor,
                                   significance: torch.Tensor, ratio_remove: float = 0.2):
    """
    Deterministically thinning event sequences guided by a probability/significance
    Input: event_type: batch * seq_len;
           event_time: batch * seq_len;
           significance: batch * seq_len, the significance of the events per sequence
           num_neg: the number of negative sequences per observed sequence
           ratio_remove: the ratio of removed events per sequence, in the range (0, 1)
    Output: thinned event_type: batch * [seq_len * (1 - ratio_remove)]
            thinned event_time: batch * [seq_len * (1 - ratio_remove)]
    """
    batch, seq_len = event_type.shape
    new_len = int(seq_len * (1 - ratio_remove))
    idx = torch.argsort(significance, dim=1, descending=True)
    thinned_type = torch.zeros_like(event_type)
    thinned_time = torch.zeros_like(event_time)
    for b in range(batch):
        num_events = torch.sum(event_type[b, :] > 0)
        new_len_b = int(num_events * (1 - ratio_remove))
        thinned_type[b, :new_len_b] = event_type[b, idx[b, :new_len_b]]
        thinned_time[b, :new_len_b] = event_time[b, idx[b, :new_len_b]]
    new_type, new_time = sorting(thinned_type[:, :new_len], thinned_time[:, :new_len])
    return reorganize(new_type, new_time)


def thinning_process_random(event_type: torch.Tensor, event_time: torch.Tensor, ratio_remove: float = 0.5):
    """
    Randomly thinning event sequences with a relatively-high removal ratio
        Input: event_type: batch * seq_len;
               event_time: batch * seq_len;
               num_neg: the number of negative sequences per observed sequence
               ratio_remove: the ratio of removed events per sequence, in the range (0, 1)
        Output: thinned event_type: batch * [seq_len * (1 - ratio_remove)]
                thinned event_time: batch * [seq_len * (1 - ratio_remove)]
    """
    rv = torch.rand_like(event_time) * (event_type > 0)
    return thinning_process_deterministic(event_type, event_time, rv, ratio_remove)


def sampling_positive_seqs(event_type: torch.Tensor, event_time: torch.Tensor,
                           significance: torch.Tensor, ratio_remove: float = 0.2):
    """
    Sampling positive sequences based on the significance of the events guided by the model
    """
    return thinning_process_deterministic(event_type, event_time, significance, ratio_remove)


def sampling_negative_seqs(event_type: torch.Tensor, event_time: torch.Tensor,
                           num_neg: int = 5, ratio_remove: float = 0.5):
    """
    Sampling negative sequences by randomly removing some events
    """
    neg_type, neg_time = thinning_process_random(event_type, event_time, ratio_remove)
    for k in range(num_neg - 1):
        tmp_type, tmp_time = thinning_process_random(event_type, event_time, ratio_remove)
        neg_type = torch.cat((neg_type, tmp_type), dim=0)
        neg_time = torch.cat((neg_time, tmp_time), dim=0)
    return neg_type, neg_time
