import torch
from torch import Tensor, nn
from typing import Tuple
import pickle

def read_syn(file_name):
    with open(file_name, 'rb') as f:
        loaded_hawkes_data = pickle.load(f)

    mu = loaded_hawkes_data['mu']
    alpha = loaded_hawkes_data['alpha']
    decay = loaded_hawkes_data['decay']
    tmax = loaded_hawkes_data['tmax']

    print("Simulated Hawkes process parameters:")
    for label, val in [("mu", mu), ("alpha", alpha), ("decay", decay), ("tmax", tmax)]:
        print("{:<20}{}".format(label, val))

    return loaded_hawkes_data,tmax


def process_loaded_sequences(loaded_hawkes_data: dict, process_dim: int) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Preprocess synthetic Hawkes data by padding the sequences.
    Args:
        loaded_hawkes_data:
        process_dim:
        tmax:
    Returns:
        sequence event times, event types and overall lengths (dim0: batch size)
    """
    # Tensor of sequence lengths (with additional BOS event)
    seq_lengths = torch.Tensor(loaded_hawkes_data['lengths']).int()

    event_times_list = loaded_hawkes_data['timestamps']
    event_types_list = loaded_hawkes_data['types']
    event_times_list = [torch.from_numpy(e) for e in event_times_list]
    event_types_list = [torch.from_numpy(e) for e in event_types_list]

    tmax = 0
    for tsr in event_times_list:
        if torch.max(tsr) > tmax:
            tmax = torch.max(tsr)

    #  Build a data tensor by padding
    seq_times = nn.utils.rnn.pad_sequence(event_times_list, batch_first=True, padding_value=tmax).float()
    seq_times = torch.cat((torch.zeros_like(seq_times[:, :1]), seq_times), dim=1) # add 0 to the sequence beginning

    seq_types = nn.utils.rnn.pad_sequence(event_types_list, batch_first=True, padding_value=process_dim)
    seq_types = torch.cat(
        (process_dim*torch.ones_like(seq_types[:, :1]), seq_types), dim=1).long()# convert from floattensor to longtensor

    return seq_times, seq_types, seq_lengths, tmax


def one_hot_embedding(labels: Tensor, num_classes: int) -> torch.Tensor:
    """Embedding labels to one-hot form. Produces an easy-to-use mask to select components of the intensity.
    Args:
        labels: class labels, sized [N,].
        num_classes: number of classes.
    Returns:
        (tensor) encoded labels, sized [N, #classes].
    """
    device = labels.device
    y = torch.eye(num_classes).to(device)
    return y[labels]