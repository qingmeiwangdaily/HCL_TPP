import torch

from data_io import prepare_dataloader, shift_and_superpose, sampling_positive_seqs, sampling_negative_seqs
import os


if __name__ == '__main__':
    # import ipdb; ipdb.set_trace()
    folder_name = os.path.join('tpp-data', 'data_retweet')
    batch_size = 3

    dataloader, num_types = prepare_dataloader(folder_name, batch_size)
    print(num_types)

    for (idx, batch) in enumerate(dataloader['train']):
        if idx < 1:
            print(idx)
            print('Time', batch[0].shape)
            print(batch[0])
            print('Event', batch[2].shape)
            print(batch[2])

            # test superposition-based data augmentation
            new_type, new_time = shift_and_superpose(batch[2], batch[0])
            print('New Time', new_time.shape)
            print(new_time)
            print('New Event', new_type.shape)
            print(new_type)
            print(torch.sum(new_type != 0, dim=1))

            # test model-guided thinning
            significance = torch.rand_like(batch[0]) * (batch[2] > 0)
            pos_type, pos_time = sampling_positive_seqs(batch[2], batch[0], significance=significance, ratio_remove=0.2)
            neg_type, neg_time = sampling_negative_seqs(batch[2], batch[0], num_neg=4, ratio_remove=0.5)
            print('Pos Time', pos_time.shape)
            print(pos_time)
            print('Pos Event', pos_type.shape)
            print(pos_type)
            print(torch.sum(pos_type != 0, dim=1))

            print('Neg Time', neg_time.shape)
            print(neg_time)
            print('Neg Event', neg_type.shape)
            print(neg_type)
            print(torch.sum(neg_type != 0, dim=1))

        else:
            break
