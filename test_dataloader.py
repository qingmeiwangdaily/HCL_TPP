from data_io import prepare_dataloader
import os


if __name__ == '__main__':
    # import ipdb; ipdb.set_trace()
    folder_name = os.path.join('tpp-data', 'data_retweet')
    batch_size = 3

    dataloader, num_types = prepare_dataloader(folder_name, batch_size)
    print(num_types)

    for (idx, batch) in enumerate(dataloader['train']):
        if idx < 3:
            print(idx)
            print('Time', batch[0].shape)
            print(batch[0])
            print('Time Gap', batch[1].shape)
            print(batch[1])
            print('Event', batch[2].shape)
            print(batch[2])
