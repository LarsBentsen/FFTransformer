from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_wind_data, Dataset_wind_data_graph, collate_graph


data_dict = {
    'Wind': Dataset_wind_data,
    'WindGraph': Dataset_wind_data_graph
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        n_closest=args.n_closest,
        all_stations=args.all_stations,
        data_step=args.data_step,
        min_num_nodes=args.min_num_nodes,
    )

    print(flag, len(data_set))
    if args.data == 'WindGraph':
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=collate_graph,
        )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

    return data_set, data_loader
