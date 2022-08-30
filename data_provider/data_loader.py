import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.timefeatures import time_features
import warnings
import torch

warnings.filterwarnings('ignore')


class Dataset_wind_data(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='S', data_path='wind_data.csv',
                 target='KVITEBJØRNFELTET', scale=True, timeenc=0, freq='10min', all_stations=False, data_step=5, **_):

        self.all_stations = all_stations        # Weather to use all the stations or only specific ones.

        self.seq_len = size[0]          # S (notation used in paper)
        self.label_len = size[1]        # L
        self.pred_len = size[2]         # P

        self.total_seq_len = self.seq_len + self.pred_len
        assert self.label_len <= self.seq_len

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.data_step = data_step    # Only use every data_step'th point. Set data_step = 1 for full dataset.

        self.features = features
        self.target = target        # The station we want to predict for if self.all_stations == False
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        _path = os.path.join(self.root_path, self.flag, self.data_path).replace('\\', '/')  # replace in case windows
        df_raw = pd.read_csv(_path, header=[0, 1])

        # Get the indices for the different stations
        self._stations = {s: i for i, s in enumerate(df_raw.columns.get_level_values(1).unique())}

        # Fit scaler based on training data only
        if self.flag != 'train':
            train_data = pd.read_csv(_path.replace(self.flag, 'train'), header=[0, 1])

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[df_raw.columns.get_level_values(0) != 'time']
            df_data = df_raw[cols_data]
            assert (df_raw.columns.get_level_values(0).unique()[-1:] == ['wind_speed']).all()
        elif self.features == 'S':
            cols_data = df_raw.columns[df_raw.columns.get_level_values(0) == 'wind_speed']
            df_data = df_raw[cols_data]

        # Only keep the relevant columns also for the training data (for which we perform scaling):
        if self.flag != 'train':
            train_data = train_data[cols_data]

        # Currently scaling based on all stations, but could be changed to scale just using the target.
        if self.scale:
            self.cols_meas = df_data.stack().columns
            if self.flag != 'train':
                self.scaler.fit(train_data.stack().values)
                del train_data      # Free up memory as train_data is no longer needed.
            else:
                self.scaler.fit(df_data.stack().values)
            # [Samples, meas, stations]
            data = df_data.values.reshape(df_data.shape[0], df_data.columns.get_level_values(0).nunique(), -1)
            data = np.stack([self.scaler.transform(data[..., i]) for i in range(data.shape[-1])], -1)
        else:
            data = df_data.values.reshape(df_data.shape[0], df_data.columns.get_level_values(0).nunique(), -1)

        if not self.all_stations:
            data = data[..., self._stations[self.target]]
            data = np.expand_dims(data, -1)

        # Find missing entries to then decide on valid sequences (which don't contain NaNs)
        nan_indxs = [np.where(np.isnan(data[..., i]).any(axis=1))[0] for i in range(data.shape[-1])]
        nan_indxs = [np.unique(np.concatenate([np.array([0]), nan_indxs[i], np.array([data[..., 0].shape[0] - 1])]))
                     for i in range(len(nan_indxs))]        # Indicate the first and last values as NaN

        # Find the slices which result in valid sequences without NaNs
        valid_slices = [np.where((nan_indxs[i][1:] - nan_indxs[i][:-1] - 1) >= self.total_seq_len)[0]
                        for i in range(len(nan_indxs))]
        valid_slices = [np.vstack([nan_indxs[i][valid_slices[i]] + 1, nan_indxs[i][valid_slices[i] + 1] - 1]).T
                        for i in range(len(nan_indxs))]

        # Now, construct an array which contains the valid start indices for the different sequences
        data_indxs = []
        for i in range(len(valid_slices)):
            start_indxs = np.zeros(data.shape[0] - self.total_seq_len + 1, dtype='bool')
            for s, e in valid_slices[i]:
                indxs_i = np.arange(s, e - self.total_seq_len + 2, 1)
                start_indxs[indxs_i] = True
            data_indxs.append(start_indxs)
        self.data_indxs = np.stack(data_indxs, -1)

        # Construct the time array
        assert df_raw[['time']].all(0).all()
        df_stamp = df_raw[['time']].iloc[:, :1]     # [border1:border2]
        df_stamp.columns = ['time']
        df_stamp['time'] = pd.to_datetime(df_stamp.time)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 10)
            data_stamp = df_stamp.drop(['time'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['time'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise ValueError("Pass timeenc as either 0 or 1")

        self.data_x = data      # The full dataset
        self.data_stamp = data_stamp    # The full time dataset
        self.valid_indxs = np.where(self.data_indxs.any(-1))[0]   # Index slices for the data
        self.valid_indxs = self.valid_indxs[::self.data_step]

        # To know which stations are available
        self.full_indx_row, self.full_indx_col = np.where(self.data_indxs[self.valid_indxs, :])

    def __getitem__(self, index):
        s_begin = self.valid_indxs[self.full_indx_row[index]]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        station = self.full_indx_col[index]

        seq_x = self.data_x[s_begin:s_end, :, station]
        seq_y = self.data_x[r_begin:r_end, :, station]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if not self.all_stations:
            return len(self.valid_indxs)
        else:
            return self.data_indxs[self.valid_indxs, :].sum()

    # Assumes inputs of shape [nodes, seq, feats] or [seq, feats]
    def inverse_transform(self, data):
        num_input_feats = data.shape[-1]
        if num_input_feats != len(self.scaler.scale_):
            data = np.concatenate([np.zeros([*data.shape[:-1], len(self.scaler.scale_) - data.shape[-1]]), data], -1)
        data = self.scaler.inverse_transform(data)

        if num_input_feats != len(self.scaler.scale_):
            data = data[..., -num_input_feats:]

        return data


class Dataset_wind_data_graph(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='S', data_path='wind_data.csv',
                 target='KVITEBJØRNFELTET', scale=True, timeenc=0, freq='10min', subset=False,
                 n_closest=None, data_step=5, min_num_nodes=2, **_):

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.data_step = data_step   # Only use every data_step'th point. Set data_step = 1 for full dataset.

        self.total_seq_len = self.seq_len + self.pred_len
        assert self.label_len <= self.seq_len
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.min_num_nodes = min_num_nodes      # Minimum number of nodes in a graph
        self.target = target      # If there is a station we want to always have in data (i.e. a target station)
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        # Set to None if we want to have fully connected graphs. If n_closest = 3, then we only allow the three
        # closest nodes to send to a particular node.
        self.n_closest = n_closest

        # If we want to only consider a subset of the stations. Just change the names to change the stations we want
        # to consider or None if we want to predict for all stations.
        if subset:
            self.subset = [
                'SNORREA',
                'SNORREB',
                'VISUNDFELTET',
                'KVITEBJØRNFELTET',
                'HULDRAFELTET',
                'VESLEFRIKKA',
                'OSEBERGC',
                'BRAGE',
                'OSEBERGSØR',
                'TROLLB',
                'GJØAFELTET'
            ]
        else:
            self.subset = None

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        _path = os.path.join(self.root_path, self.flag, self.data_path).replace('\\', '/')  # replace in case windows
        df_raw = pd.read_csv(_path, header=[0, 1])

        # Get the indices for the different stations
        self._stations = {s: i for i, s in enumerate(df_raw.columns.get_level_values(1).unique())}
        self._stations_inv = {v: k for k, v in self._stations.items()}

        # Load the static edge features:
        edge_feats = pd.read_csv(os.path.join(self.root_path, 'edge_feats.csv').replace('\\', '/'),
                                 header=[0, 1], index_col=0)
        station_info = pd.read_csv(os.path.join(self.root_path, 'station_info.csv').replace('\\', '/'))
        station_info = station_info[['id', 'lat', 'lon', 'name']]
        station_info['name'] = station_info['name'].apply(lambda x: x.replace(' ', ''))
        self.station_info = station_info
        self.edge_feats = edge_feats[edge_feats.columns[edge_feats.columns.get_level_values(0) == self.target]]
        self.edge_feats.columns = self.edge_feats.columns.get_level_values(1)

        # Fit scaler on training data
        if self.flag != 'train':
            train_data = pd.read_csv(_path.replace(self.flag, 'train'), header=[0, 1])

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[df_raw.columns.get_level_values(0) != 'time']
            df_data = df_raw[cols_data]
            assert (df_raw.columns.get_level_values(0).unique()[-1:] == ['wind_speed']).all()
        elif self.features == 'S':
            cols_data = df_raw.columns[df_raw.columns.get_level_values(0) == 'wind_speed']
            df_data = df_raw[cols_data]

        # Only keep the relevant columns also for the training data (for which we perform scaling):
        if self.flag != 'train':
            train_data = train_data[cols_data]

        # Currently scaling based on all stations, but could be changed to scale just using the subset (if relevant).
        if self.scale:
            self.cols_meas = df_data.stack().columns
            if self.flag != 'train':
                self.scaler.fit(train_data.stack().values)
                del train_data      # Free up memory as train_data is no longer needed.
            else:
                self.scaler.fit(df_data.stack().values)
            # [Samples, meas, stations]
            data = df_data.values.reshape(df_data.shape[0], df_data.columns.get_level_values(0).nunique(), -1)
            data = np.stack([self.scaler.transform(data[..., i]) for i in range(data.shape[-1])], -1)
        else:
            data = df_data.values.reshape(df_data.shape[0], df_data.columns.get_level_values(0).nunique(), -1)

        if self.subset is not None:
            subset_indxs = [self._stations[s] for s in self.subset]
            data = data[..., subset_indxs]

        # Find missing entries to then decide on valid sequences (which don't contain NaNs)
        nan_indxs = [np.where(np.isnan(data[..., i]).any(axis=1))[0] for i in range(data.shape[-1])]
        nan_indxs = [np.unique(np.concatenate([np.array([0]), nan_indxs[i], np.array([data[..., 0].shape[0] - 1])]))
                     for i in range(len(nan_indxs))]

        # Find the slices which result in valid sequences without NaNs
        valid_slices = [np.where((nan_indxs[i][1:] - nan_indxs[i][:-1] - 1) >= self.total_seq_len)[0]
                        for i in range(len(nan_indxs))]
        valid_slices = [np.vstack([nan_indxs[i][valid_slices[i]] + 1, nan_indxs[i][valid_slices[i] + 1] - 1]).T
                        for i in range(len(nan_indxs))]

        # Now, construct an array which contains the valid start indices for the different sequences
        data_indxs = []
        for i in range(len(valid_slices)):
            start_indxs = np.zeros(data.shape[0] - self.total_seq_len + 1, dtype='bool')
            for s, e in valid_slices[i]:
                indxs_i = np.arange(s, e - self.total_seq_len + 2, 1)
                start_indxs[indxs_i] = True
            data_indxs.append(start_indxs)
        self.data_indxs = np.stack(data_indxs, -1)

        # Construct the time array
        assert df_raw[['time']].all(0).all()
        df_stamp = df_raw[['time']].iloc[:, :1]     # [border1:border2]
        df_stamp.columns = ['time']
        df_stamp['time'] = pd.to_datetime(df_stamp.time)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 10)
            data_stamp = df_stamp.drop(['time'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['time'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise ValueError("Pass timeenc as either 0 or 1")

        self.data_x = data      # The full dataset
        self.data_stamp = data_stamp    # The full time dataset
        self.valid_indxs = np.where(self.data_indxs.sum(1) >= self.min_num_nodes)[0]   # Index slices for the data
        self.valid_indxs = self.valid_indxs[::self.data_step]

        # Find the n_closest number of nodes for every node. Use euclidean distance from scaled distances.
        if self.n_closest is not None:
            sub_station_info = self.station_info[self.station_info['name'].isin(self.subset)] if self.subset is not None else self.station_info
            latlon_scaler = MinMaxScaler()
            latlon_scaler.fit(sub_station_info[['lat', 'lon']].values)
            sub_station_info[['slat', 'slon']] = latlon_scaler.transform(sub_station_info[['lat', 'lon']].values)

            connectivity = {}
            for i, row_i in sub_station_info.iterrows():
                dists = np.array(sub_station_info.apply(
                    lambda row: np.sqrt((row['slat'] - row_i.slat) ** 2 + (row['slon'] - row_i.slon) ** 2),
                    axis=1).to_list())
                connectivity[row_i['name']] = sub_station_info.name.iloc[np.argsort(dists)].values
            connectivity = pd.DataFrame(connectivity)
            self.connectivity = connectivity.apply(lambda col: col.map(self._stations), axis=0)
            self.connectivity.columns = [self._stations[st] for st in self.connectivity.columns]

            self.connectivity = [
                self.connectivity.columns.values,
                self.connectivity.values
            ]

        edge_feats = []
        senders = []
        receivers = []
        for rec_i, stat_i in enumerate(self._stations.keys()):
            info_i = self.station_info[self.station_info['name'] == stat_i]
            for send_i, stat_j in enumerate(self._stations.keys()):
                receivers.append(rec_i)
                senders.append(send_i)
                info_j = self.station_info[self.station_info['name'] == stat_j]
                dlat = info_i.lat.iloc[0] - info_j.lat.iloc[0]
                dlon = info_i.lon.iloc[0] - info_j.lon.iloc[0]
                edge_feats.append([dlat, dlon])

        self.graph_struct = {
            'nodes': None,      # [NxSxD]
            'edges': np.array(edge_feats),          # [N2x2]
            'senders': np.array(senders),           # [N2,]
            'receivers': np.array(receivers),       # [N2,]
            'station_names': self._stations.keys(),
        }

    def __getitem__(self, index):
        s_begin = self.valid_indxs[index]
        stations = np.where(self.data_indxs[s_begin, :])[0]

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        seq_x = self.data_x[s_begin:s_end, :, stations]
        seq_y = self.data_x[r_begin:r_end, :, stations]

        if self.subset is not None:
            station_names = np.array(self.subset)[stations]
            stations = np.array([self._stations[s] for s in station_names])
        else:
            station_names = [self._stations_inv[i] for i in stations]

        if self.n_closest is not None:
            def my_func(col):
                return col[np.isin(col, stations)][:min(1 + self.n_closest, len(stations))]

            def my_func2(col, col_name):
                col = np.where(
                    np.stack(
                        [(self.graph_struct['receivers'] == col_name),
                         np.isin(self.graph_struct['senders'], col)]
                    ).all(0)
                )[0]
                return col

            connect = self.connectivity[1][:, np.where(np.isin(self.connectivity[0], stations))[0]]
            connect = np.apply_along_axis(my_func, axis=0, arr=connect)
            keep_edges = np.concatenate([my_func2(connect[:, i], s) for i, s in
                                          enumerate(self.connectivity[0][np.isin(self.connectivity[0], stations)])])

        else:
            keep_edges = np.where(np.stack([np.isin(self.graph_struct['senders'], stations),
                                            np.isin(self.graph_struct['receivers'], stations)]).all(0))[0]

        graph_mapping = dict(zip(stations, np.arange(len(stations))))
        senders = np.vectorize(graph_mapping.get)(self.graph_struct['senders'][keep_edges])
        receivers = np.vectorize(graph_mapping.get)(self.graph_struct['receivers'][keep_edges])
        edge_feats = self.graph_struct['edges'][keep_edges]

        graph_x = {
            'nodes': seq_x.transpose(2, 0, 1),       # [NxSxD]
            'edges': edge_feats,                     # [N2x2]
            'senders': senders,                      # [N2,]
            'receivers': receivers,                  # [N2,]
            'station_names': station_names,
        }
        graph_y = {
            'nodes': seq_y.transpose(2, 0, 1),      # [NxSxD]
            'edges': np.array(edge_feats),          # [N2x2]
            'senders': np.array(senders),           # [N2,]
            'receivers': np.array(receivers),       # [N2,]
            'station_names': station_names,
        }

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return graph_x, graph_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.valid_indxs)

    # Assumes non-graph (i.e. mainly used for the outputs...)
    #  Inputs should be either of shape [nodes, seq_len, feats] or [seq_len, feats]
    def inverse_transform(self, data):
        num_input_feats = data.shape[-1]
        if num_input_feats != len(self.scaler.scale_):
            data = np.concatenate([np.zeros([*data.shape[:-1], len(self.scaler.scale_) - data.shape[-1]]), data], -1)
        data = self.scaler.inverse_transform(data)

        if num_input_feats != len(self.scaler.scale_):
            data = data[..., -num_input_feats:]

        return data


# Custom collate function to graph samples into a batch
def collate_graph(batch):
    graph_x, graph_y, seq_x_mark, seq_y_mark = [[d[i] for d in batch] for i in range(len(batch[0]))]
    sizes_add = np.cumsum([0, *[g['nodes'].shape[0] for g in graph_x][:-1]])
    x = {
        'nodes': np.concatenate([g['nodes'] for g in graph_x], 0),
        'edges': np.concatenate([g['edges'] for g in graph_x], 0),
        'senders': np.concatenate([g['senders'] + start_i for g, start_i in zip(graph_x, sizes_add)]),
        'receivers': np.concatenate([g['receivers'] + start_i for g, start_i in zip(graph_x, sizes_add)]),
        'n_node': np.array([g['nodes'].shape[0] for g in graph_x]),
        'n_edge': np.array([g['edges'].shape[0] for g in graph_x]),
        'graph_mapping': np.stack([sizes_add, np.cumsum([g['nodes'].shape[0] for g in graph_x])], -1),
        'station_names': np.concatenate([g['station_names'] for g in graph_x]),
    }
    y = {
        'nodes': np.concatenate([g['nodes'] for g in graph_y], 0),
        'edges': np.concatenate([g['edges'] for g in graph_y], 0),
        'senders': np.concatenate([g['senders'] + start_i for g, start_i in zip(graph_y, sizes_add)]),
        'receivers': np.concatenate([g['receivers'] + start_i for g, start_i in zip(graph_y, sizes_add)]),
        'n_node': np.array([g['nodes'].shape[0] for g in graph_x]),
        'n_edge': np.array([g['edges'].shape[0] for g in graph_x]),
        'graph_mapping': np.stack([sizes_add, np.cumsum([g['nodes'].shape[0] for g in graph_x])], -1),
        'station_names': np.concatenate([g['station_names'] for g in graph_y]),
    }

    seq_x_mark = np.stack(seq_x_mark, 0)
    seq_y_mark = np.stack(seq_y_mark, 0)

    return x, y, torch.tensor(seq_x_mark), torch.tensor(seq_y_mark)
