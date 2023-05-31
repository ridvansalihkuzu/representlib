import torch
import pandas as pd
import numpy as np
import scipy
from scipy.spatial import Delaunay
from scipy.interpolate import interp1d


class Data:
    def __init__(self, **kwargs):
        self.keys = list(kwargs.keys())
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to(self, device):
        for k in self.keys:
            setattr(self, k,
                    getattr(self, k).to(device)
                    )
        return self

    @staticmethod
    def collate_fn(list_x):
        new_x = dict()
        for k in list_x[0].keys():
            if k != 'edge_index' and k != 'id' and k != 'time':
                new_x[k] = torch.cat([torch.from_numpy(_[k]).to(torch.float32) for _ in list_x], 0)
        n = 0
        batch = []
        edge_index = []
        for k, _ in enumerate(list_x):
            batch.append(k * torch.ones(_['x'].shape[0]).to(torch.long))
            edge_index.append(n + torch.from_numpy(_['edge_index']).to(torch.long))
            n += _['x'].shape[0]
        new_x['batch'] = torch.cat(batch)
        new_x['edge_index'] = torch.cat(edge_index, -1)
        new_x['time'] = torch.from_numpy(list_x[0]['time']).to(torch.float32)
        data = Data(**new_x)

        return data, [str(_['id']) for _ in list_x]

    @staticmethod
    def ASCI2PID(x):
        print(x.shape)
        print(x)
        y = np.array([''.join([chr(int(_)) for _ in p]) for p in x])
        return y


class DataReader(torch.utils.data.Dataset):
    def __init__(self, footprint_array, point_id, data_array, time, coord=None, perturb_freq=None):
        self.time = time
        self.footprint_array = footprint_array
        self.footprint_id = np.unique(footprint_array)
        self.point_id = point_id
        self.data_array = data_array
        self.coord = coord
        self.perturb_freq = perturb_freq

    def __len__(self):
        return self.footprint_id.shape[0]

    def __getitem__(self, index):
        # get footprint_id
        fp_id = self.footprint_id[index]
        # get PSs with given fp_id
        where = self.footprint_array == fp_id
        data = self.data_array[where]
        coord = self.coord[where]
        point_id = self.point_id[where]
        centroid = coord.mean(0, keepdims=True)

        # normalize coordinates
        coord = (coord - coord.min(0)[np.newaxis]) / ((coord.max(0) - coord.min(0) + 1E-8)[np.newaxis])

        # normalization coefficient
        coeff = data.std(-1).mean().reshape((1, 1))
        data = data / coeff

        # flag to identify replicas
        flag = np.zeros(coord.shape[0])
        if self.perturb_freq is not None:
            show = False
            if self.perturb_freq[0] > data.shape[0]:

                # k duplicates n = (k+1)m+d
                k = (self.perturb_freq[0] // coord.shape[0]) - 1
                d = np.random.randint(0, data.shape[0], (self.perturb_freq[0] % coord.shape[0],))
                if show:
                    fig, ax = plt.subplots(3, 1)
                    figtime = pd.to_datetime('20150216') + pd.to_timedelta(self.time, 'D')
                if k > 0 and d.shape[0] > 0:
                    duplicate = np.concatenate([data] * k + [data[i:i + 1] for i in d], 0)
                    dup_pid = np.concatenate([point_id] * k + [point_id[i:i + 1] for i in d], 0)
                elif k > 0:
                    duplicate = np.concatenate([data] * k, 0)
                    dup_pid = np.concatenate([point_id] * k, 0)
                else:
                    duplicate = np.concatenate([data[i:i + 1] for i in d], 0)
                    dup_pid = np.concatenate([point_id[i:i + 1] for i in d], 0)

                if show:
                    ax[0].plot(figtime, duplicate[0], color='k', marker='.', label='original')
                    ax[0].set_ylabel('mm')

                hatx = np.fft.rfft(duplicate, n=data.shape[-1], axis=-1)
                freq = np.fft.rfftfreq(data.shape[-1], d=(self.time[1] - self.time[0]))
                ifreq = np.where((freq <= self.perturb_freq[1]) & (freq > self.perturb_freq[2]))[0]
                if show:
                    ax[0].plot(figtime,
                               np.fft.irfft(np.where((freq <= self.perturb_freq[1]) & (freq > self.perturb_freq[2]), 0.,
                                                     hatx[0]), n=duplicate.shape[-1]), color='lightblue',
                               label='background')
                    ax[0].legend(loc='upper right')
                    ax[1].plot(freq[1:], np.abs(hatx[0, 1:]) ** 2, color='darkblue', marker='.')
                    ax[1].set_ylabel('Spectral Density\n' + r'$\left[mm^2 yr\right]$')
                    ax[1].set_xlabel(r'Frequency $\left[{yr}^{-1}\right]$')
                    ax[1].axvspan(freq[ifreq].min(), freq[ifreq].max(), color='lightcoral', alpha=0.5, label='changed')
                    ax[1].axvspan(freq.min(), freq[ifreq].min(), color='lightblue', alpha=0.5, label='unchanged')
                    ax[1].legend(loc='upper right')
                    ax[1].set_yscale('log')
                    ax[1].set_xscale('log')
                    ax[1].set_ylim([1, (np.abs(hatx[0]) ** 2).max()])
                hatx[:, ifreq] *= np.exp(2j * np.pi * np.random.uniform(0.25, 0.75, (hatx.shape[0], ifreq.shape[0])))
                duplicate = np.fft.irfft(hatx, n=data.shape[-1], axis=-1)
                if show:
                    ax[2].plot(figtime, duplicate[0], color='black', marker='.', label='replica')
                    ax[2].plot(figtime,
                               np.fft.irfft(np.where((freq <= self.perturb_freq[1]) & (freq > self.perturb_freq[2]), 0.,
                                                     hatx[0]), n=duplicate.shape[-1]), color='lightblue',
                               label='background')

                    ax[2].set_ylabel('mm')
                    ax[2].legend(loc='upper right')
                    plt.subplots_adjust(hspace=0.4, top=0.9)
                    # plt.show()

                # radius
                r = np.sqrt(((coord[np.newaxis] - coord[:, np.newaxis]) ** 2).sum(-1))[
                    np.triu_indices(coord.shape[0], 1)]
                r = 1 if r.shape[0] < 1 else r.min()
                if k > 0 and d.shape[0] > 0:
                    duplicate_coord = np.concatenate([coord] * k + [coord[i:i + 1] for i in d], 0)
                elif k > 0:
                    duplicate_coord = np.concatenate([coord] * k, 0)
                else:
                    duplicate_coord = np.concatenate([coord[i:i + 1] for i in d], 0)
                z = np.random.uniform(-1, 1, duplicate_coord.shape)
                z = r * np.random.uniform(1 / 2., 2 / 3., (z.shape[0], 1)) * (
                        z / np.sqrt((z ** 2).sum(-1)[:, np.newaxis]))
                duplicate_coord = duplicate_coord + z

                if show:
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    cmap = matplotlib.cm.nipy_spectral

                    scatter_o = ax.scatter(*coord.T, edgecolor='k', s=50, c=np.arange(1, coord.shape[0] + 1),
                                           cmap='nipy_spectral', vmin=1, vmax=coord.shape[0], linewidths=1)

                    for c in duplicate_coord:
                        i = np.argmin(((c[np.newaxis] - coord) ** 2).sum(-1))
                        ax.scatter(*c, edgecolor='red', c=i + 1, cmap=cmap, vmin=1, vmax=coord.shape[0], s=50,
                                   linewidths=1, marker='^')

                    l1 = ax.legend(*scatter_o.legend_elements(), title='PSs', loc='upper left')
                    ax.add_artist(l1)
                    legend_elements = [
                        matplotlib.lines.Line2D([], [], color='k', marker='o', label='original', markersize=10,
                                                linestyle='none'),
                        matplotlib.lines.Line2D([], [], color='r', marker='^', label='replica', markersize=10,
                                                linestyle='none'),
                    ]
                    ax.legend(handles=legend_elements, loc='lower right')
                    plt.show()

                data = np.concatenate([data, duplicate], 0)
                coord = np.concatenate([coord, duplicate_coord], 0)
                coord = (coord - coord.min(0)[np.newaxis]) / ((coord.max(0) - coord.min(0) + 1E-8)[np.newaxis])

                flag = np.concatenate([flag, np.ones(duplicate_coord.shape[0])])
                point_id = np.concatenate([point_id, dup_pid])

        # edge_index creation
        if coord.shape[0] < 5:
            edge_index = []
            for j in range(coord.shape[0] - 1):
                for i in range(j + 1, coord.shape[0]):
                    edge_index.append([j, i])
            edge_index = np.array(edge_index).T
        else:
            try:
                edge_index = self.simp2edge(Delaunay(coord, qhull_options='Qbb Qc Qz Q12'))
            except scipy.spatial.qhull.QhullError:
                try:
                    edge_index = self.simp2edge(Delaunay(coord, qhull_options='QJ'))
                except:
                    edge_index = []
                    for j in range(coord.shape[0] - 1):
                        for i in range(j + 1, coord.shape[0]):
                            edge_index.append([j, i])
                    edge_index = np.array(edge_index).T
        if len(edge_index) == 0:
            edge_index = np.arange(coord.shape[0]).reshape((1, -1)).repeat(2, 0)
        else:
            edge_index = np.concatenate(
                [edge_index, edge_index[::-1], np.arange(coord.shape[0]).reshape((1, -1)).repeat(2, 0)], -1)

        point_id = np.array([[ord(_) for _ in p] for p in point_id]).astype('int')
        return {'x': data.astype(np.float32), 'edge_index': edge_index.astype('int'), 'time': self.time,
                'pos': coord, 'id': fp_id, 'coeff': coeff, 'flag': flag, 'pid': point_id, 'centroid': centroid}

    # create edges from simplex
    def simp2edge(self, tri):
        def less_first(a, b):
            return [a, b] if a < b else [b, a]

        edge_index = []
        for tr in tri.simplices:
            for e0, e1 in [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]:
                edge_index.append(less_first(tr[e0], tr[e1]))

        return np.unique(edge_index, axis=0).T


class BuildDataLoader:
    def __init__(self,
                 data_file,
                 interp_window=31,
                 batch_size=32,
                 rank=0,
                 world_size=1,
                 seed=0,
                 n_build=None,
                 ps_range=None,
                 perturb_freq=None,
                 ):

        np.random.seed(seed)
        torch.manual_seed(seed)

        df = pd.read_csv(data_file)

        # filter dataset by number of buildings or PS range (if provided)
        df = self.subsample(df, world_size=world_size, rank=rank, n_build=n_build, ps_range=ps_range)

        # read footprint_id, ps_id, time_series, time, coordinates from dataframe
        fp_id, p_id, ts, time, coord = BuildDataLoader._read_dataframe(df, window=interp_window)

        # get dataset
        dataset = DataReader(fp_id, p_id, ts, time, coord=coord, perturb_freq=perturb_freq)

        # get data loader
        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                       shuffle=False, num_workers=1, pin_memory=True,
                                                       collate_fn=Data.collate_fn)

    def subsample(self, df_in, rank, world_size, n_build=None, ps_range=None):

        # discard buildings with number of PSs outside the range, if provided
        if ps_range is not None:
            df = []
            fp_id = []
            for a, b in df_in.groupby('footprint_id'):
                if ps_range[0] <= len(b) < ps_range[1]:
                    fp_id.append(a)
                    df.append(b)
            df = pd.concat(df, axis=0)
            fp_id = np.array(fp_id)
        else:
            fp_id = np.unique(df_in['footprint_id'].values)
            df = df_in

        # select a limited number of buildings, if provided
        if n_build is not None:
            fp_id = fp_id[:n_build]

        # select buildings for a given rank
        np.random.shuffle(fp_id)
        j = np.linspace(0, len(fp_id), world_size + 1).astype('int')
        fp_id = fp_id[j[rank]:j[rank + 1]]

        # return filtered dataframe
        return df[df['footprint_id'].isin(fp_id)]

    # interpolate series over regular grid
    @staticmethod
    def interp(x, t, window):
        w = np.where(np.isfinite(x).all(0))[0]
        func = interp1d(t[w], x[:, w])
        new_t = np.linspace(t[0], t[-1], window * (t.shape[0] // window))
        y = func(new_t)
        return y, new_t

    def get_data_loader(self):
        return self.data_loader

    # read footprint_id, ps_id, time_series, time, coordinate
    @staticmethod
    def _read_dataframe(df, window=31, smoothed=None, synthetic=False):

        fp_id = np.array(df['footprint_id'])
        coord = np.array(df[['easting', 'northing', 'height']].values)

        # get dates
        times = pd.to_datetime([pd.to_datetime(_) for _ in df.columns if _.startswith('20')])
        # convert to float time
        times = (times - times[0]).days / 365.25
        # get time series
        x = df.filter(regex="20*").values
        # interpolate over regular grid
        x, times = BuildDataLoader.interp(x, times, window)
        # zero mean
        x = x - (x.mean(-1)[:, np.newaxis])
        # PS id
        p_id = df['pid'].values
        return fp_id, p_id, x, times, coord


if __name__ == '__main__':
   
    loader = BuildDataLoader('DATA/database_test.csv').get_data_loader()

   