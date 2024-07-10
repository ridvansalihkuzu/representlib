import numpy as np
import geopandas as gpd
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os


class PSBuilding:
    # object defining buildings
    # Usage:
    # obj = PSBuilding(filename)
    # obj['series'] returns the series of displacements as a pandas Dataframe with index pid and columns dates
    # obj['trend'] returns the trend of displacements as a pandas Dataframe with index pid and columns dates
    # obj['season'] returns the seasonality of displacements as a pandas Dataframe with index pid and columns dates
    # obj[k] returns a pandas Series with index pid and column k (it must exist in geojson file)
    
    def __init__(self, filename):
        self.filename = filename
        self._setup()       
    
    def _setup(self, x=None):
        self.tseries, self.trend, self.season = self._tseries(x=x)  

    def _tseries(self, x=None):
        if x is None:
            #open file and rearrange displacements to get a dataframe with PSs as rows and Dates as columns
            df = gpd.read_file(self.filename)

            x = pd.DataFrame([[pd.to_datetime(k), xi.values[0][k], df['pid'].iloc[i]] for (i, xi) in df[['displacements']].iterrows() for k in sorted(list(xi.values[0].keys()))],
                            columns=['date', 'd', 'pid'])
            
            x = pd.concat([xi.drop('date', axis=1).set_index('pid', drop=True).rename(columns={'d': str(i)}) for i, xi in x.groupby('date')], axis=1)
        
        #trend
        trend = self._trend(x.values, pd.to_datetime(x.columns))
        trend = pd.DataFrame(trend, index=x.index, columns=x.columns)
        x = x.where(~x.isna(), trend)
    

        #season
        sea = (x - trend).T
        sea.index = pd.to_datetime(sea.index)
        sea = sea.reindex(pd.date_range(sea.index[0], sea.index[-1], freq='D')).interpolate()
        sea = sea.rolling('15D').mean()
        sea['day'] = np.clip(sea.index.dayofyear , 0, 365)
        sea = sea.groupby('day').transform('mean').reindex(x.T.index).T
        sea = pd.DataFrame(sea, index=x.index, columns=x.columns)

        return x, trend, sea  
    
    @staticmethod
    def _trend(series, dates, deg=2):
        w = np.where(np.all(np.isfinite(series), 0))[0]
   
        m = np.mean(series[:,w],-1,keepdims=True)
        x = series - m #series (samples, time)

        t = np.array((dates - dates[0]).days) #(time,)
        t = np.stack([t**i for i in range(1, deg+1)], 0) #(deg, time')
        mt = np.mean(t[:, w], -1, keepdims=True) 
        
        t0 = t[:, w] - mt
        x = x[:, w]

        vart = np.einsum('d t, D t -> d D', t0, t0)
        tx = np.einsum('d t, p t -> p d', t0, x)
        coeff = np.einsum('d D, p D -> p d', np.linalg.inv(vart), tx)
        
        trend = np.einsum('d t, p d -> p t', t - mt, coeff) + m
        
        return trend
        
    def __getitem__(self, key):
        if key == 'series':
            return self.tseries
        elif key == 'trend':
            return self.trend
        elif key == 'season':
            return self.season
        else:
            return gpd.read_file(self.filename)[['pid',key]].set_index('pid', drop=True)

    def inject(self, frac=1., anm_type=None, epoch=0.5, magnitude=3.):
        # inject anomalies in the series and return dataframe of labels 
        # params:
        # frac (fraction of PSs to modify, if > 0 at least 1 PS is modified)
        # anm_type (possible values are None, "trend", "step", "noise", "freq", "season")
        # epoch (float < 1, 0 -> series start, 1 -> series end)
        # magnitude (magnitude of anomaly)
        # return:
        # dataseries of anomaly_type with index pid
        
        if frac==0 or (anm_type is None):
            return pd.DataFrame(np.zeros(self.tseries.shape[0]), index=self.tseries.index, columns=['anomaly_type']) 

        #index of PSs to change
        mask = np.zeros(self.tseries.shape).astype('bool')
        mask[np.random.permutation(mask.shape[0])[:max(1, int(len(self.tseries)*frac))]] = True
        
        #series, trend and season arrays (ps, time)
        x = self.tseries.values
        trend = self.trend.values
        season = self.season.values

        #index of epoch
        iepoch = int((x.shape[-1]-1) * epoch)


        if anm_type=='trend':
            newtrend = magnitude * np.abs(trend[:,-1]-trend[:,iepoch]) * np.random.choice([-1,1])
            newtrend = newtrend[:,np.newaxis] * np.concatenate([np.zeros((1,iepoch)), np.linspace(0,1,x.shape[1]-iepoch)[np.newaxis]], -1)
            newtrend = np.where(np.arange(x.shape[-1])[np.newaxis] >= iepoch, newtrend, trend)

            x = x - trend + newtrend
            y = np.where(np.all(mask,-1), 1, 0)

        elif anm_type=='step':
            newtrend = magnitude * np.abs(trend[:,-1]-trend[:,iepoch]) * np.random.choice([-1,1])
            newtrend = newtrend[:,np.newaxis] * np.concatenate([np.zeros((1,iepoch)), np.ones((1,x.shape[1]-iepoch))], -1)
            newtrend = np.where(np.arange(x.shape[-1])[np.newaxis] >= iepoch, newtrend, trend)
            
            x = x - trend + newtrend
            y = np.where(np.all(mask,-1), 2, 0)
        
        elif anm_type=='noise':
            eps = x - trend - season
            eps = np.std(eps, axis=-1, keepdims=True)
            eps = magnitude * eps * np.random.uniform(-1,1, x.shape)
            eps = np.where(np.arange(x.shape[-1])[np.newaxis] >= iepoch, magnitude * eps, np.zeros(x.shape))

            x = x + eps  
            y = np.where(np.all(mask,-1), 3, 0)

        elif anm_type=='activity':
            x = pd.DataFrame(x - trend - season, columns=self.tseries.columns).T
            dates = pd.to_datetime(x.index)
            x.index = dates
            x =  x.reindex(pd.date_range(x.index[0], x.index[-1], freq='D')).interpolate()
            A = magnitude * x.abs().max(axis=0).values
            A = A[np.newaxis,:] * (np.sin(((2*np.pi)/60)*(x.index-x.index[iepoch]).days)[:, np.newaxis] > 0.75).astype('int')
            x = x.where(x.index[:,np.newaxis].repeat(x.shape[1],1) < pd.to_datetime(self.tseries.columns[iepoch]), x + A)
            x = x.reindex(dates).T.values + trend + season

            y = np.where(np.all(mask,-1), 4, 0)
        
        
        elif anm_type=='season':
            newseason = np.where(np.arange(x.shape[-1])[np.newaxis] >= iepoch, (1+magnitude) * season, season)
            x = x - season + newseason
    
            y = np.where(np.all(mask,-1), 5, 0)

        
        x = self.tseries.where(~mask, pd.DataFrame(x,index=self.tseries.index, columns=self.tseries.columns))
        self._setup(x) #recompute trend and season after anomaly generation

        return pd.DataFrame(y, index=x.index, columns=['anomaly_type']) 
        

class PSDataset(Dataset):
    def __init__(self, columns=[], datadir=None, files=None, train_frac=0.8, shuffle=True): 
        super().__init__() 

        if (datadir is None and files is None) or (datadir is not None and files is not None):
            raise Exception('Only one of datadir or files must be provided')
        
        self.columns = columns 
        if 'displacements' not in columns:
            self.columns += ['displacements']
        if 'b_id' not in columns:
            self.columns += ['b_id']
 

        if files is None:
            self.datadir = datadir
            self.train_frac = train_frac
            self._files = np.array(self._list(self.datadir))
            np.random.shuffle(self._files)
            self._files, self.test_files = np.split(self._files, [int(len(self._files)*train_frac)])
        else:
            self._files = files
        return None

    def __len__(self):
        return len(self._files)

    def __getitem__(self, id):
        
        b_obj = PSBuilding(self._files[id])

        return b_obj 
        
    def _list(self, _dir, out=None):
        # recursive search of all geojson files
        # return list of paths
        if out is None:
            out = []
        for d in os.listdir(_dir):
            cd = os.path.join(_dir, d)
            if os.path.isdir(cd):
                out += [os.path.join(cd, f) for f in os.listdir(cd) if f.endswith('.geojson')]
                self._list(cd, out)
        return out
    
    def test_dataset(self):
        return PSDataset(columns=self.columns, files=self.test_files)
    
    @staticmethod
    def _collatefunc(items):
        raise Exception('Not implemented yet')
    
        
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    np.random.seed(0)

    #dataset dir
    DATADIR = os.path.join(os.getcwd(), 'DATA')

    #train and test datasets
    train_dataset = PSDataset(datadir=DATADIR, train_frac=1)
    test_dataset = train_dataset.test_dataset() #empty if train_frac = 1

    for k in range(1,6):
        #a sample
        x = train_dataset[2]


        fig, ax = plt.subplots(2,1)

        #time-series
        ax[0].plot(x['series'].iloc[0],linestyle='-',marker='*')
        ax[0].set_xticks([])
        ax[0].set_title('original')

        if k == 1:
            anm_type = 'trend'
            magnitude = 5
        elif k == 2:
            anm_type = 'step'
            magnitude = 3
        elif k == 3:
            anm_type = 'noise'
            magnitude = 2
        elif k == 4:
            anm_type = 'activity'
            magnitude = 1
        elif k == 5:
            anm_type = 'season'
            magnitude = 5

        fig.suptitle('Anomaly Type: ' + anm_type)

        #inject anomalies
        label = x.inject(anm_type=anm_type,frac=1, magnitude=magnitude)
        
        ax[1].plot(x['series'].iloc[0],linestyle='-',marker='*')
        ax[1].set_xticks([])
        ax[1].set_title('synthetic')
        
        print(label)
    
    plt.show()