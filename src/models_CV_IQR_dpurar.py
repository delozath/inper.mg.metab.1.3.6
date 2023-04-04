
import pdb
import dload

import numpy  as np
import pandas as pd


from sklearn.svm import SVR

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import BaseCrossValidator

from sklearn.model_selection import cross_validate
from sklearn import metrics

from matplotlib import pyplot as plt

class Quantile_CV(BaseCrossValidator):
    def __init__(self, n_splits=2, n_perc=5, test_porc=.3):
        self.n_perc    = n_perc + 1
        self.n_splits  = n_splits
        self.test_porc = test_porc
    #
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
    #
    def __get_bins__(self, y):
        eps = 1E-3
        partition = np.linspace(0+eps, 1+eps, self.n_perc)
        bins      = []
        for p in partition[1:-1]:
            bins.append(np.quantile(y, p))
        #
        bins    = [min(y)-eps] + bins + [max(y)+eps]
        labels  = np.arange(len(bins))[1:]
        return bins, labels
    #
    def split(self, X, y, groups=None):
        mask = self._iter_test_indices(X, y)
        for m in mask:
            yield m
    #
    def _iter_test_indices(self, X, y, groups=None):
        bins, labels = self.__get_bins__(y)
        #
        qt       = pd.DataFrame(pd.cut(y, 
                                  bins=bins, 
                                  labels=labels)
                                )
        qt = qt.to_numpy().flatten()
        #
        for _ in range(self.n_splits):
            index = np.arange(len(qt))
            mask  = []
            for blk in np.unique(qt):
                idx = index[qt==blk].copy()
                np.random.shuffle(idx)
                #
                n_test = int(len(idx)*self.test_porc)
                mask.append(idx[-n_test:])
            #
            _mask = np.concatenate(mask)
            mask  = np.zeros_like(index).astype('bool')
            mask[_mask] = True
            #
            print("This is printed for debugging purposes")
            print('train: ', index[~mask])
            print('test: ' , index[mask], '\n\n')
            #
            yield mask
#
from sklearn import metrics
class non_linear_models(dload.load_data):
    def __init__(self, block, SRC_PATH):
        super(non_linear_models, self).__init__(block, SRC_PATH)
        self.tbnames = self.loader.get_table_attr_names()
    #
    def analysis_mtDNA(self):
        tg = 'mtDNA'
        svr_cfg = self.loader.cfg_params['global']['model']['svm-rbf'][tg]
        #
        ft = svr_cfg['features']
        df = self.loader.tb_dt_data[ft+[tg]].dropna()
        
        q1, q3 = df[tg].quantile(.25), df[tg].quantile(.75)
        iqr    = q3 - q1
        low    = q1 - (1.5*iqr)
        high   = q3 + (1.5*iqr)
        #
        mask_in = (df[tg] > low) * (df[tg] < high)
        #
        df = df.loc[mask_in, ft+[tg]]
        df = (df + df.min())/(df.max() + df.min())
        #
        model  = SVR(**svr_cfg['hyperparams'], verbose=2)
        Metrics = []
        CV     = Quantile_CV(n_splits=10)
        for spt in CV.split(df[ft], df[tg]):
            X_Train = df[ft][~spt]
            y_Train = df[tg][~spt]
            #
            X_Test = df[ft][spt]
            y_Test = df[tg][spt]
            #
            model.fit(X_Train, y_Train)
            #
            y_est  = model.predict(X_Test)
            metric = y_Test.values - y_est
            metric = metric @ metric
            Metrics.append(metric)
            print(metrics.mean_absolute_error(y_Test.values, y_est))
            print(metrics.mean_squared_error(y_Test.values, y_est))
            print(metrics.median_absolute_error(y_Test.values, y_est))
            print(metrics.mean_absolute_percentage_error(y_Test.values, y_est))
            #print(metric @ metric)
            #plt.plot(y_Test[y_Test<.75], y_est[y_Test<.75], 'o', color='blue'); plt.show()
            plt.plot(y_Test, y_est, 'o', color='blue'); plt.show()
            #
            #metric = y_Test[y_Test<.75].values - y_est[y_Test<.75]
            #metric = metric @ metric
            #print(metric)
        #scores = cross_validate(model, df[ft], df[tg], cv=CV, scoring='r2', return_estimator=True)
        #
        for m in Metrics:
            print(f"{m:7.3f}")
        pdb.set_trace()
    #
    def get_vars_analyze(self, experiment):
        self.query_vars(experiment)
        self.fts_disc = self.query_to_list("vtype=='feature' & distribution=='discrete'")
        self.fts_cont = self.query_to_list("vtype=='feature' & distribution=='continuous'")
        self.tgs_cont = self.query_to_list("vtype=='target'")
    #
    def svm_regression(self, experiment):
        self.get_vars_analyze(experiment)
        tgs_cont = np.array(self.tgs_cont)
        mask     = np.identity(len(self.tgs_cont)).astype('bool')
        params = {}
        for mk in mask:
            tg = list(tgs_cont[mk])
            df = self.loader.tb_dt_data[self.fts_disc + self.fts_cont + tg ].dropna()
            #
            transform = ColumnTransformer(
              [
               ('maxmin_scaler', MinMaxScaler (), self.fts_cont),
               ('onehot'       , OneHotEncoder(), self.fts_disc),
               ('target'       , MinMaxScaler (), pd.Index(tg))
              ]
            )
            X = transform.fit_transform(df)
            svr_cfg = self.loader.cfg_params['global']['model']['svm-rbf'][tg[0]]       
            
            model  = SVR(**svr_cfg['hyperparams'])
            CV     = Quantile_CV(n_splits=10)
            scores = cross_validate(model, X[:, 1:], X[:, 0], cv=CV, scoring='r2', return_estimator=True)
            #
            pdb.set_trace()
    #
    def svm_hyperparams(self, experiment):
        self.get_vars_analyze(experiment)
        tgs_cont = np.array(self.tgs_cont)
        mask     = np.identity(len(self.tgs_cont)).astype('bool')
        params = {}
        for mk in mask:
            tg = list(tgs_cont[mk])
            df = self.loader.tb_dt_data[self.fts_disc + self.fts_cont + tg ].dropna()
            #
            transform = ColumnTransformer(
              [
               ('maxmin_scaler', MinMaxScaler (), self.fts_cont),
               ('onehot'       , OneHotEncoder(), self.fts_disc),
               ('target'       , MinMaxScaler (), pd.Index(tg))
              ]
            )
            X = transform.fit_transform(df)
            #
            parameters = {'kernel':['rbf'], 
                          'C':np.linspace(1E-2, 2, 100), 
                          'gamma':np.linspace(1E-2, 2, 100)}
            model        = SVR()
            CV         = Quantile_CV(n_splits=8)
            clf        = RandomizedSearchCV(model, parameters, n_iter=10000, cv=CV, verbose=2, n_jobs=-1)
            #
            search = clf.fit(X[:, 1:], X[:, 0])
            params[tg[0]] = search.best_params_
        #
        pdb.set_trace()