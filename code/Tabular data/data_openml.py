import openml
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch.utils.data import Dataset


def simple_lapsed_time(text, lapsed):
    hours, rem = divmod(lapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(text+": {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


def task_dset_ids(task):
    dataset_ids = {
        'binary': [1487,44,1590,42178,1111,31,42733,1494,1017,4134,37],
        'multiclass': [188, 1596, 4541, 40664, 40685, 40687, 40975, 41166, 41169, 42734,],
        'regression':[541, 42726, 42727, 422, 42571, 42705, 42728, 42563, 42724, 42729]
    }

    return dataset_ids[task]

def concat_data(X,y):
    # import ipdb; ipdb.set_trace()
    return pd.concat([pd.DataFrame(X['data']), pd.DataFrame(y['data'][:,0].tolist(),columns=['target'])], axis=1)


def data_split(X,y,nan_mask,indices):
    x_d = {
        'data': X.values[indices],
        'mask': nan_mask.values[indices]
    }
    
    if x_d['data'].shape != x_d['mask'].shape:
        raise'Shape of data not same as that of nan mask!'
        
    y_d = {
        'data': y[indices].reshape(-1, 1)
    } 
    return x_d, y_d


def data_prep_openml(ds_id, seed, task, datasplit=[.8, .1, .1]):
    
    np.random.seed(seed) 
    dataset = openml.datasets.get_dataset(ds_id)
    
    X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)
    if ds_id == 42178:
        categorical_indicator = [True, False, True,True,False,True,True,True,True,True,True,True,True,True,True,True,True,False, False]
        tmp = [x if (x != ' ') else '0' for x in X['TotalCharges'].tolist()]
        X['TotalCharges'] = [float(i) for i in tmp ]
        y = y[X.TotalCharges != 0]
        X = X[X.TotalCharges != 0]
        X.reset_index(drop=True, inplace=True)
        print(y.shape, X.shape)
    if ds_id in [42728,42705,42729,42571]:
        # import ipdb; ipdb.set_trace()
        X, y = X[:50000], y[:50000]
        X.reset_index(drop=True, inplace=True)
    categorical_columns = X.columns[list(np.where(np.array(categorical_indicator)==True)[0])].tolist()
    cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))

    cat_idxs = list(np.where(np.array(categorical_indicator)==True)[0])
    con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))

    for col in categorical_columns:
        X[col] = X[col].astype("object")

    X["Set"] = np.random.choice(["train", "valid", "test"], p = datasplit, size=(X.shape[0],))

    train_indices = X[X.Set=="train"].index
    valid_indices = X[X.Set=="valid"].index
    test_indices = X[X.Set=="test"].index

    X = X.drop(columns=['Set'])
    temp = X.fillna("MissingValue")
    nan_mask = temp.ne("MissingValue").astype(int)
    
    cat_dims = []
    for col in categorical_columns:
    #     X[col] = X[col].cat.add_categories("MissingValue")
        X[col] = X[col].fillna("MissingValue")
        l_enc = LabelEncoder() 
        X[col] = l_enc.fit_transform(X[col].values)
        cat_dims.append(len(l_enc.classes_))
    for col in cont_columns:
    #     X[col].fillna("MissingValue",inplace=True)
        X.fillna(X.loc[train_indices, col].mean(), inplace=True)
    y = y.values
    if task != 'regression':
        l_enc = LabelEncoder() 
        y = l_enc.fit_transform(y)
    X_train, y_train = data_split(X,y,nan_mask,train_indices)
    X_valid, y_valid = data_split(X,y,nan_mask,test_indices)
    X_test, y_test = data_split(X,y,nan_mask,valid_indices)

    train_mean, train_std = np.array(X_train['data'][:,con_idxs],dtype=np.float32).mean(0), np.array(X_train['data'][:,con_idxs],dtype=np.float32).std(0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)
    # import ipdb; ipdb.set_trace()
    return cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std




class DataSetCatCon(Dataset):
    def __init__(self, X, Y, cat_cols,task='clf',labeled_ratio =0.25, continuous_mean_std=None,method: str='semisupervised'):
        self.train = True
        cat_cols = list(cat_cols)
        X_mask =  X['mask'].copy()
        self.data =  X['data'].copy()
        X = X['data'].copy()
        self.y = Y['data']
        idx = np.arange(len(self.y))
        self.labeled_idx = idx
        self.unlabeled_idx = idx
        self.idx = idx
        if (labeled_ratio > 0):
            idx = np.random.permutation(len(self.y))
        self.idx = idx
        ns = labeled_ratio * len(self.idx)
        ns = int(ns)
        labeled_idx = self.idx[:ns]
        unlabeled_idx = self.idx[ns:]
        self.labeled_idx = labeled_idx
        self.unlabeled_idx = unlabeled_idx
        self._pseudo_labels = list()
        self._pseudo_labels_weights = list()
        self.method = method.lower()
                
        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        self.X1 = X[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2 = X[:,con_cols].copy().astype(np.float32) #numerical columns
        self.X1_mask = X_mask[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2_mask = X_mask[:,con_cols].copy().astype(np.int64) #numerical columns
        self.cls = np.zeros_like(self.y,dtype=int)
        self.cls_mask = np.ones_like(self.y,dtype=int)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

            
    def __len__(self):
        if (self.method == 'pseudolabeling'):
            return len(self.idx)
        return len(self.labeled_idx)

    def get_pseudo_labels(self):
        return self._pseudo_labels

    def set_pseudo_labels(self, pseudo_labels):
        self._pseudo_labels = pseudo_labels

    def set_pseudo_labels_weights(self, pseudo_labels_weights):
        self._pseudo_labels_weights = pseudo_labels_weights

    def _pseudolabeling__getitem__(self, idx):
        idx = self.idx[idx]
        x_categ, x_cont, target = np.concatenate((self.cls[idx], self.X1[idx])), self.X2[idx], self.y[idx]
        labeled_mask = np.array([False], dtype=np.bool)
        cat_mask, con_mask = np.concatenate((self.cls_mask[idx], self.X1_mask[idx])), self.X2_mask[idx]
        if idx in self.labeled_idx:
            labeled_mask[0] = True
        idx = np.asarray([idx])
        return  x_categ, x_cont, target,cat_mask, con_mask,labeled_mask, idx, 

    def _normal__getitem__(self, idx):      
        idx = self.labeled_idx[idx]
        uidx = np.random.randint(0, len(self.unlabeled_idx))
        uidx = self.unlabeled_idx[uidx]
        x_categ, x_cont = np.concatenate((self.cls[idx], self.X1[idx])), self.X2[idx]
        u_categ, u_cont = np.concatenate((self.cls[uidx], self.X1[uidx])), self.X2[uidx]
        target, u_target = self.y[idx], self.y[uidx]
        cat_mask, con_mask = np.concatenate((self.cls_mask[idx], self.X1_mask[idx])), self.X2_mask[idx]
        u_cat_mask, u_con_mask = np.concatenate((self.cls_mask[uidx], self.X1_mask[uidx])), self.X2_mask[uidx]
        if len(self._pseudo_labels):
            utarget = self._pseudo_labels[uidx]
            uweight = self._pseudo_labels_weights[uidx]
            return x_categ, x_cont,u_categ, u_cont, target, u_target, cat_mask, con_mask, u_cat_mask, u_con_mask, utarget, uweight
        return x_categ, x_cont,u_categ, u_cont, target, u_target, cat_mask, con_mask, u_cat_mask, u_con_mask


    def _test_getitem__(self, idx):
        idx = self.labeled_idx[idx]
        x_categ, x_cont, target = np.concatenate((self.cls[idx], self.X1[idx])), self.X2[idx], self.y[idx]
        cat_mask, con_mask = np.concatenate((self.cls_mask[idx], self.X1_mask[idx])), self.X2_mask[idx]
        return x_categ, x_cont, target, cat_mask, con_mask



    def __getitem__(self, idx):
        if self.method == 'pseudolabeling' and self.train:
            return self._pseudolabeling__getitem__(idx)
        if self.method == 'semisupervised' and self.train:
            return self._normal__getitem__(idx)
        else:
            return self._test_getitem__(idx)


        
    """        
            
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous
        return np.concatenate((self.cls[idx], self.X1[idx])), self.X2[idx],self.y[idx], np.concatenate((self.cls_mask[idx], self.X1_mask[idx])), self.X2_mask[idx]

"""