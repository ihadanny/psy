import pandas
import scipy
import numpy
import sklearn
import sys
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import torch # For building the networks
import torchtuples as tt # Some useful functions

from pycox.datasets import metabric
from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv

np.random.seed(1234)
_ = torch.manual_seed(123)

# load data
data = "C:/Users/Amir/Desktop/acsisnasis/rdata/nov/acs2020.csv"
dataset = pandas.read_csv(data)

#preparing data
dataset2 =pandas.concat([dataset.iloc[:,0:59],dataset.iloc[:,62]],axis=1)
i=0
#for col in dataset2.columns:

 #   if i<11:
  #      dataset2[col] = dataset2[col].astype(float)
   # elif i == 59:
    #    dataset2[col] = dataset2[col].astype(bool)
   # else:
     #   dataset2[col] = dataset2[col].astype('category')
    #i=i+1

#splitting
df_test = dataset2.sample(frac=0.3)
df_train = dataset2.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)

#imputing
i=0
for col in dataset2.columns:
    if i<11:
        df_train[col].fillna(df_train[col].mean(), inplace=True)
        df_test[col].fillna(df_train[col].mean(), inplace=True)
        df_val[col].fillna(df_train[col].mean(), inplace=True)
    elif i<59:
        df_train[col].fillna(df_train[col].mode()[0], inplace=True)
        df_test[col].fillna(df_train[col].mode()[0], inplace=True)
        df_val[col].fillna(df_train[col].mode()[0], inplace=True)
    i=i+1

#transformation
i=0
blah=dataset2.columns.tolist()
numerics=blah[0:10]
binaries=blah[11:60]
standardize = [([col], StandardScaler()) for col in numerics]
leave = [(col, None) for col in binaries]
x_mapper = DataFrameMapper(standardize + leave)

x_train = x_mapper.fit_transform(df_train).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')

#time discritization
num_durations = 10
labtrans = DeepHitSingle.label_transform(num_durations)
get_target = lambda dataset2: (dataset2['TIME'].values, dataset2['DIED365'].values)
y_train = labtrans.fit_transform(*get_target(df_train))
y_val = labtrans.transform(*get_target(df_val))

train = (x_train, y_train)
val = (x_val, y_val)

# We don't need to transform the test labels
durations_test, events_test = get_target(df_test)

#build net
in_features = x_train.shape[1]
num_nodes = [32, 32]
out_features = labtrans.out_features
batch_norm = True
dropout = 0.1

#train model
net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)
model = DeepHitSingle(net, tt.optim.Adam, alpha=0.2, sigma=0.1, duration_index=labtrans.cuts)
batch_size = 256
lr_finder = model.lr_finder(x_train, y_train, batch_size, tolerance=3)
_ = lr_finder.plot()

print(lr_finder.get_best_lr())
model.optimizer.set_lr(0.1)
epochs = 100
callbacks = [tt.callbacks.EarlyStopping()]
log = model.fit(x_train, y_train, batch_size, epochs, callbacks, val_data=val)
_ = log.plot()
#plt.show()

surv = model.predict_surv_df(x_test)
ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')


time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
ev.brier_score(time_grid).plot()
plt.ylabel('Brier score')
_ = plt.xlabel('Time')
plt.show()
print(ev.integrated_brier_score(time_grid))