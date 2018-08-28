
import pandas as pd
import numpy as np
from sklearn.model_selection import test_test_split
import matplotlib.pyplot as plt
from scipy import interpolate
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
%matplotlib inline

train_data = pd.read_csv('data\public.train.csv', header = 0, encoding = 'utf-8')
test_data = pd.read_csv('data\public.test.csv', header = 0, encoding = 'utf-8')
elevator = train_data['发电量']
train_data = train_data.drop(['发电量'],axis=1)
train_data['ID_1'] = range(0,9000)
test_data['ID_1'] = range(0,8409)

train_data.loc[(train_data['板温']<-24.79)|(train_data['板温']>36.8),'板温'] = None
train_data.loc[(train_data['现场温度']<-156.4)|(train_data['现场温度']>78.7),'现场温度'] = None
train_data.loc[(train_data['光照强度']<0.0)|(train_data['光照强度']>941.0),'光照强度'] = None
train_data.loc[(train_data['转换效率']<0.0)|(train_data['转换效率']>2633.95),'转换效率'] = None
train_data.loc[(train_data['转换效率A']<0.0)|(train_data['转换效率A']>3377.94),'转换效率A'] = None
train_data.loc[(train_data['转换效率B']<0.0)|(train_data['转换效率B']>5135.32),'转换效率B'] = None
train_data.loc[(train_data['转换效率C']<0.0)|(train_data['转换效率C']>2882.47),'转换效率C'] = None
train_data.loc[(train_data['电压A']<0.0)|(train_data['电压A']>807.0),'电压A'] = None
train_data.loc[(train_data['电压B']<0.0)|(train_data['电压B']>831.0),'电压B'] = None
train_data.loc[(train_data['电压C']<0.0)|(train_data['电压C']>777.0),'电压C'] = None
train_data.loc[(train_data['电流A']<0.0)|(train_data['电流A']>9.57),'电流A'] = None
train_data.loc[(train_data['电流B']<0.0)|(train_data['电流B']>10.19),'电流B'] = None
train_data.loc[(train_data['电流C']<0.0)|(train_data['电流C']>32.91),'电流C'] = None
train_data.loc[(train_data['功率A']<0.0)|(train_data['功率A']>7263.0),'功率A'] = None
train_data.loc[(train_data['功率B']<0.0)|(train_data['功率B']>26802.11),'功率B'] = None
train_data.loc[(train_data['功率C']<0.0)|(train_data['功率C']>19561.2),'功率C'] = None
train_data.loc[(train_data['平均功率']<0.0)|(train_data['平均功率']>19561.2),'平均功率'] = None
train_data.loc[(train_data['风速']<0.0)|(train_data['风速']>7.2),'风速'] = None
train_data.loc[(train_data['风向']<0.0)|(train_data['风向']>511.0),'风向'] = None

inter_linear = []
for i in range(len(train_data.columns)):
    print(i,train_data.columns[i]+'插值')
    inter_linear.append(interpolate.interp1d(list(train_data.loc[((train_data[train_data.columns[i]]).notnull()),'ID_1']), list(train_data.loc[((train_data[train_data.columns[i]]).notnull()),train_data.columns[i]])))
for i in range(len(train_data.columns)):
    x_new = (train_data.loc[((train_data[train_data.columns[i]]).isnull()),'ID_1'])
    print(i,train_data.columns[i])
    train_data.loc[x_new.index,train_data.columns[i]] = list(inter_linear[i](list(x_new.index)))

test_data.loc[(test_data['板温']<-24.79)|(test_data['板温']>36.8),'板温'] = None
test_data.loc[(test_data['现场温度']<-156.4)|(test_data['现场温度']>78.7),'现场温度'] = None
test_data.loc[(test_data['光照强度']<0.0)|(test_data['光照强度']>941.0),'光照强度'] = None
test_data.loc[(test_data['转换效率']<0.0)|(test_data['转换效率']>2633.95),'转换效率'] = None
test_data.loc[(test_data['转换效率A']<0.0)|(test_data['转换效率A']>3377.94),'转换效率A'] = None
test_data.loc[(test_data['转换效率B']<0.0)|(test_data['转换效率B']>5135.32),'转换效率B'] = None
test_data.loc[(test_data['转换效率C']<0.0)|(test_data['转换效率C']>2882.47),'转换效率C'] = None
test_data.loc[(test_data['电压A']<0.0)|(test_data['电压A']>807.0),'电压A'] = None
test_data.loc[(test_data['电压B']<0.0)|(test_data['电压B']>831.0),'电压B'] = None
test_data.loc[(test_data['电压C']<0.0)|(test_data['电压C']>777.0),'电压C'] = None
test_data.loc[(test_data['电流A']<0.0)|(test_data['电流A']>9.57),'电流A'] = None
test_data.loc[(test_data['电流B']<0.0)|(test_data['电流B']>10.19),'电流B'] = None
test_data.loc[(test_data['电流C']<0.0)|(test_data['电流C']>32.91),'电流C'] = None
test_data.loc[(test_data['功率A']<0.0)|(test_data['功率A']>7263.0),'功率A'] = None
test_data.loc[(test_data['功率B']<0.0)|(test_data['功率B']>26802.11),'功率B'] = None
test_data.loc[(test_data['功率C']<0.0)|(test_data['功率C']>19561.2),'功率C'] = None
test_data.loc[(test_data['平均功率']<0.0)|(test_data['平均功率']>19561.2),'平均功率'] = None
test_data.loc[(test_data['风速']<0.0)|(test_data['风速']>7.2),'风速'] = None
test_data.loc[(test_data['风向']<0.0)|(test_data['风向']>511.0),'风向'] = None

test_inter_linear = []
for i in range(len(test_data.columns)):
    print(i,test_data.columns[i]+'插值')
    test_inter_linear.append(interpolate.interp1d(list(test_data.loc[((test_data[test_data.columns[i]]).notnull()),'ID_1']), list(test_data.loc[((test_data[test_data.columns[i]]).notnull()),test_data.columns[i]])))
for i in range(len(test_data.columns)):
    x_new = (test_data.loc[((test_data[test_data.columns[i]]).isnull()),'ID_1'])
    print(i,test_data.columns[i])
    test_data.loc[x_new.index,test_data.columns[i]] = list(test_inter_linear[i](list(x_new.index)))

train_819 = train_data.drop(['ID_1'],axis=1)
test_819 = test_data.drop(['ID_1'],axis=1)
train_data['发电量'] = elevator
train_data.index = range(0,9000) 
test_data.index = range(9000,17409)
train_data.to_csv('train_data81920.csv',index=False,sep=',')
test_819.to_csv('test_data81920.csv',index=False,sep=',')