import pandas as pd

app_train =pd.read_csv(r'public.train.csv')
app_test =pd.read_csv(r'public.test.csv')

train_id = app_train['ID']
test_id = app_test['ID']

app_train_test=[app_train,app_test]
app_train_test =pd.concat(app_train_test)

app_train_test=app_train_test.mask(app_train_test.sub(app_train_test.mean()).div(app_train_test.std()).abs().gt(3))
'''may change ffill'''
app_train_test=app_train_test.fillna(method='ffill')
app_train_test.to_csv(r'data-new/data_prc',index=False)