import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
import numpy as np
from pandas import  DataFrame
state_features = ['icustay_id','row_id','valuenum1','valuenum2','mechanical', 'out_put', '220277', 'admission_type', 'adm_order', 'gender', 'sofa', 'age',
                  'weight', 'height', 'Arterial_BE', 'CO2_mEqL', 'Ionised_Ca', 'Glucose', 'Hb',
                  'Arterial_lactate', 'paCO2', 'ArterialpH', 'paO2', 'SGPT', 'Albumin', 'SGOT', 'HCO3',
                  'Direct_bili', 'CRP', 'Calcium', 'Chloride', 'Creatinine', 'Magnesium', 'Potassium_mEqL',
                  'Total_protein', 'Sodium', 'Troponin', 'BUN', 'Ht', 'INR', 'Platelets_count', 'PT', 'PTT',
                  'RBC_count', 'WBC_count', 'Total_bili', 'flag']


df = pd.read_csv('./data/demo_sepsis_train.csv')
df['flag'] = df['flag']
df['flag'] = df['flag'].replace({0: np.nan})
df = df.bfill(axis=0)
d1 = df[df['flag'] == 15]
d1_x = d1[state_features].values
d1_y = d1['action'].values
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(d1_x, d1_y)
from collections import Counter
print(sorted(Counter(list(y_resampled)).items()))
y_resampled = y_resampled.reshape(-1,1)

print(X_resampled.shape)
print(y_resampled.shape)
state_features.append('action')
xy_resampled = np.concatenate([X_resampled,y_resampled],axis=1)
xy_resampled = DataFrame(xy_resampled)
print(list(xy_resampled.columns ))
print(len(state_features))
xy_resampled.columns = [state_features]
xy_resampled.to_csv('./data/demo_sepsis_train_oversample.csv',index=False)
# y_all = TSNE(n_components=2,perplexity = 350).fit_transform(X_resampled)
# plt.scatter(y_all[:, 0], y_all[:, 1], s=10,c = y_resampled)
# plt.show()
# plt.plot()