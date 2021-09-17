import numpy as np
import pandas as pd
import pandas as pd
from sklearn import preprocessing
from imblearn.over_sampling import RandomOverSampler
import random

random_seed = 0
class ExpertTraj:
    def __init__(self):

        """
        positive trajectory
        """
        df1 = pd.read_csv('data/demo_sepsis_train.csv')
        df2 = pd.read_csv('data/demo_sepsis_val.csv')
        df1.append(df2)

        df_ids = list(df1['icustay_id'].values)
        df_ids = set(df_ids)
        len_df_ids = len(df_ids)

        train_len = int(len_df_ids*0.7)
        train_ids = random.sample(df_ids,train_len)
        val_ids = set(df_ids)-set(train_ids)

        df_train = df1.loc[df1['icustay_id'].isin(train_ids)]
        df_train['indicate'] = df_train['flag']
        df_train['indicate'] = df_train['indicate'].replace({0: np.nan})
        df_train = df_train.bfill(axis=0)
        df = df_train[df_train['indicate'] == 15]

        state_features = ['mechanical', 'out_put', '220277', 'admission_type', 'adm_order', 'gender', 'sofa', 'age',
                          'weight', 'height', 'Arterial_BE', 'CO2_mEqL', 'Ionised_Ca', 'Glucose', 'Hb',
                          'Arterial_lactate', 'paCO2', 'ArterialpH', 'paO2', 'SGPT', 'Albumin', 'SGOT', 'HCO3',
                          'Direct_bili', 'CRP', 'Calcium', 'Chloride', 'Creatinine', 'Magnesium', 'Potassium_mEqL',
                          'Total_protein', 'Sodium', 'Troponin', 'BUN', 'Ht', 'INR', 'Platelets_count', 'PT', 'PTT',
                          'RBC_count', 'WBC_count', 'Total_bili']

        x = df.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()


        self.exp_states = df[state_features].values
        self.exp_states = min_max_scaler.fit_transform(self.exp_states)


        ac = df['action'].values.astype(int)
        ros = RandomOverSampler(random_state=0)
        self.exp_states, ac = ros.fit_resample(self.exp_states, ac)

        b = np.zeros((ac.shape[0], 25))
        b[np.arange(ac.shape[0]), ac] = 1

        self.exp_actions = b



        self.n_transitions = len(self.exp_actions)

        """
        negative trajective
        """

        # df_n = pd.read_csv('data/demo_sepsis_train.csv')
        df_n = df_train
        df_n['indicate'] = df_n['flag']
        df_n['indicate'] = df_n['indicate'].replace({0: np.nan})
        df_n = df_n.bfill(axis=0)
        df_n = df_n[df_n['indicate'] == -15]

        self.exp_states_n = df_n[state_features].values
        ac = df_n['action'].values.astype(int)
        print('self.exp_states_n',self.exp_states_n.shape)
        print('ac', ac.shape)
        self.exp_states_n, ac = ros.fit_resample(self.exp_states_n, ac)

        b = np.zeros((ac.shape[0], 25))
        b[np.arange(ac.shape[0]), ac] = 1
        self.exp_actions_n = b

        self.n_transitions_n = len(self.exp_actions_n)

        # live_val = pd.read_csv('data/demo_sepsis_val.csv')
        print('val_ids',val_ids)
        live_val = df1.loc[df1['icustay_id'].isin(val_ids)]
        live_val['indicate'] = live_val['flag']
        live_val['indicate'] = live_val['indicate'].replace({0: np.nan})
        live_val = live_val.bfill(axis=0)
        live_val = live_val[live_val['indicate'] == 15]

        self.selflive_state = live_val[state_features].values
        ac = live_val['action'].values.astype(int)

        b = np.zeros((ac.shape[0], 25))
        b[np.arange(ac.shape[0]), ac] = 1
        self.selflive_action = b
        print('self.selflive_state',self.selflive_state.shape)
        print('self.selflive_action', self.selflive_action.shape)



    def eval(self):

        return np.array( self.selflive_state), np.array(self.selflive_action)

    def train(self):

        return np.array(self.exp_states), np.array( self.exp_actions )

    # Expects tuples of (state, next_state, action, reward, done)
    def add(self, state, action):
        self.exp_states = np.insert(self.exp_states, -1, values=state, axis=0)
        self.exp_actions = np.insert(self.exp_actions, -1, values=action, axis=0)

    def sample(self, batch_size):
        indexes = np.random.randint(0, self.n_transitions, size=batch_size)
        state, action= [], []
        for i in indexes:
            s = self.exp_states[i]
            a = self.exp_actions[i]

            # print('a',a)
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
        return np.array(state), np.array(action)

    def sample_val(self, batch_size):
        n_transitions = self.selflive_state.shape[0]
        indexes = np.random.randint(0, n_transitions, size=batch_size)
        state, action = [], []
        for i in indexes:
            s = self.selflive_state[i]
            a = self.selflive_action[i]

            # print('a',a)
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
        return np.array(state), np.array(action)

    def sample_n(self, batch_size):
        indexes = np.random.randint(0, self.n_transitions_n, size=batch_size)
        state, action = [], []
        for i in indexes:
            s = self.exp_states_n[i]
            a = self.exp_actions_n[i]
            # print('a',a)
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
        return np.array(state), np.array(action)
a = ExpertTraj()
b1, b2 = a.train()
print(b2.shape)

b1, b2 = a.eval()
print(b2.shape)