import numpy as np
import pandas as pd
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from imblearn.over_sampling import RandomOverSampler
import random

random_seed = 0
class ExpertTraj:
    def __init__(self):

        """
        positive trajectory
        """
        all_traj = pd.read_csv('data/demo_sepsis_train.csv')
        df2 = pd.read_csv('data/demo_sepsis_val.csv')
        all_traj.append(df2)

        all_traj['flag'] = all_traj['flag']
        # print(df_train['flag'])
        all_traj['flag'] = all_traj['flag'].replace({0: np.nan})
        all_traj = all_traj.bfill(axis=0)



        all_user = list(all_traj['icustay_id'].values)
        train_x = []
        test_x = []
        val_x = []
        count = 0
        for i in all_user:
            temp_user = all_traj[all_traj['icustay_id'] == i].values
            len_user = temp_user.shape[0]
            len_tr = int(np.rint(len_user * 0.5))
            len_v = int(np.rint(len_user * 0.5))
            # len_te = int(np.rint(len_user * 0.1))
            print('count',count)
            count += 1

            if len_tr > 0:
                tr_x = list(temp_user[:len_tr, :])
            if len_v > 0:
                v_x = list(temp_user[len_tr:(len_tr + len_v), :])
            # if len_te > 0:
            #     te_x = list(temp_user[(len_tr + len_v):(len_tr + len_v + len_te), :])

            val_x.extend(v_x)
            train_x.extend(tr_x)
        val_x = DataFrame(np.array(val_x),columns=all_traj.columns)
        train_x =  DataFrame(np.array(train_x),columns=all_traj.columns)

        # val_x = pd.read_csv('data/sepsis_val_bytime.csv')[:2000]
        # train_x= pd.read_csv('data/sepsis_train_bytime.csv')[:10000]


        df_train = train_x

        df_train = df_train.bfill(axis=0)
        # df = df_train[df_train['flag'] == 15]
        df = df_train

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

        self.state_dim = len(state_features)
        self.action_dim = 25

        ac = df['action'].values.astype(int)
        # ros = RandomOverSampler(random_state=0)
        # self.exp_states, ac = ros.fit_resample(self.exp_states, ac)

        b = np.zeros((ac.shape[0], 25))
        b[np.arange(ac.shape[0]), ac] = 1

        self.exp_actions = b

        self.n_transitions = len(self.exp_actions)

        """
        positive subgoal trajective
        """

        g_state = self.exp_states
        print('g_state', g_state.shape)
        temp_exp_pos_goal = []
        for i in range(1,(g_state.shape[0]-4)):
            if i <= 5:
                g_temp = g_state[i-1]

            elif i%5 == 0:
                g_temp = g_state[(i+5) - 1]

            temp_exp_pos_goal.append(g_temp)


        self.exp_states_positive_goal = np.array(temp_exp_pos_goal)
        self.exp_states = self.exp_states[:-5]
        self.exp_actions = self.exp_actions[:-5]
        temp_exp_pos_goal = []
        for i in range(self.exp_states_positive_goal.shape[0]):
            index_neg = np.random.randint(self.exp_states_positive_goal.shape[0], size=(3))
            temp_exp_pos_goal.append(self.exp_states_positive_goal[index_neg])
        self.exp_states_negative_goal = np.array(temp_exp_pos_goal)
        print('self.exp_states_negative_goal', self.exp_states_negative_goal.shape)


        live_val = val_x
        # live_val = live_val[live_val['flag'] == 15]
        live_val = live_val.bfill(axis=0)
        self.selflive_state = live_val[state_features].values
        self.selflive_state = min_max_scaler.fit_transform(self.selflive_state)
        ac = live_val['action'].values.astype(int)



        b = np.zeros((ac.shape[0], 25))
        b[np.arange(ac.shape[0]), ac] = 1
        self.selflive_action = b

        val_g_state = self.selflive_state

        temp_exp_pos_goal = []
        for i in range(1, (val_g_state.shape[0] - 4)):
            if i <= 5:
                g_temp = val_g_state[i - 1]
            elif i % 5 == 0:
                g_temp = val_g_state[(i + 5) - 1]
            temp_exp_pos_goal.append(g_temp)
        self.eval_states_positive_goal = np.array(temp_exp_pos_goal)
        self.selflive_state = self.selflive_state[:-5]
        self.selflive_action = self.selflive_action[:-5]

        print('self.selflive_state',self.selflive_state.shape)
        print('self.selflive_action', self.selflive_action.shape)

        """
        negative subgoal trajectories
        """


        temp_exp_pos_goal = []
        for i in range(self.eval_states_positive_goal.shape[0]):
            index_neg = np.random.randint(self.eval_states_positive_goal.shape[0]-1, size=(3))
            temp_exp_pos_goal.append(self.eval_states_positive_goal[index_neg])
        self.eval_states_negative_goal = np.array(temp_exp_pos_goal)
        print('self.eval_states_negative_goal',self.eval_states_negative_goal.shape)



    def eval(self):

        return np.array( self.selflive_state), np.array(self.selflive_action), np.array( self.eval_states_positive_goal), np.array(self.eval_states_negative_goal)

    def train(self):

        return np.array(self.exp_states), np.array( self.exp_actions ),np.array(self.exp_states_positive_goal), np.array( self.exp_states_negative_goal )

    # Expects tuples of (state, next_state, action, reward, done)
    def add(self, state, action):
        self.exp_states = np.insert(self.exp_states, -1, values=state, axis=0)
        self.exp_actions = np.insert(self.exp_actions, -1, values=action, axis=0)

    def sample(self, batch_size):
        indexes = np.random.randint(0, self.n_transitions-5, size=batch_size)
        state, action, pos_g, neg_gs= [], [], [], []
        for i in indexes:
            s = self.exp_states[i]
            a = self.exp_actions[i]
            pg = self.exp_states_positive_goal[i]
            ng = self.exp_states_negative_goal[i]

            # print('a',a)
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            pos_g.append(np.array(pg, copy=False))
            neg_gs.append(np.array(ng, copy=False))
        return np.array(state), np.array(action), np.array(pos_g), np.array(neg_gs)

    def sample_val(self, batch_size):
        n_transitions = self.selflive_state.shape[0]
        indexes = np.random.randint(0, n_transitions, size=batch_size)
        state, action, pos_g, neg_gs= [], [], [], []
        for i in indexes:
            s = self.selflive_state[i]
            a = self.selflive_action[i]
            pg = self.exp_states_positive_goal[i]
            ng = self.exp_states_negative_goal[i]

            # print('a',a)
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            pos_g.append(np.array(pg, copy=False))
            neg_gs.append(np.array(ng, copy=False))
        return np.array(state), np.array(action), np.array(pos_g), np.array(neg_gs)

# a = ExpertTraj()
# b1, b2 = a.train()
# print(b2.shape)
#
# b1, b2 = a.eval()
# print(b2.shape)