from patient_model import TrainPatientModel
import pandas as pd
import numpy as np
import random
import torch


class Environment(object):
    def __init__(self, path):
        self.path = path
        self.tm = TrainPatientModel(43, 25, 43, 128, 48, lr=1e-3)
        self.health_model = self.tm.load_weights('data/')
        self.initial_state = self.initial_state()
        self.state_space = 43
        self.action_space = 25
        self.seed = 1

    def initial_state(self):
        initial_state = []
        df = pd.read_csv(self.path)
        state_features = ['icustay_id', 'mechanical', 'out_put', '220277', 'admission_type', 'adm_order', 'gender',
                          'sofa', 'age', 'weight', 'height', 'Arterial_BE', 'CO2_mEqL', 'Ionised_Ca', 'Glucose', 'Hb',
                          'Arterial_lactate', 'paCO2', 'ArterialpH', 'paO2', 'SGPT', 'Albumin', 'SGOT', 'HCO3',
                          'Direct_bili', 'CRP', 'Calcium', 'Chloride', 'Creatinine', 'Magnesium', 'Potassium_mEqL',
                          'Total_protein', 'Sodium', 'Troponin', 'BUN', 'Ht', 'INR', 'Platelets_count', 'PT', 'PTT',
                          'RBC_count', 'WBC_count', 'Total_bili', 'flag']
        train_states = df[state_features].values
        init_id = train_states[0, 0]
        initial_state.append(train_states[0, 1:])
        for i in range(1, train_states.shape[0]):
            if train_states[i, 0] != init_id:
                initial_state.append(train_states[i, 1:])
                init_id = train_states[i, 0]

        initial_state = np.array(initial_state)
        return initial_state

    def reset(self):
        ind = random.randint(0, self.initial_state.shape[0]-1)
        return self.initial_state[ind]

    def step(self,state, action):
        state = torch.tensor(state,dtype=torch.float32)
        action = torch.tensor(action,dtype=torch.float32)
        next_state = self.tm.hm.forward(state, action)
        return next_state

