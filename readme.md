# Adversarial Coorperative Imitation Learning 

we propose the S2IL model, to deduce the optimal dynamic treatment regimes.
###Dependencies

This code requires the following:

* numpy
* scipy
* torch
* scikit-learn
* pandas

`pip install -r requirements.txt` to install all dependencies

###Data
training dataset: `demo_sepsis_train.csv`
validation dataset: `demo_sepsis_train.csv`

Due the licence of MIMIC-III, we are not allowed to share the dataset pernally. 
You can download MIMIC-III from 'https://physionet.org/content/mimiciii-demo/1.4/' and follow the guidence of 'https://github.com/MIT-LCP/mimic-code'
to preprocess the dataset.

columns of the csv files:
* 'icustay_i': id of the sepsis patient
* 'row_id': time step of the patient

```
state_features = [mechanical', 'out_put', '220277', 'admission_type', 'adm_order', 'gender', 'sofa', 'age',
                  'weight', 'height', 'Arterial_BE', 'CO2_mEqL', 'Ionised_Ca', 'Glucose', 'Hb',
                  'Arterial_lactate', 'paCO2', 'ArterialpH', 'paO2', 'SGPT', 'Albumin', 'SGOT', 'HCO3',
                  'Direct_bili', 'CRP', 'Calcium', 'Chloride', 'Creatinine', 'Magnesium', 'Potassium_mEqL',
                  'Total_protein', 'Sodium', 'Troponin', 'BUN', 'Ht', 'INR', 'Platelets_count', 'PT', 'PTT',
                  'RBC_count', 'WBC_count', 'Total_bili', 'flag']
action = 'action'
```


###Usage

Usage Instructions:

1. S2IL model training 

``> python train.py``


###Contact

To ask questions or report issues, please open an issue on the issues tracker. Or send an email to wyu@nec-labs.com.