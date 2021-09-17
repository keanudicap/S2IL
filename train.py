import torch
import numpy as np
from s2il_full_model import S2IL
from environment import Environment
from sklearn.metrics import roc_auc_score, jaccard_score, accuracy_score
from utils_merge_bytime import ExpertTraj

def eval_result(policy, exp_state, exp_action):
    action_ini = policy.actor(torch.tensor(exp_state, dtype=torch.float32), policy.subgoalEMB(torch.tensor(exp_state, dtype=torch.float32)))
    action_ini = action_ini.data.numpy()
    action = action_map(action_ini)
    exp_action[exp_action > 0] = 1
    jac = accuracy_score(exp_action, action)
    auc_micro = roc_auc_score(exp_action, action, average='micro')

    return jac, auc_micro

def action_map(action):
    action_temp = np.zeros_like(action)
    for i in range(action.shape[0]):
        action_temp[i][np.argmax(action[i])] = 1
    return action_temp


def train():

    ######### Hyperparameters #########
    expert_buffer = ExpertTraj()
    agent_buffer = ExpertTraj()
    max_action = 1
    random_seed = 0
    lr = 0.015             # learing rate 0.001
    betas = (0.5, 0.999)    # betas for adam optimizer
    n_epochs = 5000          # number of epochs
    n_iter = 3              # number of updates per epoch
    batch_size = 48      # num of transitions sampled from expert
    subgoal_dim = 32
    ###################################

    train_file = 'data/demo_sepsis_train.csv'
    env = Environment(train_file)

    jaccard_all_train = []
    jaccard_all_val = []

    micro_train = []
    micro_val = []

    state_dim = expert_buffer.state_dim
    action_dim = expert_buffer.action_dim

    policy = S2IL(state_dim, action_dim, max_action, lr, betas, expert_buffer,subgoal_dim)

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        # env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    # training procedure
    alpha_all = []
    for alp in [0.5]:
        for epoch in range(1, n_epochs+1):
            policy.update(n_iter, alp, batch_size)

            # evaluate
            if epoch % 10 == 0:
                live_state, live_action, live_pg, live_ngs = expert_buffer.eval()
                jac, auc_micro = eval_result(policy, live_state, live_action, False)
                print("iter %d, VAL jaccard %.4f, auc %.4f" % (epoch, jac, auc_micro))
                jaccard_all_val.append(jac)
                micro_val.append(auc_micro)
                train_state, train_action, train_pg, train_ngs = expert_buffer.sample(100)
                jac, auc_micro = eval_result(policy, train_state, train_action, False)
                print("iter %d, TRAIN jaccard %.4f, auc %.4f" % (epoch, jac, auc_micro))
                jaccard_all_train.append(jac)
                micro_train.append(auc_micro)
        np.save('jaccard_all_val_all.npy',np.array(jaccard_all_val))
        np.save('jaccard_all_train_all.npy',np.array(jaccard_all_train), )
        np.save('micro_train_all.npy',np.array(micro_train),)
        np.save('micro_val_all.npy',np.array(micro_val) )


if __name__ == '__main__':
    train()
