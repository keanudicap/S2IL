import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, subgoal_dim, max_action):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32+subgoal_dim, 32)
        self.l4 = nn.Linear(32, action_dim)
        
        self.max_action = max_action
        
    def forward(self, x, subgoal):

        x = F.leaky_relu(self.l1(x))
        x = F.leaky_relu(self.l2(x))
        x = torch.cat([x, subgoal], 1)
        x = F.leaky_relu(self.l3(x))
        x = torch.softmax(self.l4(x),axis=0)
        return x

    def out_emb(self, x):
        x = F.leaky_relu(self.l1(x))
        x = F.leaky_relu(self.l2(x))
        return x


class subgoalEMB(nn.Module):
    def __init__(self, state_dim):
        super(subgoalEMB, self).__init__()

        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 32)

    def forward(self, state):
        x = F.leaky_relu(self.l1(state))
        x = F.leaky_relu(self.l2(x))
        return x




class S2IL:
    def __init__(self, state_dim, action_dim, max_action, lr, betas, expert_buffer, subgoal_dim):
        self.actor = Actor(state_dim, action_dim, subgoal_dim, max_action)
        self.optim_actor = torch.optim.SGD(self.actor.parameters(), lr=lr)
        self.subgoalEMB = subgoalEMB(state_dim)
        self.optim_subgoalEMB = torch.optim.Adam(self.subgoalEMB.parameters(), lr=lr*0.01, betas=betas)

        self.max_action = max_action
        self.expert = expert_buffer

        self.loss_fn = nn.BCELoss()
    
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        subgoal = self.subgoalEMB(state).detach()
        return self.actor(state, subgoal).cpu().data.numpy().flatten()


        
    def update(self, n_iter, alp, batch_size=100):

        loss_c_all = []
        loss_actor_all = []
        for i in range(n_iter):
            # sample expert transitions
            if i % 20 == 0:
                # sample expert states for actor
                exp_state, exp_action, pg, ngs = self.expert.sample(batch_size)
                exp_state = torch.FloatTensor(exp_state)
                exp_action = torch.FloatTensor(exp_action)
                state = torch.FloatTensor(exp_state)
                pg = torch.FloatTensor(pg)
                ngs = torch.FloatTensor(ngs)
                action = self.actor(state, self.subgoalEMB(state).detach())


                #######################
                # update subgoalEMB
                #######################
                self.optim_subgoalEMB.zero_grad()

                # score function calculation
                subgoal = self.subgoalEMB(exp_state)
                p_s = self.actor.out_emb(pg).detach()
                score_p = 0
                for i in range(subgoal.shape[0]):
                    temp_p = torch.exp(torch.matmul(subgoal[i], p_s[i].T)+0.0001)
                    score_p = score_p + temp_p


                score_n = score_p
                for i in range(3):
                    temp_n = self.actor.out_emb(ngs[:,i,:]).detach()
                    for j in range(temp_n.shape[0]):
                        score_n += torch.exp(torch.matmul(subgoal[i], temp_n[i].T)+0.0001)
                loss_subgoal = -torch.log2((score_p+0.00001)/(score_n+1))
                loss_subgoal = loss_subgoal/temp_n.shape[0]
                self.optim_subgoalEMB.zero_grad()
                loss_subgoal.backward()
                self.optim_subgoalEMB.step()


            ################
            # update policy
            ################
            exp_state, exp_action, pg, ngs = self.expert.sample(batch_size)
            exp_state = torch.FloatTensor(exp_state)
            exp_action = torch.FloatTensor(exp_action)
            state = torch.FloatTensor(exp_state)
            action = self.actor(state, self.subgoalEMB(state).detach())

            self.optim_actor.zero_grad()
            alp = alp
            loss_actor = self.loss_fn(action,exp_action)
            loss_actor.mean().backward(retain_graph=True)
            self.optim_actor.step()

            loss_actor_all.append(loss_actor.data.numpy())
        return loss_c_all

    def save(self, directory='./pre_trained', name='S2IL'):
        torch.save(self.actor.state_dict(), '{}/{}_actor.pth'.format(directory,name))
        torch.save(self.subgoalEMB.state_dict(), '{}/{}_subgoalEMB.pth'.format(directory,name))


    def load_actor(self, directory='./pre_trained', name='S2IL'):
        self.actor.load_state_dict(torch.load('{}/{}_S2IL_actor.pth'.format(directory, name)))
