import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

class LagNet(nn.Module):
    def __init__(self, q_dim):
        super(LagNet, self).__init__()
        self.q_dim = q_dim
        self.L_diag_dim = self.q_dim
        self.L_off_diag_dim = 0
        for i in range(self.q_dim):
            self.L_off_diag_dim = self.L_off_diag_dim + i
        self.L_net_fc1 = nn.Linear(self.q_dim, 512)
        self.L_net_fc2 = nn.Linear(512, 512)
        self.L_net_fc3 = nn.Linear(512, 512)
        self.L_net_diag = nn.Linear(512, self.L_diag_dim)
        self.L_net_off_diag = nn.Linear(512, self.L_off_diag_dim)
        print('q dimension', self.q_dim)
        self.g_net_fc1 = nn.Linear(self.q_dim, 512)
        self.g_net_fc2 = nn.Linear(512, 512)
        self.g_net_fc3 = nn.Linear(512, self.q_dim)

        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, q, q_t, q_tt):
        # q: [self.q_dim]
        # q_t: [self.q_dim]
        # q_tt: [self.q_dim]

        h1 = torch.sigmoid(self.L_net_fc1(q))
        h2 = torch.sigmoid(self.L_net_fc2(h1))
        h3 = torch.sigmoid(self.L_net_fc3(h2))
        l_diag = torch.exp(self.L_net_diag(h3))     # positive diagonal
        l_off_diag = self.L_net_off_diag(h3)        # real number off-diagonal

        L = self.get_L_mat(l_diag, l_off_diag)      # [q_dim, q_dim]

        dh1 = h1 * (1 - h1)
        dh1 = dh1.view(-1)
        dh1 = torch.diag_embed(dh1)
        dh1 = torch.matmul(dh1, self.L_net_fc1.weight)

        dh2 = h2 * (1 - h2)
        dh2 = dh2.view(-1)
        dh2 = torch.diag_embed(dh2)
        dh2 = torch.matmul(dh2, self.L_net_fc2.weight)

        dh3 = h3 * (1 - h3)
        dh3 = dh3.view(-1)
        dh3 = torch.diag_embed(dh3)
        dh3 = torch.matmul(dh3, self.L_net_fc3.weight)

        dh4_1 = l_diag
        dh4_1 = dh4_1.view(-1)
        dh4_1 = torch.diag_embed(dh4_1)
        dh4_1 = torch.matmul(dh4_1, self.L_net_diag.weight)

        dh4_2 = self.L_net_off_diag.weight

        dh4 = torch.cat((dh4_1, dh4_2), 0)

        dl_dq = torch.matmul(dh2, dh1)
        dl_dq = torch.matmul(dh3, dl_dq)
        dl_dq = torch.matmul(dh4, dl_dq)

        dL_dq = self.get_dL_dq_mat(dl_dq)

        comp_1 = torch.matmul(L.transpose(0, 1), q_tt.transpose(0, 1))
        comp_1 = torch.matmul(L, comp_1)

        comp_2 = torch.matmul(dL_dq.transpose(0, 1), q_t.transpose(0, 1))
        comp_2 = comp_2.squeeze(2)
        comp_2 = torch.matmul(comp_2, q_t.transpose(0, 1))
        comp_2 = torch.matmul(L, comp_2)

        comp_3_1 = torch.matmul(dL_dq, q_t.transpose(0, 1))
        comp_3_1 = comp_3_1.squeeze(2)
        comp_3_2 = torch.matmul(L.transpose(0, 1), q_t.transpose(0, 1))
        comp_3 = torch.matmul(comp_3_1, comp_3_2)

        comp_4_1 = torch.matmul(q_t, dL_dq.transpose(0, 1))

        comp_4_1 = comp_4_1.transpose(0, 1)
        comp_4_1 = comp_4_1.squeeze(0)
        comp_4_2 = torch.matmul(L.transpose(0, 1), q_t.transpose(0, 1))
        comp_4 = torch.matmul(comp_4_1, comp_4_2)

        comp_5 = torch.matmul(dL_dq.transpose(0, 1), q_t.transpose(0, 1))
        comp_5 = comp_5.squeeze(2)
        comp_5 = torch.matmul(L, comp_5)
        comp_5 = torch.matmul(q_t, comp_5)
        comp_5 = comp_5.transpose(0, 1)

        g1 = torch.sigmoid(self.g_net_fc1(q))
        g2 = torch.sigmoid(self.g_net_fc2(g1))
        g = self.g_net_fc3(g2).view(self.q_dim, 1)

        tau = comp_1 + comp_2 + comp_3 - 0.5 * comp_4 - 0.5 * comp_5 + g
        tau = tau.transpose(0, 1)
        return tau

    def get_L_mat(self, l_diag, l_off_diag):
        L = torch.zeros(self.q_dim, self.q_dim)
        for i in range(self.q_dim):
            L[i, i] = l_diag[0, i]

        L[1, 0] = l_off_diag[0, 0]
        for i in range(1, l_off_diag.size()[1]):
            r = 0
            last_r = 0
            inc = 1
            last_inc = 0
            while i>= r:
                last_r = r
                last_inc = inc
                r += inc
                inc += 1
            L[last_inc, i-last_r] = l_off_diag[0, i]
        return L

    def get_dL_dq_mat(self, dl_dq):
        # L = torch.zeros(self.q_dim, self.q_dim, self.q_dim)
        dl_dq_diag = dl_dq[0:self.q_dim, :]
        dl_dq_off_diag = dl_dq[self.q_dim:, :]
        # print('dl_dq_diag', dl_dq_diag.size())
        # print('dl_dq_off_diag', dl_dq_off_diag.size())
        temp_l_diag = dl_dq_diag[:, 0].view(1, -1)
        temp_l_off_diag = dl_dq_off_diag[:, 0].view(1, -1)
        dL_dq = self.get_L_mat(temp_l_diag, temp_l_off_diag).unsqueeze(2)
        # print('dL_dq', dL_dq.size())
        for q_index in range(1, self.q_dim):
            temp_l_diag = dl_dq_diag[:, q_index].view(1, -1)
            temp_l_off_diag = dl_dq_off_diag[:, q_index].view(1, -1)
            temp_L = self.get_L_mat(temp_l_diag, temp_l_off_diag).unsqueeze(2)
            dL_dq = torch.cat((dL_dq, temp_L), 2)
        # print('dL_dq', dL_dq.size())
        return dL_dq

    def get_H(self, q):
        h1 = torch.sigmoid(self.L_net_fc1(q))
        h2 = torch.sigmoid(self.L_net_fc2(h1))
        h3 = torch.sigmoid(self.L_net_fc3(h2))
        l_diag = torch.exp(self.L_net_diag(h3))  # positive diagonal
        l_off_diag = self.L_net_off_diag(h3)  # real number off-diagonal

        L = self.get_L_mat(l_diag, l_off_diag)
        H = torch.matmul(L, L.transpose(0, 1))
        return H

    def get_C(self, q, q_t):
        h1 = torch.sigmoid(self.L_net_fc1(q))
        h2 = torch.sigmoid(self.L_net_fc2(h1))
        h3 = torch.sigmoid(self.L_net_fc3(h2))
        l_diag = torch.exp(self.L_net_diag(h3))  # positive diagonal
        l_off_diag = self.L_net_off_diag(h3)  # real number off-diagonal
        L = self.get_L_mat(l_diag, l_off_diag)
        dh1 = h1 * (1 - h1)
        dh1 = dh1.view(-1)
        dh1 = torch.diag_embed(dh1)
        dh1 = torch.matmul(dh1, self.L_net_fc1.weight)

        dh2 = h2 * (1 - h2)
        dh2 = dh2.view(-1)
        dh2 = torch.diag_embed(dh2)
        dh2 = torch.matmul(dh2, self.L_net_fc2.weight)

        dh3 = h3 * (1 - h3)
        dh3 = dh3.view(-1)
        dh3 = torch.diag_embed(dh3)
        dh3 = torch.matmul(dh3, self.L_net_fc3.weight)

        dh4_1 = l_diag
        dh4_1 = dh4_1.view(-1)
        dh4_1 = torch.diag_embed(dh4_1)
        dh4_1 = torch.matmul(dh4_1, self.L_net_diag.weight)

        dh4_2 = self.L_net_off_diag.weight

        dh4 = torch.cat((dh4_1, dh4_2), 0)

        dl_dq = torch.matmul(dh2, dh1)
        dl_dq = torch.matmul(dh3, dl_dq)
        dl_dq = torch.matmul(dh4, dl_dq)

        dL_dq = self.get_dL_dq_mat(dl_dq)

        comp_2 = torch.matmul(dL_dq.transpose(0, 1), q_t.transpose(0, 1))
        comp_2 = comp_2.squeeze(2)
        comp_2 = torch.matmul(L, comp_2)

        comp_3 = torch.matmul(dL_dq, q_t.transpose(0, 1))
        comp_3 = comp_3.squeeze(2)
        comp_3 = torch.matmul(comp_3, L.transpose(0, 1))

        comp_4 = torch.matmul(q_t, dL_dq.transpose(0, 1))
        comp_4 = comp_4.transpose(0, 1)
        comp_4 = comp_4.squeeze(0)
        comp_4 = torch.matmul(comp_4, L.transpose(0, 1))

        comp_5 = torch.matmul(L, dL_dq)
        comp_5 = comp_5.transpose(0, 1)
        comp_5 = torch.matmul(q_t, comp_5.transpose(0, 1))
        comp_5 = comp_5.transpose(0, 1)
        comp_5 = comp_5.squeeze(0)
        C = comp_2 + comp_3 - 0.5 * comp_4 - 0.5 * comp_5
        return C

    def get_G(self, q):
        g1 = torch.sigmoid(self.g_net_fc1(q))
        g2 = torch.sigmoid(self.g_net_fc2(g1))
        g = self.g_net_fc3(g2).view(self.q_dim, 1)
        return g

    def get_weight_loss(self):
        weight_loss = torch.sum(torch.norm(self.L_net_fc1.weight, p=2, dim=1))
        weight_loss = weight_loss + torch.norm(self.L_net_fc1.bias, p=2, dim=0)
        weight_loss = weight_loss + torch.sum(torch.norm(self.L_net_fc2.weight, p=2, dim=1))
        weight_loss = weight_loss + torch.norm(self.L_net_fc2.bias, p=2, dim=0)
        weight_loss = weight_loss + torch.sum(torch.norm(self.L_net_fc3.weight, p=2, dim=1))
        weight_loss = weight_loss + torch.norm(self.L_net_fc3.bias, p=2, dim=0)

        weight_loss = weight_loss + torch.sum(torch.norm(self.L_net_diag.weight, p=2, dim=1))
        weight_loss = weight_loss + torch.norm(self.L_net_diag.bias, p=2, dim=0)
        weight_loss = weight_loss + torch.sum(torch.norm(self.L_net_off_diag.weight, p=2, dim=1))
        weight_loss = weight_loss + torch.norm(self.L_net_off_diag.bias, p=2, dim=0)

        weight_loss = weight_loss + torch.sum(torch.norm(self.g_net_fc1.weight, p=2, dim=1))
        weight_loss = weight_loss + torch.norm(self.g_net_fc1.bias, p=2, dim=0)
        weight_loss = weight_loss + torch.sum(torch.norm(self.g_net_fc2.weight, p=2, dim=1))
        weight_loss = weight_loss + torch.norm(self.g_net_fc2.bias, p=2, dim=0)
        weight_loss = weight_loss + torch.sum(torch.norm(self.g_net_fc3.weight, p=2, dim=1))
        weight_loss = weight_loss + torch.norm(self.g_net_fc3.bias, p=2, dim=0)
        return weight_loss

    def train(self, q, q_t, q_tt, tau):
        H = self.get_H(q)
        C = self.get_C(q, q_t)
        G = self.get_G(q)
        temp_tau = torch.matmul(H, q_tt.transpose(0, 1)) + torch.matmul(C, q_t.transpose(0, 1)) + G
        backward_loss = self.loss_fn(temp_tau.transpose(0, 1), tau)

        temp_q_tt = tau.transpose(0, 1) - torch.matmul(C, q_t.transpose(0, 1)) - G
        temp_q_tt = torch.matmul(H.inverse(), temp_q_tt)
        forward_loss = self.loss_fn(temp_q_tt.transpose(0, 1), q_tt)

        loss = backward_loss + forward_loss
        # print('backward_loss', backward_loss)
        # print('forward_loss', forward_loss)
        print('loss', forward_loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == '__main__':
    # Euler Lagrange Equation: H * q_tt + C * q_t + G = tau

    lag_net = LagNet(q_dim=4)
    q = torch.rand(1, 4)        #  joint configuration
    q_t = torch.rand(1, 4)      # joint velocity
    q_tt = torch.rand(1, 4)     # joint acceleration

    tau = torch.rand(1, 4)
    for i in range(50):
        lag_net.train(q, q_t, q_tt, tau)

    H = lag_net.get_H(q)
    C = lag_net.get_C(q, q_t)
    G = lag_net.get_G(q)

    print('H', H)
    print('C', C)
    print('G', G)


