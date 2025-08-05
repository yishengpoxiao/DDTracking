import math
import torch

import numpy as np
from torch import nn
from torch.nn import functional as F


class VarianceSchedule(nn.Module):

    def __init__(self, num_steps, mode='linear', beta_1=1e-4, beta_T=5e-2, cosine_s=8e-3):
        super().__init__()
        assert mode in ('linear', 'cosine')
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == 'cosine':
            timesteps = (
                    torch.arange(num_steps + 1) / num_steps + cosine_s
            )
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)  # Padding
        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()
        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i - 1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)
        # self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alpha_bars))
        # self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alpha_bars - 1))

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps + 1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas


class D2MP_OB(nn.Module):

    def __init__(self, loss_type, net, var_sched: VarianceSchedule, eps=1e-2):
        super().__init__()
        if loss_type in ['l1', 'l2', 'cosine', 'w_cosine']:
            self.loss_type = loss_type
        else:
            raise ValueError('Undefine loss type')
        self.net = net
        self.var_sched = var_sched
        self.eps = eps
        self.weight = True

    def q_sample(self, x_start, noise, t, C): # t: (B, )
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1))) # (B, d)
        x_noisy = x_start + C * time + torch.sqrt(time) * noise # (B, d)
        return x_noisy

    def pred_x0_from_xt(self, xt, noise, C, t):
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        x0 = xt - C * time - torch.sqrt(time) * noise
        return x0

    def pred_C_from_xt(self, xt, noise, t):
        time = t.reshape(noise.shape[0], *((1,) * (len(noise.shape) - 1)))
        C = (xt - torch.sqrt(time) * noise) / (time - 1)
        return C

    def pred_xtms_from_xt(self, xt, noise, C, t, s):
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        s = s.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        mean = xt + C * (time - s) - C * time - s / torch.sqrt(time) * noise
        epsilon = torch.randn_like(mean, device=xt.device)
        sigma = torch.sqrt(s * (time - s) / time)
        xtms = mean + sigma * epsilon
        return xtms

    def forward(self, x_0, global_feat, local_feat=None, mask=None, t=None):
        batch_size, point_dim = x_0.size()
        if t == None:
            t = torch.rand(x_0.shape[0], device=x_0.device) * (1. - self.eps) + self.eps # (B,)

        beta = t.log() / 4
        e_rand = torch.randn_like(x_0) # (B, d)
        C = -1 * x_0 # (B, d)
        x_noisy = self.q_sample(x_start=x_0, noise=e_rand, t=t, C=C)
        t = t.reshape(-1, 1)

        pred = self.net(x_noisy, beta, global_cond=global_feat, local_cond=local_feat)
        C_pred = pred
        noise_pred = (x_noisy - (t - 1) * C_pred) / t.sqrt()

        if self.loss_type == 'l1':
            if mask is not None:
                loss_C = F.smooth_l1_loss(C_pred.view(-1, point_dim), C.view(-1, point_dim), reduction='none') * mask
                loss_noise = F.smooth_l1_loss(noise_pred.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='none') * mask
            else:
                loss_C = F.smooth_l1_loss(C_pred.view(-1, point_dim), C.view(-1, point_dim), reduction='none')
                loss_noise = F.smooth_l1_loss(noise_pred.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='none')

        elif self.loss_type == 'l2':
            if mask is not None:
                loss_C = F.mse_loss(C_pred.view(-1, point_dim), C.view(-1, point_dim), reduction='none') * mask
                loss_noise = F.mse_loss(noise_pred.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='none') * mask
            else:
                loss_C = F.mse_loss(C_pred.view(-1, point_dim), C.view(-1, point_dim), reduction='none')
                loss_noise = F.mse_loss(noise_pred.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='none')
        elif self.loss_type == 'cosine':
            if mask is not None:
                loss_C = 1 - torch.clamp(F.cosine_similarity(C_pred.view(-1, point_dim), C.view(-1, point_dim), dim=-1, eps=self.eps), -1, 1).unsqueeze(1) * mask
                loss_noise = 1 - torch.clamp(F.cosine_similarity(noise_pred.view(-1, point_dim), e_rand.view(-1, point_dim), dim=-1, eps=self.eps), -1, 1).unsqueeze(1) * mask
            else:
                loss_C = 1 - torch.clamp(F.cosine_similarity(C_pred.view(-1, point_dim), C.view(-1, point_dim), dim=-1, eps=self.eps), -1, 1).unsqueeze(1)
                loss_noise = 1 - torch.clamp(F.cosine_similarity(noise_pred.view(-1, point_dim), e_rand.view(-1, point_dim), dim=-1, eps=self.eps), -1, 1).unsqueeze(1)
        elif self.loss_type == 'w_cosine':
            if mask is not None:
                loss_C1 = F.smooth_l1_loss(C_pred.view(-1, point_dim), C.view(-1, point_dim), reduction='none') * mask
                loss_noise1 = F.smooth_l1_loss(noise_pred.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='none') * mask
                
                loss_C2 = (1 - torch.clamp(F.cosine_similarity(C_pred.view(-1, point_dim), C.view(-1, point_dim), dim=-1, eps=self.eps), -1, 1).unsqueeze(1) * mask) ** 2
                loss_noise2 = (1 - torch.clamp(F.cosine_similarity(noise_pred.view(-1, point_dim), e_rand.view(-1, point_dim), dim=-1, eps=self.eps), -1, 1).unsqueeze(1) * mask) ** 2
            else:
                loss_C1 = F.smooth_l1_loss(C_pred.view(-1, point_dim), C.view(-1, point_dim), reduction='none')
                loss_noise1 = F.smooth_l1_loss(noise_pred.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='none')
                
                loss_C2 = (1 - torch.clamp(F.cosine_similarity(C_pred.view(-1, point_dim), C.view(-1, point_dim), dim=-1, eps=self.eps), -1, 1).unsqueeze(1)) ** 2
                loss_noise2 = (1 - torch.clamp(F.cosine_similarity(noise_pred.view(-1, point_dim), e_rand.view(-1, point_dim), dim=-1, eps=self.eps), -1, 1).unsqueeze(1)) ** 2
                
            loss_C = loss_C1 / (self.eps + loss_C1.detach()) + loss_C2 / (self.eps + loss_C2.detach())
            loss_noise = loss_noise1 / (self.eps + loss_noise1.detach()) + loss_noise2 / (self.eps + loss_noise2.detach())
            
        if not self.weight:
            loss = 0.5 * loss_C + 0.5 * loss_noise
        else:
            simple_weight1 = (t ** 2 - t + 1) / (t + self.eps)
            simple_weight2 = (t ** 2 - t + 1) / (1 - t + self.eps)
            
            simple_weight1 = torch.clamp(simple_weight1, max=1/self.eps)
            simple_weight2 = torch.clamp(simple_weight2, max=1/self.eps)

            loss = simple_weight1 * loss_C + simple_weight2 * loss_noise
            
        if mask is not None:
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.mean()

            # loss = F.smooth_l1_loss(noise_pred.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')

        return loss

    def sample(self, global_feat, bestof, point_dim=3, ret_traj=False, local_feat=None):
        # context = context.to(self.var_sched.betas.device)
        batch_size = global_feat.size(0)
        if bestof:
            x_T = torch.randn([batch_size, point_dim]).to(global_feat.device)
        else:
            x_T = torch.zeros([batch_size, point_dim]).to(global_feat.device)

        self.var_sched.num_steps = 1
        traj = {self.var_sched.num_steps: x_T}

        cur_time = torch.ones((batch_size,), device=x_T.device)
        step = 1. / self.var_sched.num_steps
        for t in range(self.var_sched.num_steps, 0, -1):
            s = torch.full((batch_size,), step, device=x_T.device)
            if t == 1:
                s = cur_time

            x_t = traj[t]
            beta = cur_time.log() / 4
            t_tmp = cur_time.reshape(-1, 1)
            pred = self.net(x_t, beta, global_cond=global_feat, local_cond=local_feat)
            C_pred = pred
            noise_pred = (x_t - (t_tmp - 1) * C_pred) / t_tmp.sqrt()

            x0 = self.pred_x0_from_xt(x_t, noise_pred, C_pred, cur_time)
            x0.clamp_(-1., 1.)
            C_pred = -1 * x0
            x_next = self.pred_xtms_from_xt(x_t, noise_pred, C_pred, cur_time, s)
            cur_time = cur_time - s
            traj[t - 1] = x_next.detach()  # Stop gradient and save trajectory.
            if not ret_traj:
                del traj[t]

        return traj[0]
