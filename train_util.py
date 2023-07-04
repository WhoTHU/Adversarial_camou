import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def reg_dist(x, dist='uniform', mode='cf', sample_num=200):
    i = 1
    t = x.new(size=[sample_num]).normal_()


    if dist == 'uniform':
        t_abs = t.abs()
        f_real = torch.sin(i * t_abs) / (t_abs + 1e-10)
        f_img = 2 * torch.sin(i * t_abs / 2).square() / (t_abs + 1e-10) * t.sign()
    else:
        raise NotImplementedError

    #     estimate f
    f_e_real = 1 / x.shape[-1] * torch.cos(t.unsqueeze(-1)*x.unsqueeze(-2)).sum(-1)
    f_e_img = 1 / x.shape[-1] * torch.sin(t.unsqueeze(-1)*x.unsqueeze(-2)).sum(-1)

    diff = (f_real - f_e_real)*(f_real - f_e_real) + (f_img - f_e_img)*(f_img - f_e_img)
    diff = diff  / (t * t + 1e-10) / (-t * t / 2).exp()
    diff = diff.sum(-1) / sample_num
    return diff
