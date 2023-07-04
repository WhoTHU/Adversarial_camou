import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from PIL import Image, ImageDraw
from torch.nn.modules.activation import LeakyReLU, ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.instancenorm import InstanceNorm2d
from torch.nn.modules.linear import Linear
import torch.nn.init as init


def hsv2rgb(hsv, saturate = None, color = None, bright = None):
    assert hsv.shape[1] == 3
    hsv_t = torch.zeros(hsv.shape).to(hsv.device)
    if saturate is not None:
        hsv_t[:,1] = torch.mul(hsv[:,1], saturate[1]) + torch.mul((1-hsv[:,1]),saturate[0])
    else:
        hsv_t[:,1] = hsv[:,1]
    if color is not None:
        hsv_t[:,0] = (hsv[:,0] * (color[1]-color[0]) + color[0]) % 360
    else:
        hsv_t[:,0] = hsv[:,0]*360
    if bright is not None:
        hsv_t[:,2] = hsv[:,2] * (bright[1]-bright[0]) + bright[0]
    else:
        hsv_t[:,2] = hsv[:,2]
    rgb = torch.zeros(hsv.shape).to(hsv.device)
    rgb_t = torch.zeros(hsv.shape).to(hsv.device)
    rgb[:,0] = hsv_t[:,2]
    rgb[:,1] = hsv_t[:,2] - hsv_t[:,1] * hsv_t[:,2] * torch.abs((hsv_t[:,0]/60)%2 -1)
    rgb[:,2] = hsv_t[:,2] - hsv_t[:,1]*hsv_t[:,2]
    # print(rgb)
    rgb_t[:,0] = torch.where(hsv_t[:,0] < 120, rgb[:,1], rgb[:,2])
    rgb_t[:,0] = torch.where(hsv_t[:,0]>= 240, rgb[:,1], rgb_t[:,0])
    rgb_t[:,0] = torch.where(hsv_t[:,0] < 60, rgb[:,0], rgb_t[:,0])
    rgb_t[:,0] = torch.where(hsv_t[:,0]>= 300, rgb[:,0], rgb_t[:,0])
    rgb_t[:,1] = torch.where(hsv_t[:,0] < 240, rgb[:,1], rgb[:,2])
    rgb_t[:,1] = torch.where(hsv_t[:,0] < 180, rgb[:,0], rgb_t[:,1])
    rgb_t[:,1] = torch.where(hsv_t[:,0] < 60, rgb[:,1], rgb_t[:,1])
    rgb_t[:,2] = torch.where(hsv_t[:,0]>= 120, rgb[:,1], rgb[:,2])
    rgb_t[:,2] = torch.where(hsv_t[:,0]>= 180, rgb[:,0], rgb_t[:,2])
    rgb_t[:,2] = torch.where(hsv_t[:,0]>= 300, rgb[:,1], rgb_t[:,2])
    return rgb_t


def blend(img1, img2):
    # Blend the second RGBA image to the first one. 
    # Note: This is not a symmetric function for the two images!
    # input images must be of the same size (N,4,H,W), RGBA.
    assert img1.shape[1] == 4 and img2.shape[1] == 4
    assert img1.shape[2] == img2.shape[2] and img1.shape[3] == img2.shape[3]
    img_blended_color = img1[:,:3,:,:]* img1[:,3,:,:](1-img2[:,3,:,:]) + img2[:,:3,:,:]*img2[:,3,:,:]
    img_blended_alpha = img1[:,3,:,:] + img2[:,3,:,:] - img1[:,3,:,:] * img2[:,3,:,:]
    img_blended = torch.cat([img_blended_color,img_blended_alpha],dim=1)
    return img_blended


def random_mask(figsize, num_geometry, prev_mask=None):
    # return a random 0/1 mask
    img_np = np.zeros([figsize,figsize,3])
    for i in range(num_geometry):
        img = Image.new("RGB", (figsize,figsize))
        draw = ImageDraw.Draw(img)
        x = np.random.randint(1,10)
        center = np.random.randint(0,figsize,[2])
        if x == 1:
            angle = np.random.randint(0,360,2)
            radius = np.random.randint(20,figsize/4)
            draw.chord([center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius],angle[0], angle[1],fill=(1,1,1))
        if x == 2:
            radius = np.random.randint(20,figsize/4)
            draw.ellipse([center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius],fill=(1,1,1))
        if x == 3:
            offsets = np.random.randint(-100, 100, [3,2])
    #         print(center+offsets[0])
            draw.polygon([center[0]+offsets[0,0],center[1]+offsets[0,1], center[0]+offsets[1,0],center[1]+offsets[1,1]
                        ,center[0]+offsets[2,0],center[1]+offsets[2,1]],fill=(1,1,1))
        img_np += np.array(img)
    img_np = img_np % 2
    if prev_mask is None:
        prev_mask = np.ones(img_np.shape)
    img_np = torch.from_numpy(img_np * prev_mask)
    return img_np


def drawtriangles(origin_triangles,coordinates, fig_size):
    coordinates = coordinates.expand(origin_triangles.shape[0],-1,-1,-1).permute(1,2,0,3)
    triangles = origin_triangles*1.5*fig_size - 0.25*fig_size
    # print(triangles)
    s = (coordinates - triangles[:,0])*(triangles[:,1]-triangles[:,0])/torch.norm(triangles[:,1]-triangles[:,0])
    s = s.sum(dim = -1)
    l1 = s.le(0)*torch.norm(coordinates - triangles[:,0],dim=-1)
    l1 += s.ge(torch.norm(triangles[:,1]-triangles[:,0]))*torch.norm(coordinates - triangles[:,1],dim=-1)
    s = torch.norm(coordinates-triangles[:,0],dim=-1)**2-(s*s)
    s = s.ge(0)*(s)
    l1 += l1.le(1e-6)*torch.sqrt(s)

    s = (coordinates - triangles[:,1])*(triangles[:,2]-triangles[:,1])/torch.norm(triangles[:,2]-triangles[:,1])
    s = s.sum(dim = -1)
    l2 = s.le(0)*torch.norm(coordinates - triangles[:,1],dim=-1)
    l2 += s.ge(torch.norm(triangles[:,2]-triangles[:,1]))*torch.norm(coordinates - triangles[:,2],dim=-1)
    s = torch.norm(coordinates-triangles[:,1],dim=-1)**2-(s*s)
    s = s.ge(0)*(s)
    l2 += l2.le(1e-6)*torch.sqrt(s)


    s = (coordinates - triangles[:,2])*(triangles[:,0]-triangles[:,2])/torch.norm(triangles[:,0]-triangles[:,2])
    s = s.sum(dim = -1)
    l3 = s.le(0)*torch.norm(coordinates - triangles[:,2],dim=-1)
    l3 += s.ge(torch.norm(triangles[:,0]-triangles[:,2]))*torch.norm(coordinates - triangles[:,0],dim=-1)
    s = torch.norm(coordinates-triangles[:,2],dim=-1)**2-(s*s)
    s = s.ge(0)*(s)
    l3 += l3.le(1e-6)*torch.sqrt(s)

    distance = torch.min(l1,l2)
    distance = torch.min(distance,l3)

    t1 = (coordinates - triangles[:,0])
    t2 = (coordinates - triangles[:,1])
    t3 = (coordinates - triangles[:,2])
    q1 = (t1[...,0]*t2[...,1] - t1[...,1]*t2[...,0])
    q2 = (t2[...,0]*t3[...,1] - t2[...,1]*t3[...,0])
    q3 = (t3[...,0]*t1[...,1] - t3[...,1]*t1[...,0])
    q = (q1*q2).ge(0)*(q1*q3).ge(0) *(q2*q3).ge(0) * 2 -1
    prob = torch.sigmoid(distance**2 * q / 3)
    # print(prob.sum())
    return prob

def xor_mask(prob_map):
    prob_map = prob_map.sum(dim=-1)%2
    prob_xor = prob_map.ge(1+1e-10) * (2 - prob_map) + prob_map * prob_map.le(1)
    # print(prob_xor.max())
    return prob_xor

def xor_mask_color(prob_map,color):
    # print(prob_map)
    prob_map = prob_map.expand(3,-1,-1,-1).permute(1,2,3,0)
    color_map = (prob_map * color).sum(dim=-2) % 2
    color_xor = color_map.ge(1+1e-10) * (2 - color_map) + color_map * color_map.le(1)
    return color_xor

def drawcircles(original_circles, coordinates, fig_size, alpha=None):
    coordinates = coordinates.expand(original_circles.shape[0],-1,-1,-1).permute(1,2,0,3)
    circles = original_circles * fig_size
    dist = torch.norm(coordinates - circles[:,:2],dim=-1)
    dist = dist - circles[:,2]
    if alpha is not None:
        prob = torch.sigmoid(dist/alpha)
    else:
        prob = torch.sigmoid(dist)
    return prob

def drawcircles_with_blur(original_circles, coordinates, fig_size):
    # origin_circles: (num_circles x 4). (cx, cy, radius, blur)
    coordinates = coordinates.expand(original_circles.shape[0],-1,-1,-1).permute(1,2,0,3)
    circles = original_circles[:,:3] * fig_size
    dist = torch.norm(coordinates - circles[:,:2],dim=-1)
    dist = dist - circles[:,2]
    # circles[:,3] \in [0,1], clip them to [0.9,1].
    dist = dist*(original_circles[:,3]+1)/2
    prob = torch.sigmoid(dist)
    return prob


def drawcircles_fix_color(original_circles, coordinates, colors, fig_size_h, fig_size_w,blur=1):
    assert original_circles.shape[0] == colors.shape[0]
    coordinates = coordinates.expand(original_circles.shape[1],-1,-1,-1).permute(1,2,0,3)
    circle0 = original_circles[...,0]*fig_size_h
    circle1 = original_circles[...,1]*fig_size_w
    circles = torch.stack([circle0,circle1],dim=-1)
    dist_sum = torch.zeros([colors.shape[0],fig_size_h,fig_size_w]).to(coordinates.device)
    for color_idx in range(colors.shape[0]):
        dist = torch.norm(coordinates-circles[color_idx,:,:2],dim=-1)
        # dist = dist / (circles[color_idx,:,2]+1)
        dist_sum[color_idx] = torch.exp(-dist/blur).sum(dim=-1)
        # print(dist_sum[color_idx])
    # print(dist_sum[0])
    dist_sum = dist_sum ** 2
    dist_sum = dist_sum/dist_sum.sum(dim=0)
    # print(dist_sum[0])
    color_map = torch.matmul(dist_sum.permute(1,2,0), colors).permute(2,0,1)
    # print(color_map)
    return color_map


def prob_fix_color(original_circles, coordinates, colors, fig_size_h, fig_size_w,blur=1):
    assert original_circles.shape[0] == colors.shape[0]
    coordinates = coordinates.expand(original_circles.shape[1],-1,-1,-1).permute(1,2,0,3)
    # circles = original_circles * fig_size_h
    circle0 = original_circles[...,0]*fig_size_h
    circle1 = original_circles[...,1]*fig_size_w
    circles = torch.stack([circle0,circle1],dim=-1)
    dist_sum = torch.zeros([colors.shape[0],fig_size_h,fig_size_w]).to(coordinates.device)
    for color_idx in range(colors.shape[0]):
        dist = torch.norm(coordinates-circles[color_idx,:,:2],dim=-1)
        # dist = torch.norm(coordinates-circles[color_idx,:,:2],dim=-1)
        # dist = dist / (circles[color_idx,:,2]+1)
        dist_sum[color_idx] = torch.exp(-dist/blur).sum(dim=-1)
        # print(dist_sum[color_idx])
    # print(dist_sum[0])
    dist_sum = dist_sum/dist_sum.sum(dim=0)
    return dist_sum

def gumbel_color_fix_seed(prob_map, seed, color, tau=0.3, type='gumbel'):
    # print(prob_map.shape, seed.shape, color.shape)
    if type == 'gumbel':
        color_map = F.softmax((torch.log(prob_map) + seed)/tau, dim=-1)
    elif type == 'determinate':
        color_ind = (torch.log(prob_map) + seed).max(-1)[1]
        color_map = F.one_hot(color_ind, prob_map.shape[-1]).to(prob_map)
    else:
        raise ValueError
    tex = torch.matmul(color_map, color).unsqueeze(0)
    return tex


def ctrl_loss(circles, fig_h, fig_w, sigma=40):
    circles = circles.repeat(circles.shape[1],1,1,1).permute(1,0,2,3)
    diff = circles - circles.permute(0,2,1,3)
    diff_ell2 = (diff[...,0] * diff[...,0]*fig_h*fig_h + diff[...,1] * diff[..., 1]*fig_w*fig_w)
    loss_c = torch.exp(-diff_ell2/(sigma**2)).mean() - 1/circles.shape[1]
    return loss_c
        