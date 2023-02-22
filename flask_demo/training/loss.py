# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

import numpy as np
from torchvision import models
import torch

from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d

from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
import PIL.Image
import os
#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

def save_image_grid(img, fname, drange, grid_size):
    
    grid = make_grid(img[0], nrow = 6)
    img = to_pil_image(grid.add(1).div(2).clamp(0, 1).cpu())
    img.save(fname)
#     gw, gh = grid_size
#     _N, C, H, W = img.shape
#     img = img.reshape([gh, gw, C, H, W])
#     img = img.transpose(0, 3, 1, 4, 2)
#     img = img.reshape([gh * H, gw * W, C])

#     assert C in [1, 3]
#     if C == 1:
#         PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
#     if C == 3:
#         PIL.Image.fromarray(img, 'RGB').save(fname)
#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        # self.vgg_loss           = VGGLoss()

    def run_G(self, z, c, cond_img = None, update_emas=False):
        # skips_out = reversed(self.G.appended_net(cond_img)) if cond_img is not None else None
        skips_out, replaced_w = self.G.appended_net(cond_img)
        skips_out.reverse()

        # ws = self.G.mapping(z, c, update_emas=update_emas)
        # if self.style_mixing_prob > 0:
        #     with torch.autograd.profiler.record_function('style_mixing'):
        #         cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
        #         cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
        #         ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        # img = self.G.synthesis(ws, skips = skips_out, update_emas=update_emas)
        img = self.G.synthesis(replaced_w, skips = skips_out, update_emas=update_emas)
        
        return img, replaced_w

    def run_D(self, img, c, blur_sigma=0, update_emas=False, grid_size = None):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        if not  grid_size is None:
            save_image_grid(img.detach(), os.path.join('.', 'blur.png'), drange=[-1,1], grid_size=grid_size)
        logits = self.D(img, c, update_emas=update_emas)
        return logits
    
    def run_mse(self, gen, real, blur_sigma):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            f = torch.arange(-blur_size, blur_size + 1, device=gen.device).div(blur_sigma).square().neg().exp2()
            gen = upfirdn2d.filter2d(gen, f / f.sum())
            real = upfirdn2d.filter2d(real, f / f.sum())
        loss = torch.nn.functional.mse_loss(gen, real)
        return loss

    def run_vgg_loss(self, gen, real, blur_sigma):
      blur_size = np.floor(blur_sigma * 3)
      if blur_size > 0:
          f = torch.arange(-blur_size, blur_size + 1, device=gen.device).div(blur_sigma).square().neg().exp2()
          gen = upfirdn2d.filter2d(gen, f / f.sum())
          real = upfirdn2d.filter2d(real, f / f.sum())
      loss = self.vgg_loss(gen, real)
      return loss
    
    @torch.no_grad()
    def freeze_for_affine(self):
      self.D.requires_grad_(False)
      for n,p in self.G.named_parameters():
        if n=='synthesis.input.affine.weight' or n=='synthesis.input.affine.bias':
          pass
        else:
          p.requires_grad_(False)


    def accumulate_gradients(self, phase, real_img, cond_img, real_c, gen_z, gen_c, gain, cur_nimg, mute = True, grid_size = None, train_affine = False, use_vgg=False, gan_factor=0.6, target_factor=0.8, d_factor = 1.3):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if train_affine:
          self.freeze_for_affine()
        
        if self.pl_weight == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                real_img_tmp = real_img.detach().requires_grad_(phase in ['Gmain', 'Gboth']) ############## hi
                con_img_temp = cond_img.detach().requires_grad_(phase in ['Gmain', 'Gboth'])
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, con_img_temp)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits).mean() # -log(sigmoid(gen_logits))

                # loss_Gtarget = torch.nn.functional.mse_loss(gen_img, real_img_tmp)
                if use_vgg:
                  loss_Gtarget = self.run_vgg_loss(gen_img, real_img_tmp, blur_sigma=blur_sigma) * 0.2
                else:
                  loss_Gtarget = self.run_mse(gen_img, real_img_tmp, blur_sigma=blur_sigma)
                
                if train_affine:
                    loss_Gnet = loss_Gtarget
                    if not mute:
                        print(f'target_err: {loss_Gtarget.item():.6}')
                else:
                    loss_Gnet = loss_Gmain*gan_factor + loss_Gtarget*target_factor
                # loss_Gnet = (loss_Gmain * 0.1 + loss_Gtarget * 2.2).clamp(0,3)
                # loss_Gnet = (loss_Gmain * 0.6 + loss_Gtarget * 0.3).clamp(0,3)
                # loss_Gnet = (loss_Gmain * 0.8 + loss_Gtarget * 0.2).clamp(0,3)
                
                
                
                
                    if not mute:
                      if use_vgg:
                        print(f'net_err: {loss_Gnet.item():.6}\t main_err: {loss_Gmain.item():.6}\t vgg_err: {loss_Gtarget.item():.6}')
                      else:
                        print(f'net_err: {loss_Gnet.item():.6}\t main_err: {loss_Gmain.item():.6}\t target_err: {loss_Gtarget.item():.6}')

                training_stats.report('Loss/G/loss', loss_Gnet)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gnet.mul(gain).backward()

        # Gpl: Apply path length regularization.
        if phase in ['Greg', 'Gboth']:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                con_img_temp = cond_img.detach().requires_grad_(phase in ['Greg', 'Gboth'])[:batch_size]
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], con_img_temp) ###### ?????
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients(self.pl_no_weight_grad):
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                loss_Gpl.mean().mul(gain).backward()
                
        if not train_affine:
            # Dmain: Minimize logits for generated images.
            loss_Dgen = 0
            if phase in ['Dmain', 'Dboth']:
                with torch.autograd.profiler.record_function('Dgen_forward'):
                    cond_img_temp = cond_img.detach().requires_grad_(phase in ['Gmain', 'Gboth'])
                    gen_img, _gen_ws = self.run_G(gen_z, gen_c, cond_img_temp, update_emas=True)
                    gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                    training_stats.report('Loss/scores/fake', gen_logits)
                    training_stats.report('Loss/signs/fake', gen_logits.sign())
                    loss_Dgen = torch.nn.functional.softplus(gen_logits)
                    # -log(1 - sigmoid(gen_logits))
                with torch.autograd.profiler.record_function('Dgen_backward'):
                    loss_Dgen = loss_Dgen.mean().mul(gain) * d_factor
                    loss_Dgen.backward()

            # Dmain: Maximize logits for real images.
            # Dr1: Apply R1 regularization.
            if phase in ['Dmain', 'Dreg', 'Dboth']:
                name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
                with torch.autograd.profiler.record_function(name + '_forward'):
                    real_img_tmp = real_img.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                    real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                    training_stats.report('Loss/scores/real', real_logits)
                    training_stats.report('Loss/signs/real', real_logits.sign())

                    loss_Dreal = 0
                    if phase in ['Dmain', 'Dboth']:
                        loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                        training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                    loss_Dr1 = 0
                    if phase in ['Dreg', 'Dboth']:
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                        r1_penalty = r1_grads.square().sum([1,2,3])
                        loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                        training_stats.report('Loss/r1_penalty', r1_penalty)
                        training_stats.report('Loss/D/reg', loss_Dr1)

                with torch.autograd.profiler.record_function(name + '_backward'):
                    loss_D=(loss_Dreal + loss_Dr1).mean().mul(gain)
                    loss_D.backward()
                    if not mute:
                        print(f'loss_D: {loss_D.item():.6}')


#----------------------------------------------------------------------------

class VGGLoss(torch.nn.Module):
    def __init__(self, gpu_ids=0):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = torch.nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss



class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
