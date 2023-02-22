# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generator architecture from the paper
"Alias-Free Generative Adversarial Networks"."""

import numpy as np
import scipy.signal
import scipy.optimize
import torch
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import filtered_lrelu
from torch_utils.ops import bias_act
from training.appended_net_v2 import AppendedNet
from training.feature_classifier import FeatureClassifier

from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from math import log2
import json

# from torchvision.transforms.functional import gaussian_blur
from torchvision.transforms.functional import affine
from kornia.morphology import dilation, erosion

from torch_utils.ops import upfirdn2d


#----------------------------------------------------------------------------

@misc.profiled_function
def modulated_conv2d(
    x,                  # Input tensor: [batch_size, in_channels, in_height, in_width]
    w,                  # Weight tensor: [out_channels, in_channels, kernel_height, kernel_width]
    s,                  # Style tensor: [batch_size, in_channels]
    demodulate  = True, # Apply weight demodulation?
    padding     = 0,    # Padding: int or [padH, padW]
    input_gain  = None, # Optional scale factors for the input channels: [], [in_channels], or [batch_size, in_channels]
):
    with misc.suppress_tracer_warnings(): # this value will be treated as a constant
        batch_size = int(x.shape[0])
    out_channels, in_channels, kh, kw = w.shape
    misc.assert_shape(w, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(s, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs.
    if demodulate:
        w = w * w.square().mean([1,2,3], keepdim=True).rsqrt()
        s = s * s.square().mean().rsqrt()

    # Modulate weights.
    w = w.unsqueeze(0) # [NOIkk]
    w = w * s.unsqueeze(1).unsqueeze(3).unsqueeze(4) # [NOIkk]

    # Demodulate weights.
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
        w = w * dcoefs.unsqueeze(2).unsqueeze(3).unsqueeze(4) # [NOIkk]

    # Apply input scaling.
    if input_gain is not None:
        input_gain = input_gain.expand(batch_size, in_channels) # [NI]
        w = w * input_gain.unsqueeze(1).unsqueeze(3).unsqueeze(4) # [NOIkk]

    # Execute as one fused op using grouped convolution.
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_gradfix.conv2d(input=x, weight=w.to(x.dtype), padding=padding, groups=batch_size)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        bias            = True,     # Apply additive bias before the activation function?
        lr_multiplier   = 1,        # Learning rate multiplier.
        weight_init     = 1,        # Initial standard deviation of the weight tensor.
        bias_init       = 0,        # Initial value of the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) * (weight_init / lr_multiplier))
        bias_init = np.broadcast_to(np.asarray(bias_init, dtype=np.float32), [out_features])
        self.bias = torch.nn.Parameter(torch.from_numpy(bias_init / lr_multiplier)) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain
        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no labels.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output.
        num_layers      = 2,        # Number of mapping layers.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.998,    # Decay for tracking the moving average of W during training.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        # Construct layers.
        self.embed = FullyConnectedLayer(self.c_dim, self.w_dim) if self.c_dim > 0 else None
        features = [self.z_dim + (self.w_dim if self.c_dim > 0 else 0)] + [self.w_dim] * self.num_layers
        for idx, in_features, out_features in zip(range(num_layers), features[:-1], features[1:]):
            layer = FullyConnectedLayer(in_features, out_features, activation='lrelu', lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)
        self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        misc.assert_shape(z, [None, self.z_dim])
        if truncation_cutoff is None:
            truncation_cutoff = self.num_ws

        # Embed, normalize, and concatenate inputs.
        x = z.to(torch.float32)
        x = x * (x.square().mean(1, keepdim=True) + 1e-8).rsqrt()
        if self.c_dim > 0:
            misc.assert_shape(c, [None, self.c_dim])
            y = self.embed(c.to(torch.float32))
            y = y * (y.square().mean(1, keepdim=True) + 1e-8).rsqrt()
            x = torch.cat([x, y], dim=1) if x is not None else y

        # Execute layers.
        for idx in range(self.num_layers):
            x = getattr(self, f'fc{idx}')(x)

        # Update moving average of W.
        if update_emas:
            self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast and apply truncation.
        x = x.unsqueeze(1).repeat([1, self.num_ws, 1])
        if truncation_psi != 1:
            x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

    def extra_repr(self):
        return f'z_dim={self.z_dim:d}, c_dim={self.c_dim:d}, w_dim={self.w_dim:d}, num_ws={self.num_ws:d}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisInput(torch.nn.Module):
    def __init__(self,
        w_dim,          # Intermediate latent (W) dimensionality.
        channels,       # Number of output channels.
        size,           # Output spatial size: int or [width, height].
        sampling_rate,  # Output sampling rate.
        bandwidth,      # Output bandwidth.
    ):
        super().__init__()
        self.w_dim = w_dim
        self.channels = channels
        self.size = np.broadcast_to(np.asarray(size), [2])
        self.sampling_rate = sampling_rate
        self.bandwidth = bandwidth

        # Draw random frequencies from uniform 2D disc.
        freqs = torch.randn([self.channels, 2])
        radii = freqs.square().sum(dim=1, keepdim=True).sqrt()
        freqs /= radii * radii.square().exp().pow(0.25)
        freqs *= bandwidth
        phases = torch.rand([self.channels]) - 0.5

        # Setup parameters and buffers.
        self.weight = torch.nn.Parameter(torch.randn([self.channels, self.channels]))
        self.affine = FullyConnectedLayer(w_dim, 4, weight_init=0, bias_init=[1,0,0,0])
        self.register_buffer('transform', torch.eye(3, 3)) # User-specified inverse transform wrt. resulting image.
        self.register_buffer('freqs', freqs)
        self.register_buffer('phases', phases)

    def forward(self, w):
        # Introduce batch dimension.
        transforms = self.transform.unsqueeze(0) # [batch, row, col]
        freqs = self.freqs.unsqueeze(0) # [batch, channel, xy]
        phases = self.phases.unsqueeze(0) # [batch, channel]

        # Apply learned transformation.
        t = self.affine(w) # t = (r_c, r_s, t_x, t_y)
        t = t / t[:, :2].norm(dim=1, keepdim=True) # t' = (r'_c, r'_s, t'_x, t'_y)
        m_r = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1]) # Inverse rotation wrt. resulting image.
        m_r[:, 0, 0] = t[:, 0]  # r'_c
        m_r[:, 0, 1] = -t[:, 1] # r'_s
        m_r[:, 1, 0] = t[:, 1]  # r'_s
        m_r[:, 1, 1] = t[:, 0]  # r'_c
        m_t = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1]) # Inverse translation wrt. resulting image.
        m_t[:, 0, 2] = -t[:, 2] # t'_x
        m_t[:, 1, 2] = -t[:, 3] # t'_y
        transforms = m_r @ m_t @ transforms # First rotate resulting image, then translate, and finally apply user-specified transform.

        # Transform frequencies.
        phases = phases + (freqs @ transforms[:, :2, 2:]).squeeze(2)
        freqs = freqs @ transforms[:, :2, :2]

        # Dampen out-of-band frequencies that may occur due to the user-specified transform.
        amplitudes = (1 - (freqs.norm(dim=2) - self.bandwidth) / (self.sampling_rate / 2 - self.bandwidth)).clamp(0, 1)

        # Construct sampling grid.
        theta = torch.eye(2, 3, device=w.device)
        theta[0, 0] = 0.5 * self.size[0] / self.sampling_rate
        theta[1, 1] = 0.5 * self.size[1] / self.sampling_rate
        grids = torch.nn.functional.affine_grid(theta.unsqueeze(0), [1, 1, self.size[1], self.size[0]], align_corners=False)

        # Compute Fourier features.
        x = (grids.unsqueeze(3) @ freqs.permute(0, 2, 1).unsqueeze(1).unsqueeze(2)).squeeze(3) # [batch, height, width, channel]
        x = x + phases.unsqueeze(1).unsqueeze(2)
        x = torch.sin(x * (np.pi * 2))
        x = x * amplitudes.unsqueeze(1).unsqueeze(2)

        # Apply trainable mapping.
        weight = self.weight / np.sqrt(self.channels)
        x = x @ weight.t()

        # Ensure correct shape.
        x = x.permute(0, 3, 1, 2) # [batch, channel, height, width]
        misc.assert_shape(x, [w.shape[0], self.channels, int(self.size[1]), int(self.size[0])])
        return x

    def extra_repr(self):
        return '\n'.join([
            f'w_dim={self.w_dim:d}, channels={self.channels:d}, size={list(self.size)},',
            f'sampling_rate={self.sampling_rate:g}, bandwidth={self.bandwidth:g}'])


@persistence.persistent_class
class CondSynthesisInput(torch.nn.Module):
    def __init__(self,
        w_dim,          # Intermediate latent (W) dimensionality.
        channels,       # Number of output channels.
        in_size,
        out_size            # Output spatial size: int or [width, height].
    ):
        super().__init__()
        self.w_dim = w_dim
        self.conv = torch.nn.Conv2d(channels,channels,1,padding=0)
        self.padding = int((out_size - in_size)/2)
    
    def forward(self,x):
        x = self.conv(x)
        x = torch.nn.functional.pad(x,(self.padding,self.padding,self.padding,self.padding), mode='reflect')
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        w_dim,                          # Intermediate latent (W) dimensionality.
        is_torgb,                       # Is this the final ToRGB layer?
        is_critically_sampled,          # Does this layer use critical sampling?
        use_fp16,                       # Does this layer use FP16?

        # Input & output specifications.
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        in_size,                        # Input spatial size: int or [width, height].
        out_size,                       # Output spatial size: int or [width, height].
        in_sampling_rate,               # Input sampling rate (s).
        out_sampling_rate,              # Output sampling rate (s).
        in_cutoff,                      # Input cutoff frequency (f_c).
        out_cutoff,                     # Output cutoff frequency (f_c).
        in_half_width,                  # Input transition band half-width (f_h).
        out_half_width,                 # Output Transition band half-width (f_h).

        # Hyperparameters.
        conv_kernel         = 3,        # Convolution kernel size. Ignored for final the ToRGB layer.
        filter_size         = 6,        # Low-pass filter size relative to the lower resolution when up/downsampling.
        lrelu_upsampling    = 2,        # Relative sampling rate for leaky ReLU. Ignored for final the ToRGB layer.
        use_radial_filters  = False,    # Use radially symmetric downsampling filter? Ignored for critically sampled layers.
        conv_clamp          = 256,      # Clamp the output to [-X, +X], None = disable clamping.
        magnitude_ema_beta  = 0.999,    # Decay rate for the moving average of input magnitudes.
    ):
        super().__init__()
        self.w_dim = w_dim
        self.is_torgb = is_torgb
        self.is_critically_sampled = is_critically_sampled
        self.use_fp16 = use_fp16
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = np.broadcast_to(np.asarray(in_size), [2])
        self.out_size = np.broadcast_to(np.asarray(out_size), [2])
        self.in_sampling_rate = in_sampling_rate
        self.out_sampling_rate = out_sampling_rate
        self.tmp_sampling_rate = max(in_sampling_rate, out_sampling_rate) * (1 if is_torgb else lrelu_upsampling)
        self.in_cutoff = in_cutoff
        self.out_cutoff = out_cutoff
        self.in_half_width = in_half_width
        self.out_half_width = out_half_width
        self.conv_kernel = 1 if is_torgb else conv_kernel
        self.conv_clamp = conv_clamp
        self.magnitude_ema_beta = magnitude_ema_beta

        # Setup parameters and buffers.
        self.affine = FullyConnectedLayer(self.w_dim, self.in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(torch.randn([self.out_channels, self.in_channels, self.conv_kernel, self.conv_kernel]))
        self.bias = torch.nn.Parameter(torch.zeros([self.out_channels]))
        self.register_buffer('magnitude_ema', torch.ones([]))

        # Design upsampling filter.
        self.up_factor = int(np.rint(self.tmp_sampling_rate / self.in_sampling_rate))
        assert self.in_sampling_rate * self.up_factor == self.tmp_sampling_rate
        self.up_taps = filter_size * self.up_factor if self.up_factor > 1 and not self.is_torgb else 1
        self.register_buffer('up_filter', self.design_lowpass_filter(
            numtaps=self.up_taps, cutoff=self.in_cutoff, width=self.in_half_width*2, fs=self.tmp_sampling_rate))

        # Design downsampling filter.
        self.down_factor = int(np.rint(self.tmp_sampling_rate / self.out_sampling_rate))
        assert self.out_sampling_rate * self.down_factor == self.tmp_sampling_rate
        self.down_taps = filter_size * self.down_factor if self.down_factor > 1 and not self.is_torgb else 1
        self.down_radial = use_radial_filters and not self.is_critically_sampled
        self.register_buffer('down_filter', self.design_lowpass_filter(
            numtaps=self.down_taps, cutoff=self.out_cutoff, width=self.out_half_width*2, fs=self.tmp_sampling_rate, radial=self.down_radial))

        # Compute padding.
        pad_total = (self.out_size - 1) * self.down_factor + 1 # Desired output size before downsampling.
        pad_total -= (self.in_size + self.conv_kernel - 1) * self.up_factor # Input size after upsampling.
        pad_total += self.up_taps + self.down_taps - 2 # Size reduction caused by the filters.
        pad_lo = (pad_total + self.up_factor) // 2 # Shift sample locations according to the symmetric interpretation (Appendix C.3).
        pad_hi = pad_total - pad_lo
        self.pad_hi = int(pad_hi[0])
        self.padding = [int(pad_lo[0]), int(pad_hi[0]), int(pad_lo[1]), int(pad_hi[1])]
        self.device = torch.device('cuda')

        self.cla_classes = out_channels
        self.cla_in_res = self.out_size[0]-20
        self.cla_num_layers = int(log2(self.cla_in_res)-3)
    
    def set_cluster(self, c):
      self.cluster = c.to(self.device)

    def forward(self, x, w, noise_mode='random', force_fp32=False, update_emas=False, op = None):
        assert noise_mode in ['random', 'const', 'none'] # unused
        misc.assert_shape(x, [None, self.in_channels, int(self.in_size[1]), int(self.in_size[0])])
        misc.assert_shape(w, [x.shape[0], self.w_dim])

        # Track input magnitude.
        if update_emas:
            with torch.autograd.profiler.record_function('update_magnitude_ema'):
                magnitude_cur = x.detach().to(torch.float32).square().mean()
                self.magnitude_ema.copy_(magnitude_cur.lerp(self.magnitude_ema, self.magnitude_ema_beta))
        input_gain = self.magnitude_ema.rsqrt()

        # Execute affine layer.
        styles = self.affine(w)
        if self.is_torgb:
            weight_gain = 1 / np.sqrt(self.in_channels * (self.conv_kernel ** 2))
            styles = styles * weight_gain

        # Execute modulated conv2d.
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
        x = modulated_conv2d(x=x.to(dtype), w=self.weight, s=styles,
            padding=self.conv_kernel-1, demodulate=(not self.is_torgb), input_gain=input_gain)

        # Execute bias, filtered leaky ReLU, and clamping.
        gain = 1 if self.is_torgb else np.sqrt(2)
        slope = 1 if self.is_torgb else 0.2
        x = filtered_lrelu.filtered_lrelu(x=x, fu=self.up_filter, fd=self.down_filter, b=self.bias.to(x.dtype),
            up=self.up_factor, down=self.down_factor, padding=self.padding, gain=gain, slope=slope, clamp=self.conv_clamp)

        ######## op

        if op is not None:
          # print(f'in: {x.shape}')
          x_d = affine(x, angle=op["angle"], translate=op['translate'], scale=op['scale'], shear=0)
          x_d = erosion(x_d, torch.ones((op["erosion"], op["erosion"]),dtype=dtype).to(self.device)) if op["erosion"] else x_d
          x_d = dilation(x_d, torch.ones((op["dilation"], op["dilation"]),dtype=dtype).to(self.device)) if op["dilation"] else x_d
          
          if op['cluster'] != -1:
            sl = self.cluster == op['cluster']
            sl = sl.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat([1,1,x.shape[2],x.shape[3]])
            x_d = torch.where(sl,x_d,x)
            # print(self.cluster.shape)

            # x_d = x[:,self.cluster == op['cluster'],:,:]

          x = x_d

        # Ensure correct shape and dtype.
        # misc.assert_shape(x, [None, self.out_channels, int(self.out_size[1]), int(self.out_size[0])])
        assert x.dtype == dtype, 'dtype error'
        return x

    @staticmethod
    def design_lowpass_filter(numtaps, cutoff, width, fs, radial=False):
        assert numtaps >= 1

        # Identity filter.
        if numtaps == 1:
            return None

        # Separable Kaiser low-pass filter.
        if not radial:
            f = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)
            return torch.as_tensor(f, dtype=torch.float32)

        # Radially symmetric jinc-based filter.
        x = (np.arange(numtaps) - (numtaps - 1) / 2) / fs
        r = np.hypot(*np.meshgrid(x, x))
        f = scipy.special.j1(2 * cutoff * (np.pi * r)) / (np.pi * r)
        beta = scipy.signal.kaiser_beta(scipy.signal.kaiser_atten(numtaps, width / (fs / 2)))
        w = np.kaiser(numtaps, beta)
        f *= np.outer(w, w)
        f /= np.sum(f)
        return torch.as_tensor(f, dtype=torch.float32)

    def extra_repr(self):
        return '\n'.join([
            f'w_dim={self.w_dim:d}, is_torgb={self.is_torgb},',
            f'is_critically_sampled={self.is_critically_sampled}, use_fp16={self.use_fp16},',
            f'in_sampling_rate={self.in_sampling_rate:g}, out_sampling_rate={self.out_sampling_rate:g},',
            f'in_cutoff={self.in_cutoff:g}, out_cutoff={self.out_cutoff:g},',
            f'in_half_width={self.in_half_width:g}, out_half_width={self.out_half_width:g},',
            f'in_size={list(self.in_size)}, out_size={list(self.out_size)},',
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}'])

#----------------------------------------------------------------------------


# @persistence.persistent_class
# class SkipConcatLayer(torch.nn.Module):
#     def __init__(self,
#         use_fp16                       # Does this layer use FP16?
#     ):
#         super().__init__()
#         self.use_fp16 = use_fp16

#     def forward(self,x):

#         x = 
        
#         return x
#----------------------------------------------------------------------------
@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                          # Intermediate latent (W) dimensionality.
        img_resolution,                 # Output image resolution.
        img_channels,                   # Number of color channels.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_layers          = 14,       # Total number of layers, excluding Fourier features and ToRGB.
        num_critical        = 2,        # Number of critically sampled layers at the end.
        first_cutoff        = 2,        # Cutoff frequency of the first layer (f_{c,0}).
        first_stopband      = 2**2.1,   # Minimum stopband of the first layer (f_{t,0}).
        last_stopband_rel   = 2**0.3,   # Minimum stopband of the last layer, expressed relative to the cutoff.
        margin_size         = 10,       # Number of additional pixels outside the image.
        output_scale        = 0.25,     # Scale factor for the output image.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.

        # skip_channels_idx   = [0, 3, 6, 7, 10],
        # skip_connection     = [0, 0, 0, 0,  0],
        encoder_receive     = [],
        encoder_receive_c   = [],
        bottleneck_size     = 16,
        encode_rgb          = 1,
        blur_sigma          = 3,
        **layer_kwargs                 # Arguments for SynthesisLayer.
    ):
        super().__init__()
        self.w_dim = w_dim
        self.num_ws = num_layers + 2
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.num_layers = num_layers
        self.num_critical = num_critical
        self.margin_size = margin_size
        self.output_scale = output_scale
        self.num_fp16_res = num_fp16_res
        self.encode_rgb = encode_rgb
        # self.skip_channels_idx = skip_channels_idx
        # self.skip_connection = skip_connection

        # Geometric progression of layer cutoffs and min. stopbands.
        last_cutoff = self.img_resolution / 2 # f_{c,N}
        last_stopband = last_cutoff * last_stopband_rel # f_{t,N}
        exponents = np.minimum(np.arange(self.num_layers + 1) / (self.num_layers - self.num_critical), 1)
        cutoffs = first_cutoff * (last_cutoff / first_cutoff) ** exponents # f_c[i]
        stopbands = first_stopband * (last_stopband / first_stopband) ** exponents # f_t[i]

        # Compute remaining layer parameters.
        sampling_rates = np.exp2(np.ceil(np.log2(np.minimum(stopbands * 2, self.img_resolution)))) # s[i]
        half_widths = np.maximum(stopbands, sampling_rates / 2) - cutoffs # f_h[i]
        sizes = sampling_rates + self.margin_size * 2
        sizes[-2:] = self.img_resolution
        self.sizes = sizes
        channels = np.rint(np.minimum((channel_base / 2) / cutoffs, channel_max))
        channels[-1] = self.img_channels
        self.channels = channels

        # assert len(self.skip_down_channels) == len(self.skip_down_sizes), f'whatt???? \n{self.skip_down_channels}\n{self.skip_down_sizes}'
        print('hiiii generator')
        print(encoder_receive)
        print(encoder_receive_c)

        # Construct layers.
        # self.input = SynthesisInput(
        #     w_dim=self.w_dim, channels=int(channels[0]), size=int(sizes[0]),
        #     sampling_rate=sampling_rates[0], bandwidth=cutoffs[0])
        self.input = CondSynthesisInput(w_dim=self.w_dim, 
                                        channels=int(channels[0]),
                                        in_size=bottleneck_size,
                                        out_size = int(sizes[0]))
        self.layer_names = []
        self.layer_fp16 = []

        assert len(encoder_receive) == self.num_layers + 1
        assert len(encoder_receive_c) == self.num_layers + 1
        encoder_receive.reverse()
        encoder_receive_c.reverse()
        self.encoder_receive = encoder_receive
        
        self.device = torch.device('cuda')
        self.blur_sigma = blur_sigma
        self.blur_size = np.floor(self.blur_sigma * 3)
        f = torch.arange(-self.blur_size , self.blur_size  + 1, device=self.device).div(self.blur_sigma).square().neg().exp2()
        self.register_buffer('filter', f)
        
        for idx in range(self.num_layers + 1):
            prev = max(idx - 1, 0)
            is_torgb = (idx == self.num_layers)
            is_critically_sampled = (idx >= self.num_layers - self.num_critical)
            use_fp16 = (sampling_rates[idx] * (2 ** self.num_fp16_res) > self.img_resolution)
            self.layer_fp16.append(use_fp16)

            c_in = int(channels[prev]) + encoder_receive_c[idx]


            layer = SynthesisLayer(
                w_dim=self.w_dim, is_torgb=is_torgb, is_critically_sampled=is_critically_sampled, use_fp16=use_fp16,
                in_channels=c_in, out_channels= int(channels[idx]),
                in_size=int(sizes[prev]), out_size=int(sizes[idx]),
                in_sampling_rate=int(sampling_rates[prev]), out_sampling_rate=int(sampling_rates[idx]),
                in_cutoff=cutoffs[prev], out_cutoff=cutoffs[idx],
                in_half_width=half_widths[prev], out_half_width=half_widths[idx],
                **layer_kwargs)
            name = f'L{idx}_{layer.out_size[0]}_{layer.out_channels}'
            setattr(self, name, layer)
            self.layer_names.append(name)

    def test_func(self):
      print('hii')     

    def generate_operation_template(self, f_path, start = 4, end = 5):
        op = {}
        for idx, name in enumerate(self.layer_names):
            if idx >= start and idx<=end:
                op[name] = {
                    "erosion": 0,
                    "dilation": 0,
                    "angle": 0,
                    "scale": 1,
                    "translate": [0,0],
                    "cluster": -1
                }
        with open(f'{f_path}/operation_template.json', 'w') as file:
            json.dump(op, file,indent=4)

        return op

    def forward(self, replaced_w, skips = None, operations = {}, **layer_kwargs):
        
        misc.assert_shape(replaced_w, [None, self.num_ws, self.w_dim])
        # ws = ws.to(torch.float32).unbind(dim=1)
        replaced_w = replaced_w.to(torch.float32).unbind(dim=1)
        
        # Execute layers.
        # x = self.input(ws[0]) if len(replaced_w) == 0 else self.input(replaced_w[0])
        x = upfirdn2d.filter2d(skips[0], self.filter / self.filter.sum()) if self.blur_sigma else skips[0]
        x = self.input(x)
        
        
        if skips is None:
          assert False, 'missing skip inputs'

        # print(len(self.layer_names))
        # print(len(ws[1:]))
        # print(len(skips))
        for idx, (name, w, s) in enumerate(zip(self.layer_names, replaced_w[1:], self.encoder_receive)):
            # print(name)
            # print(x.shape)
            layer = getattr(self, name)
            if s:
              # print(f'x: {x.shape}')
              # print(f's: {skip.shape}')

              concat = torch.nn.functional.pad(skips[s-1],(10,10,10,10), mode='reflect')
            
              concat = upfirdn2d.filter2d(concat, self.filter / self.filter.sum()) if self.blur_sigma else concat
              x = torch.cat([x,concat],dim=1)

            op = operations[name] if name in operations.keys() else None
            x = layer(x, w, op=op, **layer_kwargs)


        if self.output_scale != 1:
            x = x * self.output_scale

        # Ensure correct shape and dtype.
        # misc.assert_shape(x, [None, self.img_channels, self.img_resolution, self.img_resolution])
        x = x.to(torch.float32)
        return x

    def set_receive(self, r):
      self.encoder_receive = r

    def extra_repr(self):
        return '\n'.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'num_layers={self.num_layers:d}, num_critical={self.num_critical:d},',
            f'margin_size={self.margin_size:d}, num_fp16_res={self.num_fp16_res:d}'])


#----------------------------------------------------------------------------

@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        projecting_img_dim  = (3,256,256),

        # g_channels_res     = [256, 256, 256, 256, 256, 128, 128, 128, 64, 64, 32, 32, 16, 16, 16],
        # encoder_out_res    = [128, 64, 64, 32, 32, 16, 16,  16,   8,   4],
        # encoder_channels   = [ 64,128,128,256,256,512,512, 512,1024,1024],
        # encoder_connect_to = [  0,  0,  5,  0,  4,  3,  2,   1,   0,   0],
        # encoder_receive    = [  0,   0,   0,   0,   0,   0,   0,   0,  5,  0,  4,  0,  3,  0,   2],
        # encoder_receive_c  = [  0,   0,   0,   0,   0,   0,   0,   0,256,  0,512,  0,512,  0, 512],

        g_channels_res     = [256, 256, 256, 256, 256, 128, 128, 128, 64, 64, 32, 32, 16, 16, 16],
        encoder_out_res    = [128, 64, 64, 32, 32, 32, 16, 16,  16],
        encoder_channels   = [ 64,128,128,256,256,256,512,512, 512],
        encoder_connect_to = [  0,  0,  0,  0,  0,  4,  3,  2,   1],
        encoder_receive    = [  0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  4,  0,  3,  0,   2],
        encoder_receive_c  = [  0,   0,   0,   0,   0,   0,   0,   0,  0,  0,512,  0,512,  0, 512],

        bottleneck_size    = 16,
        connection_start        = 0,
        connection_end          = 11,
        connection_grow_from    = 4,
        num_appended_ws         = 6,
        encode_rgb              = True,
        blur_sigma              = 3,
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        # skip_channels_idx = np.arange(connection_start,connection_end,1,dtype=int).tolist()
        # skip_connection = np.zeros(connection_end - connection_start, dtype=int)
        # skip_connection[:connection_grow_from] = 1
        # skip_connection = skip_connection.tolist()
        # print(skip_channels_idx)
        # print(skip_connection)
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.p_dim = projecting_img_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        assert len(projecting_img_dim)==3, f'projecting_img_dim takes image dim as tuple like (3, 256, 256)'
        self.p_size = projecting_img_dim[1]
        self.p_channel = projecting_img_dim[0]
        self.synthesis = SynthesisNetwork(w_dim=w_dim, 
                                          img_resolution=img_resolution, 
                                          img_channels=img_channels, 
                                          # skip_channels_idx=skip_channels_idx, 
                                          # skip_connection=skip_connection, 
                                          encoder_receive = encoder_receive,
                                          encoder_receive_c = encoder_receive_c,
                                          bottleneck_size = bottleneck_size,
                                          encode_rgb = encode_rgb,
                                          blur_sigma = blur_sigma,
                                          **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        # self.mapping = MappingNetwork(z_dim=z_dim, 
        #                               c_dim=c_dim, 
        #                               w_dim=w_dim, 
        #                               num_ws=self.num_ws, 
        #                               **mapping_kwargs)
        self.appended_net = AppendedNet(self.synthesis.channels, 
                                        self.synthesis.sizes, 
                                        self.p_dim, 
                                        encoder_out_res, 
                                        encoder_channels, 
                                        encoder_connect_to, 
                                        self.synthesis.layer_fp16, 
                                        num_appended_ws)


    def forward(self, z, c, skips_in, truncation_psi=1, truncation_cutoff=None, update_emas=False, operations = {}, **synthesis_kwargs):
        # print(skips_in.shape)
        skips_out, replaced_w = self.appended_net(skips_in)
        # print(len(skips_out))
        
        skips_out.reverse()

        # ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        
        img = self.synthesis(replaced_w, skips = skips_out, update_emas=update_emas, operations = operations, **synthesis_kwargs)
        return img

    @torch.no_grad()
    def generate_cluster_for_layer(self, layer_idx, classifier_bottleneck, classifier_path, skips_in, num_clusters = 5):
        device = torch.device('cuda')
        z = np.random.randn(1, 512)
        z = torch.from_numpy(z).to(device)

        skips_out, replaced_w = self.appended_net(skips_in)

        skips_out.reverse()
        ws = self.mapping(z, None, truncation_psi=1, truncation_cutoff=None, update_emas=False)

        ws = ws.to(torch.float32).unbind(dim=1)
        replaced_w = replaced_w.to(torch.float32).unbind(dim=1)
        x = upfirdn2d.filter2d(skips_out[0], self.synthesis.filter / self.synthesis.filter.sum()) if self.synthesis.blur_sigma else skips_out[0]
        x = self.synthesis.input(x)
        for idx, (name, w, s) in enumerate(zip(self.synthesis.layer_names, ws[1:], self.synthesis.encoder_receive)):
            layer = getattr(self.synthesis, name)
            if s:
              concat = torch.nn.functional.pad(skips_out[s-1],(10,10,10,10), mode='reflect')
              x = torch.cat([x,concat],dim=1)
            if layer.is_torgb and self.synthesis.encode_rgb:
              x = layer(x, replaced_w[-1])
            elif idx < len(replaced_w[1:]):

              x = layer(x, replaced_w[1:][idx])
            else:
              x = layer(x, w)
            
            if idx == layer_idx:
              print(f'clustering {name}')
              
              x = torch.nn.functional.pad(x.detach(),[-10,-10,-10,-10])
              x = x.to(torch.float32).add(1).div(4).clamp(0, 1)
              c = x[0].unsqueeze(1).mul(2).add(-1)
              classifier = FeatureClassifier(layer.cla_num_layers,
                                            layer.cla_classes,
                                            layer.cla_in_res,
                                            classifier_bottleneck).to(device)
              classifier_state_dict = torch.load(classifier_path)
              classifier.load_state_dict(classifier_state_dict)
              classifier.to(device)
              vec, _ = classifier(c)
              print(vec.shape)
              print(c.shape)
              cluster_ids_x = classifier.get_cluster(vec, num_clusters)
              
              layer.set_cluster(cluster_ids_x)
              return self.grid_clusters(x, cluster_ids_x.to(device), num_clusters, img_per_row=12)

    
    @torch.no_grad()
    def grid_clusters(self, x_show, cluster_ids_x, num_clusters, img_per_row=15):
        grids = []
        for i in range(num_clusters):
          x_select = x_show[:,cluster_ids_x == i,:,:]
          g = make_grid(x_select[0,:img_per_row].unsqueeze(1).to(torch.float32), nrow = img_per_row).unsqueeze(0)
          s = (50,img_per_row*50)
          g = torch.nn.functional.interpolate(g, size=s, mode='bilinear')
          grids.append(g)

        grid = torch.cat(grids,dim=2)
        return to_pil_image(grid[0].cpu())




#----------------------------------------------------------------------------
