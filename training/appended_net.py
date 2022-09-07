from torch_utils import persistence

import torch
import numpy as np


@persistence.persistent_class
class DownConv2dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        kernel_size,                    # Width and height of the convolution kernel.
        paddings        = 10,
        bias            = True,
        activation      = 'lrelu',
        is_fp16         = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.is_fp16 = is_fp16
        self.padding = paddings
        self.weight = torch.nn.Parameter(torch.randn([self.out_channels, self.in_channels, kernel_size, kernel_size]))

        self.bias = torch.nn.Parameter(torch.zeros([self.out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act = torch.nn.LeakyReLU(0.2) if activation=='lrelu' else None
        self.register_buffer('magnitude_ema', torch.ones([]))

    
    def forward(self, x, gain=None):
      dtype = torch.float16 if self.is_fp16 else torch.float32

      w = (self.weight * self.weight_gain).to(dtype)
      b = self.bias.to(dtype)
      # print(w.shape)
      x = torch.nn.functional.conv2d(x, w, b, stride = 1, padding = self.padding)
      if self.act is not None:
        x = self.act(x)
      if gain is not None:
        x = x * gain
      # print(x.shape)
      return x

@persistence.persistent_class
class ResConvBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        mid_channels,
        out_channels,                   # Number of output channels.
        kernel_size     = 3,      
        paddings        = 0,      
        activation      = 'lrelu',
        scale_factor    = 1,
        is_fp16         = False,

    ):
        super().__init__()
        self.in_channels = in_channels
        self.is_fp16 = is_fp16
        self.paddings = paddings 
        self.conv1 = DownConv2dLayer(in_channels, mid_channels, kernel_size = kernel_size, paddings = kernel_size//2, bias = True, activation = activation, is_fp16 = is_fp16)
        self.conv2 = DownConv2dLayer(mid_channels, out_channels, kernel_size = kernel_size, paddings = kernel_size//2, bias = True, activation = activation, is_fp16 = is_fp16)

        self.scale_factor = scale_factor
        if not scale_factor == 1:
          self.pool = torch.nn.AvgPool2d(2)
        # self.skip_mapping = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, padding = 0, bias=True, dtype=torch.float16 if is_fp16 else torch.float32, padding_mode='reflect') if not in_channels == out_channels else torch.nn.Identity()
        self.skip_mapping = DownConv2dLayer(in_channels, out_channels, kernel_size=1, paddings = 0, bias=True, activation='linear', is_fp16 = is_fp16) if not in_channels == out_channels else torch.nn.Identity()


    def forward(self, x):
      dtype = torch.float16 if self.is_fp16 else torch.float32
      x = x.to(dtype)
      if not self.scale_factor == 1:
        # x = torch.nn.functional.interpolate(x, scale_factor=self.scale_facto, mode='nearest')
        x = self.pool(x)
      short_cut = self.skip_mapping(x) * np.sqrt(0.5)
      short_cut = torch.nn.functional.pad(short_cut,(self.paddings,self.paddings,self.paddings,self.paddings), mode='reflect')
      # print(f'shortcut:{short_cut.shape}')
      x = self.conv1(x)
      x = torch.nn.functional.pad(x,(self.paddings,self.paddings,self.paddings,self.paddings), mode='reflect')
      x = self.conv2(x, gain=np.sqrt(0.5))
      x = short_cut.add_(x)

      

      # pass
      return x


@persistence.persistent_class
class AppendedNet(torch.nn.Module):
  def __init__(self,
        img_channels,
        img_sizes,
        p_dim = (3,256,256),
        skip_channels_idx = [0, 3, 6, 7, 10],
        layer_fp16 = []
    ):
        super().__init__()
        self.p_dim = p_dim
        self.img_sizes = img_sizes
        self.img_channels = img_channels
        self.skip_channels_idx = skip_channels_idx
        print('hiiii appended net')
        self.device = torch.device('cuda')
        layer_fp16.reverse()
        self.layer_fp16 = layer_fp16

        self.skip_down_channels = []
        self.skip_down_sizes = []
        for idx, (c, s) in enumerate(zip(self.img_channels, self.img_sizes)):
          if idx in skip_channels_idx:
            self.skip_down_channels.append(int(c//2))
            self.skip_down_sizes.append(int(s))
          else:
            self.skip_down_channels.append(0)
            self.skip_down_sizes.append(0)

        self.skip_down_channels.reverse()
        self.skip_down_sizes.reverse()

        print(self.skip_down_channels)
        print(self.skip_down_sizes)
        assert len(self.skip_down_channels) == len(self.skip_down_sizes), f'what \n{self.skip_down_channels}\n{self.skip_down_sizes}'
        # print(p_dim)
        
        ### compute paddings
        first_size = next((x for x in self.skip_down_sizes if x), None) #276
        paddings = (first_size - p_dim[1])//4
        print(f'padding {paddings}')

        self.in_proj = DownConv2dLayer(p_dim[0], next((x for x in self.skip_down_channels if x), None), 1, paddings = 0, bias=True, activation='linear', is_fp16 = layer_fp16[0])

        self.down_names = []
        out_size = p_dim[1]


        down_channels = [c for c in self.skip_down_channels if c]
        count = 0
        for idx, (c,s,fp) in enumerate(zip(self.skip_down_channels, self.skip_down_sizes, layer_fp16)):
          if c and s:
            
            c_in = c if count == 0 else down_channels[count-1]
            c_mid = c
            c_out = c
            layer = ResConvBlock(c_in, c_mid, c_out, kernel_size=3, paddings = paddings if count>0 else 0, scale_factor = 0.5 if count>0 else 1, is_fp16 = fp)
            name = f'AL{idx}_C{c_out}_R'
            self.down_names.append(name)
            setattr(self, name, layer)
            count+=1
          else:
            self.down_names.append(None)

        # self.test_conv = torch.nn.Conv2d(3,3,3,stride=1,padding=1,bias=True)

  def forward(self, x):
    x = torch.nn.functional.pad(x,(10,10,10,10),mode='reflect')
    x = self.in_proj(x.to(torch.float16))
    skips = []
    for idx, name in enumerate(self.down_names):
      if name:
        # print(x.shape)
        # print(name)
        x = getattr(self,name)(x)
        skips.append(x)
      else:
        skips.append(None)

    # for s in skips:
    #   if s is not None:
    #     print(s.shape)
    #   else:
    #     print(s)

    return skips