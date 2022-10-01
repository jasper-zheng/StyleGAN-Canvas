from torch_utils import persistence

import torch
import numpy as np

from math import log2

@persistence.persistent_class
class DownConv2dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        kernel_size,                    # Width and height of the convolution kernel.
        paddings        = 0,
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
        self.conv2 = DownConv2dLayer(in_channels, out_channels, kernel_size = kernel_size, paddings = kernel_size//2, bias = True, activation = activation, is_fp16 = is_fp16)

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
      # short_cut = torch.nn.functional.pad(short_cut,(self.paddings,self.paddings,self.paddings,self.paddings), mode='reflect')
      # print(f'shortcut:{short_cut.shape}')
      x = self.conv1(x)
      x = self.conv2(x, gain=np.sqrt(0.5))
      # x = torch.nn.functional.pad(x,(self.paddings,self.paddings,self.paddings,self.paddings), mode='reflect')
      
      x = short_cut.add_(x)
      x = torch.nn.functional.leaky_relu(x, 0.2)

      # x = torch.nn.functional.pad(x,(self.paddings,self.paddings,self.paddings,self.paddings), mode='reflect')

      return x

class AppendEpilogue(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        in_res,
        out_channels                   # Number of output channels.
    ):
        super().__init__()
        print(f'epilogue: ({in_channels}, {in_res}, {in_res})')
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.mapping_1 = torch.nn.Linear(in_channels*in_res*in_res, out_channels, bias=True)
        self.mapping_2 = torch.nn.Linear(out_channels, out_channels, bias=True)

    def forward(self, x):
      x = self.flatten(x)
      x = torch.nn.functional.leaky_relu(x, 0.2)
      x = self.mapping_1(x)
      x = torch.nn.functional.leaky_relu(x, 0.2)
      x = self.mapping_2(x)
      x = torch.nn.functional.leaky_relu(x, 0.2)
      return x

@persistence.persistent_class
class AppendedNet(torch.nn.Module):
  def __init__(self,
        img_channels,
        img_sizes,
        p_dim = (3,256,256),
        # skip_channels_idx = [0, 3, 5, 7, 10],
        # skip_channels_idx  = [  0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  0,  0,  0,  0],

        # g_channels_res     = [256, 256, 256, 256, 256, 128, 128, 128, 64, 64, 32, 32, 16, 16, 16],
        encoder_out_res    = [128, 64, 64, 32, 32, 16, 16,  8,  4],
        encoder_channels   = [ 64, 64,128,256,256,512,512,512,512],
        encoder_connect_to = [  0,  0,  0,  4,  3,  2,  1,  0,  0],
        # encoder_receive    = [  0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  4,  3,  2,  1,  0],
        layer_fp16 = [],
        num_appended_ws = 3,
    ):
        super().__init__()
        self.p_dim = p_dim
        self.img_sizes = img_sizes
        self.img_channels = img_channels
        # self.skip_channels_idx = skip_channels_idx
        self.num_appended_ws = num_appended_ws
        self.encoder_connect_to = encoder_connect_to
        # self.encoder_receive = encoder_receive
        # self.skip_connection = skip_connection
        print('hiiii appended net')
        self.device = torch.device('cuda')
        layer_fp16.reverse()
        self.layer_fp16 = layer_fp16

        self.skip_down_channels = []
        self.skip_down_sizes = []
        self.skip_connection = []
        self.skip_scale = []
        count = 0
        for idx, (c, s) in enumerate(zip(self.img_channels, self.img_sizes)):
          self.skip_down_channels.append(int(c//1))

        self.skip_down_channels.reverse()
        # self.skip_down_sizes.reverse()
        # self.skip_connection.reverse()
        # self.skip_scale.reverse()

        # print(f'c: {self.skip_down_channels}')
        # print(self.skip_down_sizes)
        # print(self.skip_connection)
        # print(self.skip_scale)
        # assert len(self.skip_down_channels) == len(self.skip_down_sizes), f'what \n{self.skip_down_channels}\n{self.skip_down_sizes}'
        # print(p_dim)
        print(encoder_out_res)
        print(encoder_connect_to)
        print(encoder_channels)
        
        ### compute paddings
        # first_size = next((x for x in self.skip_down_sizes if x), None) #276
        # paddings = (first_size - p_dim[1])//2
        paddings = 10
        print(f'padding {paddings}')

        self.in_proj = DownConv2dLayer(p_dim[0], encoder_channels[0], 1, paddings = 0, bias=True, activation='lrelu', is_fp16 = layer_fp16[0])

        self.down_names = []
        out_size = p_dim[1]

        for idx, (c,r,fp,ct) in enumerate(zip(encoder_channels, encoder_out_res, layer_fp16, encoder_connect_to)):

          c_in = c
          c_mid = c
          c_out = c if idx == len(encoder_channels)-1 else encoder_channels[idx+1]
          scale_factor = 0.5 if idx == 0 else r/encoder_out_res[idx-1]
          out_size = int(out_size*scale_factor)
          computed_fp = r>=32
          layer = ResConvBlock(c_in, c_mid, c_out, kernel_size=3, paddings = 0, scale_factor = scale_factor, is_fp16 = computed_fp)
          # out_size = int(0.5*out_size) if count>0 else out_size
          name = f'AL{idx}_R{out_size}_C{c_out}_{ct}'
          self.down_names.append(name)
          print(f'{name}\t fp16: {computed_fp}')
          setattr(self, name, layer)

          # for i in range(int(log2(out_size))-2):

        print(f'last skip shape: {out_size}')

        self.epilogue = AppendEpilogue(c_out, out_size, 512)


  def forward(self, x, num_appended_ws_len = None):
    # x = torch.nn.functional.pad(x,(10,10,10,10),mode='reflect')
    x = self.in_proj(x.to(torch.float16))
    skips = []
    # if self.in_scale is not None:
    #     x = self.in_scale(x)
        # print(f'-> {x.shape}')
        # print("scale")
    for idx, (name, ct) in enumerate(zip(self.down_names, self.encoder_connect_to)):
      layer = getattr(self,name)
      x = layer(x)
      if ct:
        skips.append(x)
      
    x = self.epilogue(x)

    appended_ws_len = self.num_appended_ws if num_appended_ws_len == None else num_appended_ws_len
    x = x.unsqueeze(1).repeat([1, appended_ws_len, 1])
    return skips, x









