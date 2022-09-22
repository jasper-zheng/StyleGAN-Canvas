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
      x = self.conv2(x, gain=np.sqrt(0.5))
      x = torch.nn.functional.pad(x,(self.paddings,self.paddings,self.paddings,self.paddings), mode='reflect')
      
      x = short_cut.add_(x)

      x = torch.nn.functional.leaky_relu(x, 0.2)

      # pass
      return x


@persistence.persistent_class
class AppendedNet(torch.nn.Module):
  def __init__(self,
        img_channels,
        img_sizes,
        p_dim = (3,256,256),
        skip_channels_idx = [0, 3, 6, 7, 10],
        skip_connection   = [1, 1, 0, 0,  0],
        layer_fp16 = [],
        num_appended_ws = 3
    ):
        super().__init__()
        self.p_dim = p_dim
        self.img_sizes = img_sizes
        self.img_channels = img_channels
        self.skip_channels_idx = skip_channels_idx
        self.num_appended_ws = num_appended_ws
        # self.skip_connection = skip_connection
        print('hiiii appended net')
        self.device = torch.device('cuda')
        layer_fp16.reverse()
        self.layer_fp16 = layer_fp16

        self.skip_down_channels = []
        self.skip_down_sizes = []
        self.skip_connection = []
        count = 0
        for idx, (c, s) in enumerate(zip(self.img_channels, self.img_sizes)):
          if idx in skip_channels_idx:
            self.skip_down_channels.append(int(c//2))
            self.skip_down_sizes.append(int(s))
            if skip_connection[count]:
                self.skip_connection.append(1)
            else:
                self.skip_connection.append(0)
            count += 1
          else:
            self.skip_down_channels.append(0)
            self.skip_down_sizes.append(0)
            self.skip_connection.append(0)

        self.skip_down_channels.reverse()
        self.skip_down_sizes.reverse()
        self.skip_connection.reverse()

        print(self.skip_down_channels)
        print(self.skip_down_sizes)
        print(self.skip_connection)
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
            out_size = int(0.5*out_size) if count>0 else out_size
            name = f'AL{idx}_R{out_size}_C{c_out}'
            self.down_names.append(name)
            setattr(self, name, layer)
            count+=1
            
          else:
            self.down_names.append(None)
        print(f'last skip shape: {out_size}')

        self.epilogue = []
        # 16 -> 8 -> 4
        for idx in range(2):
          out_size = out_size//2
          c_in = down_channels[-1]//2 if idx > 0 else down_channels[-1]
          c_min = down_channels[-1]//2
          c_out = down_channels[-1]//2
          layer = ResConvBlock(c_in, c_mid, c_out, kernel_size=3, paddings = 0, scale_factor = 0.5, is_fp16 = False)
          name = f'EP{idx}_R{out_size}_C{c_out}'
          self.epilogue.append(name)
          setattr(self, name, layer)
        
        print(f'epilogue: R{out_size} C{c_out}')

        name = f'EP_flatten'
        layer = torch.nn.Flatten(start_dim=1)
        self.epilogue.append(name)
        setattr(self, name, layer)
        computed_flatten_size = out_size * out_size * c_out
        in_c = int(computed_flatten_size)
        out_c = 512
        name = f'EP_mapping_1'
        layer = torch.nn.Linear(in_c,out_c,bias=True)
        self.epilogue.append(name)
        setattr(self, name, layer)
        name = f'EP_mapping_1_act'
        layer = torch.nn.LeakyReLU(0.2)
        self.epilogue.append(name)
        setattr(self, name, layer)
        name = f'EP_mapping_2'
        layer = torch.nn.Linear(out_c,out_c,bias=True)
        self.epilogue.append(name)
        setattr(self, name, layer)
        name = f'EP_mapping_2_act'
        layer = torch.nn.LeakyReLU(0.2)
        self.epilogue.append(name)
        setattr(self, name, layer)




        # self.test_conv = torch.nn.Conv2d(3,3,3,stride=1,padding=1,bias=True)

  def forward(self, x, num_appended_ws_len = None):
    x = torch.nn.functional.pad(x,(10,10,10,10),mode='reflect')
    x = self.in_proj(x.to(torch.float16))
    skips = []
    for idx, (name, connect) in enumerate(zip(self.down_names,self.skip_connection)):
      if name:
        # print(x.shape)
        # print(name)
        x = getattr(self,name)(x)
        
        if connect:
            skips.append(x)
        else:
            skips.append(None)
      else:
        skips.append(None)

    x = torch.nn.functional.pad(x,(-10,-10,-10,-10))
    for idx, name in enumerate(self.epilogue):
      # print(name)
      # print(x.shape)
      x = getattr(self,name)(x)
    appended_ws_len = self.num_appended_ws if num_appended_ws_len == None else num_appended_ws_len
    x = x.unsqueeze(1).repeat([1, appended_ws_len, 1])
    # for s in skips:
    #   if s is not None:
    #     print(s.shape)
    #   else:
    #     print(s)

    return skips, x