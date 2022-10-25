
import argparse
from PIL import Image
import os

import legacy
import dnnlib

import torch

from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from IPython import display

import torchvision
from torch_utils.ops import upfirdn2d

# from training.training_loop import Preprocess

convert_tensor = ToTensor()

if __name__ == '__main__':
  
  device = torch.device('cuda')

  parser = argparse.ArgumentParser()

  parser.add_argument('--network-pkl', type=str)
  parser.add_argument('--out-path', type=str)
  parser.add_argument('--data-path', type=str)
  parser.add_argument('--layer-idx', type=int)
  parser.add_argument('--layer-res', type=int)
  parser.add_argument('--layer-channel', type=int)
  parser.add_argument('--total-kimg', type=int, default=10)
  parser.add_argument('--div', type=int, default=3)
  parser.add_argument('--blur-skips', type=int, default=0)

  args = parser.parse_args()

  convert_tensor = ToTensor()

  with dnnlib.util.open_url(args.network_pkl) as f:
    model = legacy.load_network_pkl(f)
    g_model = model['G'].eval().requires_grad_(False).to(device)
  
  print(g_model)
  
  layer_name = f'L{args.layer_idx}_{args.layer_res+20}_{args.layer_channel}'
  print(f'collecting features from {layer_name}')

  base_path = args.out_path
  os.mkdir(f'{base_path}')
  for i in range(args.layer_channel):
    os.mkdir(f'{base_path}/{i:04}')

  for k_idx in range(args.total_kimg):
    img_in_folder = f'{args.data_path}/{k_idx:05}'
    kimg = k_idx

    for img_idx in range(kimg*1000,(kimg+1)*1000):
      if img_idx%100==0:
        print(f'collecting training set from {img_in_folder}/img{img_idx:08}.png')

      img = Image.open(f'{img_in_folder}/img{img_idx:08}.png').resize((256,256)).convert("RGB")
      img_tensor = convert_tensor(img).to(device).unsqueeze(0)*2-1

      skips_out, replaced_w = g_model.appended_net(img_tensor.to(torch.float16))

      # replaced_w = replaced_w_2
      # skips_out = skips_out_2

      skips_out.reverse()
      replaced_w = replaced_w.to(torch.float32).unbind(dim=1)
      
      # Execute layers.

      x = upfirdn2d.filter2d(skips_out[0], g_model.synthesis.filter / g_model.synthesis.filter.sum()) if g_model.synthesis.blur_sigma else skips_out[0]
      x = g_model.synthesis.input(x)

      for idx, (name, w, s) in enumerate(zip(g_model.synthesis.layer_names, replaced_w[1:], g_model.synthesis.encoder_receive)):
          # print(name)
          # print(x.shape)
          layer = getattr(g_model.synthesis, name)
          if s:
            concat = torch.nn.functional.pad(skips_out[s-1],(10,10,10,10), mode='reflect')
            concat = upfirdn2d.filter2d(concat, g_model.synthesis.filter / g_model.synthesis.filter.sum()) if g_model.synthesis.blur_sigma else concat
            x = torch.cat([x,concat],dim=1)
          x = layer(x, w)

          if name==layer_name:
            y = x
            y = torch.nn.functional.pad(x.detach(),[-10,-10,-10,-10])

            for c_idx, c in enumerate(y[0]):
              map = c.to(torch.float32).unsqueeze(0).add(1).div(args.div).clamp(0, 1)
              map = to_pil_image(map.cpu())
              map.save(f'{base_path}/{c_idx:04}/{img_idx:05}.png')

            break
      