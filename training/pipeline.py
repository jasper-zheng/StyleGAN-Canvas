import torch
import numpy as np
from torch_utils.ops import upfirdn2d

# from training.canny_filter import CannyFilter

class Preprocess(torch.nn.Module):
    def __init__(self, blur_sigma = 21, scale_factor = 0.1, out_size = 256):
        super().__init__()
        # self.filter = CannyFilter(k_gaussian=5,mu=0,sigma=5,k_sobel=5)
        self.device = torch.device('cuda')
        self.blur_size = np.floor(blur_sigma * 3)
        self.blur_sigma = blur_sigma
        self.scale_factor = scale_factor
        self.out_size = out_size
        self.f = torch.arange(-self.blur_size , self.blur_size  + 1, device=self.device).div(self.blur_sigma).square().neg().exp2()
    @torch.no_grad()
    
    def preprocess_to_conditions(self, img_tensor):
        '''
        parameter: 
            img_tensor: [-1,1]
        return: 
            img_tensor: [-1,1]
        '''
        img_tensor = torch.nn.functional.interpolate(img_tensor, size = (self.out_size,self.out_size), mode='bilinear')
        img_tensor = torch.nn.functional.pad(img_tensor, (30,30,30,30), 'reflect')
        img_tensor = upfirdn2d.filter2d(img_tensor, self.f / self.f.sum())
        img_tensor = torch.nn.functional.pad(img_tensor, (-30,-30,-30,-30))
        
        return img_tensor