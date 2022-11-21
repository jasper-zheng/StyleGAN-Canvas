#### 07.01.2022  

start upgrading the [rewriting model](https://github.com/jasper-zheng/rewriting-activation-maps)    
* experiment on more models   
* migrating to StyleGAN3   

#### 08.01.2022  

Setting up demo in Flask    

#### 08.08.2022  

implement the rewriting model to Flask     

#### 08.22.2022  

extending to higher resolution    

#### 09.02.2022   

2nd tutorial    
adding skip connections    
adapting StyleGAN3's training framework    

#### 09.12.2022  

revised encoder network (CNN -> ResNet)   
changing skip connections' arrangements (limit to 5 layers)   

#### 09.19.2022  

training deblurring model on 256x256    
change the conditional input layer (learned transformation -> direct mapping)    
investigating the model collapse    

#### 09.26.2022  

training edge-to-face model on 256x256    
training inversion model on 256x256   
investigating the model collapse    

#### 09.28.2022  

3rd tutorial   
investigating the model collapse    

#### 10.03.2022  

model collapse due to the missing of batch normalisation and the spikes in signal caused by casting FP32 to FP16   
training inversion model on 512x512    

#### 10.10.2022   

change the sequence of layers in the encoder block (maxpool -> stride)   
retraining inversion model on 512x512    

#### 10.17.2022   

reproducing the classifier in the clustering method in network bending   
setup graphic interface   

#### 10.26.2022   

4th tutorial
training deblurring model on FFHQ 512x512   
training deblurring model on LHQ 512x512   

#### 10.31.2022   

user test planning    
ablation tests    

#### 11.07.2022   

finalise model architecture (move the feature fusion point before the filter non-linearities)   
retraining deblurring model on FFHQ 512x512   
retraining deblurring model on LHQ 512x512    

user test   

#### 11.14.2022    

training edge-to-face model on LHQ 512x512    
