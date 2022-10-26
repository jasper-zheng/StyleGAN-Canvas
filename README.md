# Pixel2StyleGAN3  

<table> 
  <tr>
    <td> <img src='./docs/demo_imgs/1_cond.png'></td>
    <td> <img src='./docs/demo_imgs/1_gen.png'></td>
    <td> <img src='./docs/demo_imgs/2_cond.png'></td>
    <td> <img src='./docs/demo_imgs/2_gen.png'></td>
    <td> <img src='./docs/demo_imgs/3_cond.png'></td>
    <td> <img src='./docs/demo_imgs/3_gen.png'></td>
  </tr>
  <tr>
    <td> <img src='./docs/demo_imgs/4_cond.png'></td>
    <td> <img src='./docs/demo_imgs/4_gen.png'></td>
    <td> <img src='./docs/demo_imgs/5_cond.png'></td>
    <td> <img src='./docs/demo_imgs/5_gen.png'></td>
    <td> <img src='./docs/demo_imgs/6_cond.png'></td>
    <td> <img src='./docs/demo_imgs/6_gen.png'></td>
  </tr>
</table>  
 
#### Pix2StyleGAN3: Extended StyleGAN3 Architecture for Expressive Feature Exploration and Exploitation  

We present a new framework that extends StyleGAN3 architecture for real-time image-to-image translation tasks. First, we propose an appended encoder network with skip connections inserted directly into the StyleGAN3 generator, allowing the translation preserves more ﬁne details than a regular encoder-decoder. By leveraging state-of-the-art generator architecture, our approach solves a variety of image-to-image translation tasks while maintaining the image quality and the internal behaviour of StyleGAN3. Next, after demonstrating the framework is lightweight enough to run real-time inference, we propose implementing our framework on network bending, which is an approach for grouping and manipulating features in semantically meaningful ways to create divergence. 
