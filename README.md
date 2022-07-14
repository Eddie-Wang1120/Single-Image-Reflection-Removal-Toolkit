# Single-Image-Reflection-Removal-Toolkit
This is a Toolkit uses five different models to remove the reflection on single image.The results of the removal (including PSNR/SSIM and the image after reflection removal) will be obtained by using a web-application constructed by flask.

## Five models
1 BDN<br>
  Seeing Deeply and Bidirectionally: A Deep Learning Approach for Single Image Reflection Removal<br>
  Jie Yang\*, Dong Gong\*, Lingqiao Liu, Qinfeng Shi.(ECCV 2018)<br>

2 IBCLN<br>
  Single Image Reflection Removal through Cascaded Refinement<br>
  Chao Li, Yixiao Yang, Kun He, Stephen Lin, John E. Hopcroft.(CVPR 2020)<br>
  
3 PRN<br>
  Single Image Reflection Removal with Perceptual Losses<br>
  Xuaner Zhang, Ren Ng, Qifeng Chen.(CVPR 2018)<br>
  
4 RR<br>
  Single Image Reflection Removal with Physically-Based Training Images<br>
  Soomin Kim, Yuchi Huo, and Sung-Eui Yoon.(CVPR 2020)<br>
  
5 IBLNN<br>
  It's a model designed by me. The paper has not been publicted, so I will explain this model in the following paragraphs, thus to let the readers understand the details.
  
## IBLNN model

network structure
<img>

loss functions
<img>

model effects
<img>

## Toolkit functions
1 Using one image with reflection as input, receiving five images (each uses a different model) as output.
<img>

2 Using one testset as input(including image with reflection-I, image without reflection-B, residual image-R), receving PSNR/SSIM effects comparison and image after reflection removal as output.
<img>
