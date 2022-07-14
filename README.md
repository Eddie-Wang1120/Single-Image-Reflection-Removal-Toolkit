# Single-Image-Reflection-Removal-Toolkit
This is a Toolkit uses five different models to remove the reflection on single image.The results of the removal (including PSNR/SSIM and the image after reflection removal) will be obtained by using a web-application constructed by flask.

## Refection Removal Results
<img src="./img/result.png">

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

network structure<br>
<img src="./img/network.png">

loss functions<br>
<img src="./img/loss.png">

model effects<br>
<img src="./img/model_effect.png">

## Toolkit functions
1 Using one image with reflection as input, receiving five images (each uses a different model) as output.<br>
<img src="./img/one_input.png">
<img src="./img/one_res.png">

2 Using one testset as input(including image with reflection-I, image without reflection-B, residual image-R), receving PSNR/SSIM effects comparison and image after reflection removal as output.<br>
<img src="./img/multi_input.png">
<img src="./img/multi_res.png">

## Toolkit instructions
1 neccesary resources downloading<br>
[Supplment files](https://pan.baidu.com/s/1fF4x0eraelU1O1Ank77QSw) code: q5ro<br>
need to put the files into the right position in the toolkit folder<br>

2 Dataset<br>
[[DORM]](https://github.com/Eddie-Wang1120/Single-Image-Reflection-Removal-Dorm-Dataset)<br>
[[SIR2]](https://rose1.ntu.edu.sg/dataset/sir2Benchmark/)<br>
[[Wen et al. (syn)]](https://github.com/csqiangwen/Single-Image-Reflection-Removal-Beyond-Linearity#reflection-removal)<br>
[[BDN(syn)]](https://github.com/yangj1e/bdn-refremv#datasets)<br>
[[Zhang et al.]](https://drive.google.com/drive/folders/1NYGL3wQ2pRkwfLMcV2zxXDV8JRSoVxwA)<br>

3 tips<br>
When upload a testset to the web application, please change the test data into the right position(just like the example), compress the folder into the .zip format,and rename it to dataset.zip.
