# Stereo-from-Monocular-Image
This repository ia a finalproject for 3D Computer Vision course in 2022 fall semester at National Taiwan University.

## Description


## Image to Depth
Due to limitation of CPU RAM size and the ability of GPU, this project use MiDaS-Small instead of MiDaS-Large which is a more accurate pretrained model. Then, attached MiDaS-Small with a self-built CNN model to generate depth.
![image](picture or gif url)

## Image Warpping
Before warpping the images, we first processed the depth to perform an additional unlinear transform. After the preocessing, we combine Bilinear Interpolation and Backgoud Filling to warp the images.
![image](compare/gif/groudtruth.gif)
(Groudtruth stereo image)
![image](compare/gif/without_processing.gif)
(Synthesized stereo image w/o processing)
![image](compare/gif/with_processing.gif)
(Synthesized stereo image w/ processing)
