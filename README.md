# FYP_DeepFake-Detection
## Datasets:
## 1 Deepfakes Datasets

Datasets|Year|Ratio<br>tampered:original|Total videos|Source|Participants Consent|Tools
:-------:|:----:|:-----------:|:----:|:---:|:-----:|:--:
[FaceForensics](https://arxiv.org/abs/1803.09179)|2018|1 : 1.00|2008|YouTube|N|Face2Face
[FaceForensics++](https://github.com/ondyari/FaceForensics)|2019|1 : 0.25|5000|YouTube|N|faceswap <br> DeepFake <br> Face2Face <br> NeuralTextures
[DeepFakeDetection<br>(part of FaceForensics++)](https://deepfakedetectionchallenge.ai/dataset)|2019|1 : 0.12|3363|Actors|Y
[Celeb-DF](http://www.cs.albany.edu/~lsw/celeb-deepfakeforensics.html)|2019|1 : 0.51|1203|YouTube|N|a refined version of the DeepFake
[DFDC Preview Dataset](https://deepfakedetectionchallenge.ai/dataset)|2019|1 : 0.28|5214|Actors|Y|Unkonwn

## 2 Pre-Processing

Faces were extracted from each frame using KCF Trackers and Opencv
