# FYP_DeepFake-Detection
## Datasets:
## 1.Deepfake Datasets

Datasets|Year|Ratio<br>tampered:original|Total videos|Source|Participants Consent|Tools
:-------:|:----:|:-----------:|:----:|:---:|:-----:|:--:
[FaceForensics](https://arxiv.org/abs/1803.09179)|2018|1 : 1.00|2008|YouTube|N|Face2Face
[FaceForensics++](https://github.com/ondyari/FaceForensics)|2019|1 : 0.25|5000|YouTube|N|faceswap <br> DeepFake <br> Face2Face <br> NeuralTextures
[DeepFakeDetection<br>(part of FaceForensics++)](https://deepfakedetectionchallenge.ai/dataset)|2019|1 : 0.12|3363|Actors|Y
[Celeb-DF](http://www.cs.albany.edu/~lsw/celeb-deepfakeforensics.html)|2019|1 : 0.51|1203|YouTube|N|a refined version of the DeepFake
[DFDC Preview Dataset](https://deepfakedetectionchallenge.ai/dataset)|2019|1 : 0.28|5214|Actors|Y|Unknown

## 2.Pre-Processing

Faces were extracted from each frame using KCF Trackers and Opencv

Time and Success rate analysis for each tracker

![KCF](https://user-images.githubusercontent.com/52126773/168428799-0efdafe2-44b0-4d61-b2de-3e4779718ac1.PNG)

## KCF Tracker
Kernelized Correlation Filters is an abbreviation for Kernelized Correlation Filters. This tracker expands on the concepts introduced in the boosting and MIL trackers. This tracker takes advantage of the fact that the MIL tracker's multiple positive samples have large overlapping regions. This overlapping data produces some interesting mathematical properties, which this tracker uses to make tracking faster and more accurate at the same time.

Because we know that a huge amount of data benefits our learning algorithm and that translating samples in a circulant manner provides a circulant structure of a large amount of data, we may develop and use the circulant nature of HOG features for a visual tracking advantage. Visual representation of HOG features is done in the image below. As a result, C(x) can now be described as a very high-dimensional data set that contains all possible translations of base sample HOG features (in the case of a 1D image) or base patch HOG features (in the case of a 2D image). The importance of such a data matrix (with shifted translations) for a learning algorithm is that it illustrates the various ways (input distribution) in which the samples can be encountered by the learning algorithm.

![1](https://user-images.githubusercontent.com/52126773/168428857-f8bb941f-6af4-46b2-840f-5c1a6249d62c.PNG)

## 3.Model Architecture

Vision Transformer:

While the Transformer architecture has become the highest standard for tasks involving natural language processing (NLP), its use cases relating to computer vision (CV) remain only a few. In computer vision, attention is either used in conjunction with convolutional networks (CNN) or used to substitute certain aspects of convolutional networks while keeping their entire composition intact. However, this dependency on CNN is not mandatory, and a pure transformer applied directly to sequences of image patches can work exceptionally well on image classification tasks.

Recently, Vision Transformers (ViT) have achieved highly competitive performance in benchmarks for several computer vision applications, such as image classification, object detection, and semantic image segmentation.

Model overview is : We split an image into fixed-size patches, linearly embed each of them,
add position embeddings, and feed the resulting sequence of vectors to a standard Transformer
encoder. In order to perform classification, we use the standard approach of adding an extra learnable
“classification token” to the sequence.

The performance of a vision transformer model depends on decisions such as that of the optimizer, network depth, and dataset-specific hyperparameters. Compared to ViT, CNNs are easier to optimize.

The disparity on a pure transformer is to marry a transformer to a CNN front end. The usual ViT stem leverages a 16*16 convolution with a 16 stride. In comparison, a 3*3 convolution with stride 2 increases the stability and elevates precision.

CNN turns basic pixels into a feature map. Later, the feature map is translated by a tokenizer into a sequence of tokens that are then inputted into the transformer. The transformer then applies the attention technique to create a sequence of output tokens. Eventually, a projector reconnects the output tokens to the feature map. The latter allows the examination to navigate potentially crucial pixel-level details. This thereby lowers the number of tokens that need to be studied, lowering costs significantly.

Particularly, if the ViT model is trained on huge datasets that are over 14M images, it can outperform the CNNs. If not, the best option is to stick to ResNet or EfficientNet. The vision transformer model is trained on a huge dataset even before the process of fine-tuning. The only change is to disregard the MLP layer and add a new D times KD*K layer, where K is the number of classes of the small dataset.





![transformer](https://user-images.githubusercontent.com/52126773/168429019-1a9561ad-9571-4c8f-b198-7da80ed50441.PNG)


Transformer Architecture

## 4. Our Final Architecture

![as](https://user-images.githubusercontent.com/52126773/168429055-305987fa-963d-4215-8c84-66da78ff3863.PNG)

## 5.Results:

https://www.youtube.com/watch?v=BJHqc2oKOUs



