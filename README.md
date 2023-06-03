<p align="center"><img src="https://socialify.git.ci/adnan760/Plant_Health_Detection/image?description=1&amp;name=1&amp;pattern=Circuit%20Board&amp;theme=Auto" alt="project-image"></p>
 
<h2>Motivation</h2>

*   To use deep learning techniques namely CNN and Mask RCNN in detecting plant diseases at an early stage based on respective plant leaves.
*   Performance analysis of proposed deep learning techniques to help choose the right model based on the needs.
  
<h2>Features</h2>

*   Simplified disease detection on potato and tomato plant leaves by CNN approach.
*   Detailed disease detection on potato and tomato plant leaves, highlighting the affected regions by Mask RCNN approach.
*   Detection of healthy leaves of these plant species by CNN and Mask RCNN approach. 

<h2>Dataset</h2>

[New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
<p></p>

<h2>Modules</h2>

*  tensorflow
*  keras
*  matplotlib
*  sklearn
*  skimage
*  anvil
*  mrcnn by [matterport](https://github.com/matterport/Mask_RCNN)

<h2>Model Architecture</h2>

<h3>CNN</h3>

The proposed CNN model uses a sequential approach. In the CNN model, the input image passes through a series of convolution layers with filters followed by pooling layers and a fully connected layer for classification using Softmax function for classification of multiple classes. The convolution layer will extract the features by applying 3x3 filters/kernels and 32, 64 and 128 channels before proceeding to the pooling layer. The Rectified Linear Unit (ReLU) is a non-linear activation function that is applied between convolution and pooling layers for faster training of data. Pooling layer helps in reducing the parameters of image so as to reduce the computing power required for training the model. Unlike the convolutional layer, the pooling layer also applies 2x2 filters. In the pooling layer max pooling is done by selecting the largest value from the matrix by convolutional layer  and this value is transferred to a new matrix. The matrix is then flattened and fed to a fully connected layer which is a network of interconnected neurons. The fully connected layer provides classification of respective leaf images by considering the feature extracted based on classes defined.
<p></p>
<img src="https://user-images.githubusercontent.com/94967712/219949390-2a095026-573f-462a-9b0d-38f1559d37c1.png" width="65%" height="65%/">

<h3>Mask RCNN</h3>

In the Mask RCNN model, the input image passes through the ResNet-101 which is a model based on Convolutional Neural Network architecture pre-trained on MS COCO dataset which is a large-scale image dataset containing 328,000 images of everyday objects and humans. The ResNet-101 model does the feature extraction of the respective input image and generates feature maps. Regional Proposal Network (RPN) then does the object detection on image by analyzing the object in image and extracting the regions which is then processed by Region of Interest (ROI) Pooling layer which arranges the regions into a fixed aspect ratio. The feature maps and regions are then processed by a fully connected layer providing a respective classification of image and a bounding box for outlining the regions which is then followed by mask generation phase for masking/highlighting the regions on the respective leaf image. 
<p></p>
<img src="https://user-images.githubusercontent.com/94967712/219949499-2c7cdcdb-da53-4bfb-a12f-92e08e8aa487.png" width="65%" height="65%/">

<h2>Results</h2>
Six classes:

For diseased leaf:

* Potato Early Blight
* Potato Late Blight
* Tomato Leaf Mold
* Tomato Leaf Spot

For healthy leaf:
* Potato Healthy
* Tomato Healthy

<h3>Results from CNN model</h3>

<img src="https://user-images.githubusercontent.com/94967712/219949684-b5658d28-1932-4d54-9fd2-90870a02ec34.png" width="65%" height="65%/">
<h3>Results from Mask RCNN model</h3>

<img src="https://user-images.githubusercontent.com/94967712/219949727-0822fdfc-0103-4703-9e31-6c597af4de90.png" width="65%" height="65%/">

<h2>Analysis</h2>
<h3>Model Accurateness</h3>
Table shows the accuracy measures of CNN and Mask RCNN models. One metric for assessing classification models is accuracy. The percentage of accurate predictions provided by the model is known as accuracy. A common statistic for evaluating the accuracy of an object detection model is the Mean Average Precision (mAP). mAP is the average precision (AP) value obtained over recall values ranging from 0 to 1. CNN model gives an accuracy of 0.971 whereas Mask RCNN gives an accuracy of 0.845.
The F1-score is an error metric that calculates the harmonic mean of precision and recall to evaluate model performance. It offers reliable results for both balanced and unbalanced datasets and considers the model's precision and recall capabilities.CNN model gives F1-score of 0.971 while Mask RCNN gives F1-score of 0.871.
<p></p>
<img src="https://user-images.githubusercontent.com/94967712/219950166-a584b071-9643-4435-b12e-a3939d36200b.png" width="65%" height="65%/">

<h3>Model Losses</h3>
Table shows the loss values for CNN and Mask RCNN models. Loss is a value that symbolises the total of our model's errors. It indicates how well (or poorly) the model is performing. CNN model is trained on 5 epochs reducing the training and validation losses to 0.011 and 0.102 respectively while Mask RCNN model is trained on 25 epochs reducing the training loss to 0.625 and validation loss to 0.748.
<p></p>
<img src="https://user-images.githubusercontent.com/94967712/219950437-13d5763a-2f6e-4c16-bbd1-8f8921129d5d.png" width="65%" height="65%/">

 
