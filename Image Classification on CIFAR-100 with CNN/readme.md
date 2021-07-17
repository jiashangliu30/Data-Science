# Image Classification on CIFAR-100 with CNN

## Problem
CNN, or the convolutional neural network model is a standard practice in the world of computer vision. There are plenty of factors that affect the performance of a CNN model, including dataset, regularization, kernel settings and ways to eliminate overfitting. 
 
The datasets are selected base on the overall effectiveness in identifying objects around us, as well as for easy manipulation for training and benchmarking.
The following are the strategies and ways that the project will assess on:

•	Model architecture: number of CNN layers and number of nodes

•	Amount of datapoints used: 200 images/500 images/1000 images/ 1000 images + data augmentation

•	avg pooling in the transitional layers

•	Dropout & Increasing dropouts

•	Weight decay

•	data generation/augmentation 

•	Batch Normalization

#### The Goal of the project is to benchmark the effect of different CNN architectures and strategies to increase the performance of the model regarding to the validation accuracy as the metrics.


## Dataset
The data is from (https://www.cs.toronto.edu/~kriz/cifar.html).
The cifar-100 dataset is photo collection of creatures, objects and plants. For example, some of the classes are Cloud, Beaver, bicycle. There are 100 classes and 60,000 photos in total, with 600 per class. It has 10 super classes that generalize the 100 detailed classes. In the benchmarking process, the 100 detailed classes will be assessed. Below shows graphs of the classes and some image examples.

The complete classes and some example photos are shown below:
![image](https://user-images.githubusercontent.com/77212888/125891194-5f1613d5-31f8-4507-920c-9322cbabe41c.png)
![image](https://user-images.githubusercontent.com/77212888/125891341-0c161ad5-448c-4bee-8345-9bb2901b84d3.png)   

## Model and Benchmarking process
The benchmarking process are divided into two main parts: 1. model architecture exploration(Trial 0);  2. Techniques benchmarking(Trial 1 +). The purpose is to find explore among different architectures and find the one architecture that shows the best result or have the greatest potential. The best one architecture chosen previously will be deployed throughout in the later benchmarking process.

Trial 0: Model architectures 

Trial 1: Base + avg pooling in the transitional layers

Trial 2: Base + 0.2 Dropouts

Trial 3: Base + Increasing Dropouts 

Trial 4: Base + weight decay 

Trial 5: Base + data augmentation

Trial 6: Base + Batch Normalization

Trial 7: Base + Increasing Dropouts + weight decay

Trial 8: Base + 0.2 Dropouts + Batch Normalization + Data augmentation

Trial 9: Base + Increasing Dropouts + Batch Normalization + Data augmentation + weight decay + 300 epochs

Trial 10: Base + Increasing Dropouts + Batch Normalization + Data augmentation + weight decay + 300 epochs

## Trial 0:Model architecture
This part will benchmark through different architecture of the CNN network, Starting from the simplest architecture with one convolution layer with (32, (3,3)) and maxpooling, to slightly more sophisticated multiple VGG blocks. The main goal of Trial 0 is to benchmark through some of the simplest architectures to stick with one simple proficient base model to keep going with further training and benchmarking.

### Trial 0 -1: one convolution layer 
loss: 0.1856 - accuracy: 0.9456 - val_loss: 8.7 - val_accuracy: 0.2732

Model Plot            |  Accuracy & loss
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/77212888/125994296-130f3b95-a337-4bcb-b14c-7dc1b9f3404b.png)  |  ![image](https://user-images.githubusercontent.com/77212888/125994319-4e5ba061-ec53-47be-a27b-39a7adb070eb.png)


With only one convolution layer without anything else added. The model only achieves a 27.32% validation accuracy. On the contrary, the training accuracy has achieved as high as 94.56%. In later epochs, The training and validation loss and accuracy are starting to diverge significantly. It is clear that the input is not valued through this architecture for good results, and the overfitting is very severe.

### Trial 0 -2: One VGG block
loss: 0.1104 - accuracy: 0.9638 - val_loss: 10.9858 - val_accuracy: 0.2841

Model Plot            |  Accuracy & loss
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/77212888/125994515-e38f1e32-863d-478b-b6ee-c989da006012.png)  |  ![image](https://user-images.githubusercontent.com/77212888/125994526-99086d40-5fa4-476b-a5f1-0bc753c8c099.png)

### Trial 0 -3: three VGG block (Base Model)
loss: 0.4077 - accuracy: 0.8688 - val_loss: 6.4263 - val_accuracy: 0.3068

Model Plot            |  Accuracy & loss
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/77212888/125994591-b84c9962-7fae-434f-bde8-7536cf538ca9.png)  |  ![image](https://user-images.githubusercontent.com/77212888/125994596-291035b4-c01d-4cfc-a7fa-b58c7a44b5e4.png)


### VGG blocks:
For trial 0-2 and trial 0-3, the structure of a VGG block was used, where a consist of two 3 x 3 convolution layers followed with a 2 x 2 maxpooling layer according to the original VGG paper from Simonyan & Zisserman in 2014 ([1409.1556] Very Deep Convolutional Networks for Large-Scale Image Recognition (arxiv.org). 
As the result implies, the validation accuracy steadily increases with more complex convolution architectures. The Learning process of all three architectures are quite similar with a scissors shape, where the training accuracy keeps rising, validation accuracy keeps on dropping. But the gap is noticeable smaller as the layers become deeper. 
The Trial 0-3 model with three VGG block will be used as the base model for subsequent iterations. This model is chosen because it provides efficient results but more importantly each iteration will be trained swiftly for comparison.

•	1 Conv Layer: 27.32%

•	1 VGG block:  28.41%

•	3 VGG block:  30.68%

## Trial 1-10: Techniques benchmarking

### Trial 1: Base + avg pooling in the transitional layers
loss: 0.5371 - accuracy: 0.8318 - val_loss: 5.4147 - val_accuracy: 0.3225

![image](https://user-images.githubusercontent.com/77212888/125994883-bb7df49f-8be0-40ea-bd5a-b94a421ed9f1.png)

Average pooling in one of the transitional layers sometimes is a good practice. The practice was adopted from the author of  Densenet:[1608.06993] Densely Connected Convolutional Networks (arxiv.org). The reason being putting the average pooling in the transitional layer would better signify the overall strength of the images.
The result indeed indicated that the validation accuracy increases by around 2% from 30.68% to 32.25% comparing to the base model. However, how would this average pooling in the transitional convolution layer perform still awaits testing in a VGG architecture or in a more complex CNN model.
