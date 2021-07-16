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
