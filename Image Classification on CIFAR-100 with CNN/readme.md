# Image Classification on CIFAR-100 with CNN

## Problem
CNN, or the convolutional neural network model is a standard practice in the world of computer vision. There are plenty of factors that affect the performance of a CNN model, including dataset, regularization, kernel settings and ways to eliminate overfitting. 
 
The datasets are selected base on the overall effectiveness in identifying objects around us, as well as for easy manipulation for training and benchmarking.
The following are the strategies and ways that the project will assess on:

*	Model architecture: number of CNN layers and number of nodes

*	Amount of datapoints used: 200 images/500 images/1000 images/ 1000 images + data augmentation

*	avg pooling in the transitional layers

*	Dropout & Increasing dropouts

*	Weight decay

*	data generation/augmentation 

*	Batch Normalization

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
loss: 0.1856 - accuracy: 0.9456 - val_loss: 8.7 - **val_accuracy: 0.2732**

Model Plot            |  Accuracy & loss
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/77212888/125994296-130f3b95-a337-4bcb-b14c-7dc1b9f3404b.png)  |  ![image](https://user-images.githubusercontent.com/77212888/125994319-4e5ba061-ec53-47be-a27b-39a7adb070eb.png)


With only one convolution layer without anything else added. The model only achieves a 27.32% validation accuracy. On the contrary, the training accuracy has achieved as high as 94.56%. In later epochs, The training and validation loss and accuracy are starting to diverge significantly. It is clear that the input is not valued through this architecture for good results, and the overfitting is very severe.

### Trial 0 -2: One VGG block
loss: 0.1104 - accuracy: 0.9638 - val_loss: 10.9858 - **val_accuracy: 0.2841**

Model Plot            |  Accuracy & loss
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/77212888/125994515-e38f1e32-863d-478b-b6ee-c989da006012.png)  |  ![image](https://user-images.githubusercontent.com/77212888/125994526-99086d40-5fa4-476b-a5f1-0bc753c8c099.png)

### Trial 0 -3: three VGG block (Base Model)
loss: 0.4077 - accuracy: 0.8688 - val_loss: 6.4263 - **val_accuracy: 0.3068**

Model Plot            |  Accuracy & loss
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/77212888/125994591-b84c9962-7fae-434f-bde8-7536cf538ca9.png)  |  ![image](https://user-images.githubusercontent.com/77212888/125994596-291035b4-c01d-4cfc-a7fa-b58c7a44b5e4.png)


### VGG blocks:
For trial 0-2 and trial 0-3, the structure of a VGG block was used, where a consist of two 3 x 3 convolution layers followed with a 2 x 2 maxpooling layer according to the original VGG paper from Simonyan & Zisserman in 2014 ([1409.1556] Very Deep Convolutional Networks for Large-Scale Image Recognition (arxiv.org). 
As the result implies, the validation accuracy steadily increases with more complex convolution architectures. The Learning process of all three architectures are quite similar with a scissors shape, where the training accuracy keeps rising, validation accuracy keeps on dropping. But the gap is noticeable smaller as the layers become deeper. 
The Trial 0-3 model with three VGG block will be used as the base model for subsequent iterations. This model is chosen because it provides efficient results but more importantly each iteration will be trained swiftly for comparison.

*	**1 Conv Layer: 27.32%**

*	**1 VGG block:  28.41%**

*	**3 VGG block:  30.68%**

## Trial 1-10: Techniques benchmarking

### Trial 1: Base + avg pooling in the transitional layers
loss: 0.5371 - accuracy: 0.8318 - val_loss: 5.4147 - **val_accuracy: 0.3225**

![image](https://user-images.githubusercontent.com/77212888/125994883-bb7df49f-8be0-40ea-bd5a-b94a421ed9f1.png)

Average pooling in one of the transitional layers sometimes is a good practice. The practice was adopted from the author of  Densenet. The reason being putting the average pooling in the transitional layer would better signify the overall strength of the images.
The result indeed indicated that the validation accuracy increases by around 2% from 30.68% to 32.25% comparing to the base model. However, how would this average pooling in the transitional convolution layer perform still awaits testing in a VGG architecture or in a more complex CNN model.

### Trial 2: Base + 0.2 Dropouts 
loss: 2.2588 - accuracy: 0.4030 - val_loss: 2.3865 - **val_accuracy: 0.3925**

![image](https://user-images.githubusercontent.com/77212888/126078074-c45f7b19-5317-49b7-a654-1ddde7022716.png)

Dropout is one of the best practices that could close the gap of training accuracy and validation accuracy or help mitigate overfitting. The first trial test with 0.2 dropouts after each maxpooling layers and after the first dense layer. Hence in sum there are four dropout total. The result is exceptional in reducing overfitting as well as increasing accuracy. The validation accuracy is 39.25% with dropout on top of the base model. It is also clear that the graph does not show a scissors shape, but rather a more aligned train and test result.

### Trial 3: Base + Increasing Dropouts 
loss: 2.8490 - accuracy: 0.2860 - val_loss: 2.7526 - **val_accuracy: 0.3101**

![image](https://user-images.githubusercontent.com/77212888/126078168-41b9dc05-4a94-48a4-8717-1a2b18001166.png)


Trying different values of dropouts may or may not help the model performance. But is is common to have gradual increasing dropouts as the model become deeper. That forces the model to drop some of the deeper nodes out of the network. In this trial, the dropout rates goes from 0.2 -> 0.3 -> 0.4 -> 0.4. Model performance is inferior comparing to the 0.2 dropout trial with only 31.01% accuracy comparing to the 39.25%. However, the graphs tells part of the problem where the training accuracy is even lower than the validation accuracy at all time. This implies some important nodes were dropped out, which cause the model to underperform.

### Trial 4: Base + weight decay 
loss: 1.8614 - accuracy: 0.5816 - val_loss: 2.9800 - **val_accuracy: 0.3923**

![image](https://user-images.githubusercontent.com/77212888/126078171-890e2568-a308-4962-b81b-efc1d1cebb6f.png)


Weight decay is a regularization, or balancing technique that strive to penalize the model with its size of weight by updating the loss function. Larger weights are often penalized where they usually signify overfitting. The results of weight decay is convincing with 39.23% accuracy. The overfitting was lessened comparing to the base model, but it has less effect of combating overfitting comparing to using dropout.
In the trial, l2 regularization was used with 0.0005. Using 0.001 in this case, would result in too much penalty on the weight resulting in inefficient learning. However, the value depends on different models and datasets.

*“kernel_regularizer=regularizers.l2(0.0005)”*

### Trial 5: Base + data augmentation
loss: 2.4034 - accuracy: 0.3736 - val_loss: 2.4252 - **val_accuracy: 0.3742**

![image](https://user-images.githubusercontent.com/77212888/126078297-3c9940b1-dd7b-4800-8430-635ed0294727.png)

Data augmentation is also often used to generalize and add in distorted input picture so that the model would capture the important features. In this trail, the pictures were generated with 0.2 shear, 0.2 zoom, 30-degree rotation, 0.1 width and height shift and horizontal flip. These modifications will occur in a random fashion to the input’s pictures. And the result on top of the base model is promising with 37.42% accuracy. It is also clear that overfitting is mitigated.


### Trial 6: Base + Batch Normalization
loss: 0.3724 - accuracy: 0.8779 - val_loss: 3.8523 - **val_accuracy: 0.4138**

![image](https://user-images.githubusercontent.com/77212888/126078316-9c39d0eb-c941-4f35-a136-1f5c1de94f82.png)

Batch normalization is a technique that will normalize the inputs and it has effect of speed up the training process. As the result had shown, the validation accuracy achieved 41.38%, which is the best performing techniques among dropout, weight decay and data augmentation. However, there is still a significant overfitting in as shown on the graph.

## Summary of technique on top of base model benchmarking:
* Base + avg pooling: 		32.25% 
* Base + 0.2 Dropouts:         		39.25%
* Base + Increasing Dropouts:        31.01%
* Base + weight decay:                    39.23%
* Base + data augmentation:           37.42%
* Base + Batch Normalization:        41.38%

### Trial 7: Base + Increasing Dropouts + weight decay
loss: 3.3258 - accuracy: 0.2146 - val_loss: 3.0704 - **val_accuracy: 0.2678**

![image](https://user-images.githubusercontent.com/77212888/126078514-86257f9d-e93d-485a-a704-1b39b8fc4563.png)

The intuition of this trial is to balance and offset the effect of performance of increasing dropout that reduced overfitting with weight decay, which shows good results but with overfitting. As the validation accuracy implies, the performance does not translate from the intuition where it is even lower than the base model (30.68%)

### Trial 8: Base + 0.2 Dropouts + Batch Normalization + Data augmentation
loss: 2.3105 - accuracy: 0.3934 - val_loss: 1.9669 - **val_accuracy: 0.4766**

![image](https://user-images.githubusercontent.com/77212888/126078578-f29622a6-48e4-4f3f-9df7-035d50243b7b.png)

The combination of 0.2 dropouts, batch normalization and data augmentation resulted in much better result comparing to the previous model: 47.66%. This combination gave one of the best result yet. 

## VGG16:
![image](https://user-images.githubusercontent.com/77212888/126078984-7c4c9960-c080-40b1-a705-f003d018e1a4.png)

*(https://neurohive.io/en/popular-networks/vgg16/)*

In subsequent trials, VGG16 architecture will be applied. VGG16 is a architecture that employ the same concept of previous VGG block, the concept is shown up above with much more depth than the previous trials. It would also take much longer time to train the model as well. Trial 9 and Trial 10 will be two trails that strive to push to higher validation accuracy with 300 epoch.

### Trial 9: VGG16 + Increasing Dropouts + Batch Normalization + Data augmentation + weight decay + 300 epochs
loss: 0.6771 - accuracy: 0.7974 - val_loss: 1.9170 -  **val_accuracy: 0.5881**

### Trial 10: VGG16 + Increasing Dropouts + Batch Normalization + Data augmentation + weight decay + 300 epochs + Increasing nodes

loss: 0.9322 - accuracy: 0.7311 - val_loss: 1.4545 -  **val_accuracy: 0.6515**

![image](https://user-images.githubusercontent.com/77212888/126079178-7be3754c-18dd-4fb8-b66f-7d8df24e7039.png)

The two models are both performing significantly better with 58.81% accuracy with trial 9, and 65.15% in trial 10. The only difference between the two is that trial 10 follows the official VGG architecture with more nodes that trial 9, which resulted in 6.34% increase in validation accuracy.
One interesting observation is that the training process In both graphs shows big spikes all over, and the biggest spike both appeared around 160 epoch, but it bounce back up. This might be due to a small batch outlier. Using SGD might solve this issue

## Make Prediction:
The photos are pictures found on internet, pictures that the model had never seen before. Using these photos for testing would test the model’s performance in a real-world case and uncover some limitations to the model. The photos will be preprocessing and turn to 32 x 32 before using for prediction. Both prediction of the cloud and leopard are correct. While the model guessed the truck as a tank and guessed the mushroom as leopard. The wrong guess on the truck is understandable as the guess and input are in the same category. But the mushroom is no where near the leopard. One of the limitation is that the photos are 32 x 32, which eliminate many features, and some of the photos are hard to identify with human. In this case some of the features of the mushroom might be picked up as a leopard.
* Good Result: 85 tank, 23 cloud, 42 leopard
* Bad Result: Mushroom as 42 leopard
![image](https://user-images.githubusercontent.com/77212888/126079258-00907d13-9ede-4fe9-a6ac-dc88f1e4fde5.png)
![image](https://user-images.githubusercontent.com/77212888/126079260-28d65636-60dd-422f-82b4-523bdcc9e42c.png)

## Takeaways:
The benchmarking process uncovered plenty of interesting model behavior, including different behaviors with cifar-10 and cifar-100, as well as synergies of different techniques together. The effect of data augmentation and dropout are surprising and significant to both datasets. Furthermore, the model architectures are one of the most important contributors to a success model. Where the result can be improved tremendously by trying different methods as listed below.

*Further improvement:*
* Transfer Learning
* Explore more techniques, such as standardization.
* More complex and advanced architectures
* Using SGD and different learning rate
