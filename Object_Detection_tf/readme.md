# Product Listing Auto Filler using Object Detection


Product                    |  Listing 
:-------------------------:|:-------------------------:
<img src="https://user-images.githubusercontent.com/77212888/128608971-e4af1c9f-2a8b-49ab-b2ab-40fb1bd4cada.gif">    |  ![image](https://user-images.githubusercontent.com/77212888/128619980-e23ca446-b879-40fd-aac6-491c3808e38d.png)


## Problem
Object Detection can be used for a variety of purposes. To name a few, video editing, face recognition, and health care. They have all been taking advantage of the technology. It can also be used to be implemented to improve user experience.

The foundation of the inspiration came from online marketplaces such as eBay. I had also worked on a similar application and received user feedback. A very common voice in the feedback was, the listing process can be cumbersome, including taking pictures, setting a price, and type product information, descriptions. Furthermore, it makes users discouraged to list more products, even though it could lead to more profits. What is a feature that could be implemented that could address the feedback and improve user experience? 

The project idea is to take advantage of object detection during the picture-taking, or uploading stage to identify the product, and subsequently, automatically fill out the basic information as well as the advised selling price or average selling price. This feature would make the seller’s listing process simpler. 

## Goal
The goal of the project is to make a proof of concept prototype of an object detector. Moreover, to benchmark the capability of object detection using TensorFlow Object Detection API with various model architectures and parameters, and finally, find the most efficient model that has a high mAP score and suitable speed.

## Datasets
### PASCAL VOC 2007
The dataset is from https://academictorrents.com/details/c9db37df1eb2e549220dc19f70f60f7786d067d4. It has been used extensively in image recognition and object detection tasks, with sizable instances (9961 images total) and common objects classes. Moreover, It would be a good dataset to benchmark through different pre-trained models and hyper-parameters for the most efficient results. The dataset is annotated with bounding box annotation in pascal XML format, as well as segmentation annotations. In this project, we will focus on bounding box annotation. 

The classes are:
- Person: person
- Animal: bird, cat, cow, dog, horse, sheep
- Vehicle: aeroplane, bicycle, boat, bus, car, motorbike, train
- Indoor: bottle, chair, dining table, potted plant, sofa, tv/monitor
Classes including indoor items such as chairs, monitors are common objects listed on the marketplace which will be useful to detect.

<img src="https://user-images.githubusercontent.com/77212888/128619337-134ccf5d-da49-487c-abd9-a1efd0d19ce0.png" width="75%" height="75%">

### Custom Labeled Product Dataset
The dataset is collected and annotated with a bounding box in XML format as a PoC for future deployment. To test the ability of the transferred learning model with our minimum viable product, the classes will be kept to 3 with 6-8 instances per class, which leads to 21 images total for the training. The annotation task is performed on CVAT. From there, either tfrecord or XML format can be exported for training.
The classes are:
- Wallet
- Keyboard
- Mouse
<img src="https://user-images.githubusercontent.com/77212888/128619353-ac107077-51b8-41bc-85c8-f0b8729c4c66.png" width="75%" height="75%">

## Evaluation Results on PASCAL VOC 2007
The experiment results were compared to the TensorFlow 2 Model Zoo. In table 1, the model was organized by the fastest speed to the slowest speed. It was clear that higher inference time has a higher COCO mAP. However, it was not the case for CenterNet HourClass104 512x512. It has a somewhat acceptable inference time but the highest accuracy. This was why the CenterNet was included in the experiment.

<img src="https://user-images.githubusercontent.com/77212888/128619369-f2b1fbc1-02cf-45be-8efb-567bbcc89359.png" width="75%" height="75%">
<img src="https://user-images.githubusercontent.com/77212888/128619519-95db5966-7c90-4f1d-a66c-31758b1e9375.png" width="75%" height="75%">

In table 2, it was interesting that SSD ResNet50 had an overall much higher mAP score than SSD ResNet101. According to table 1, SSD ResNet101 has a longer inference time with a higher mAP score than ResNet50. One possibility could be the different model architectures affected the model to train differently. 

For this project, it is more important to have a higher mAP and AR score than a higher speed for detecting. For instance, a self-driving car would need a lower inference time because it requires real-time detection. This project is designed for detecting objects in the listing phase for online markets to extract information from the product, and then auto-fill for the seller. The speed difference of milliseconds would not make a big difference in the user experience. It is more important for the model to identify the product and be able to show it to the user. Therefore, the best-fitted model architecture would be CenterNet HourGlass104. It achieved 61.5 in mAP@0.5(often used as a pascal VOC metric) and 40.2 in mAP. Moreover, the AR performance is 63.8.

## Predictions on PASCAL VOC 2007
<img src="https://user-images.githubusercontent.com/77212888/128619549-ed382842-e1c1-44ef-817b-b4e8e824c7a8.png" width="75%" height="75%">

First, we look at how the model performed against its own evaluation dataset. In figure 1, it identifies the person in the back with a precise bounding box. However, the person in front of him was not detected. We assume that objects in the back or smaller are harder to detect, however the evaluation result contradicts our assumption. The reason could be the person in the front only shows ¼ of his body with a unique pose.

<img src="https://user-images.githubusercontent.com/77212888/128619559-6740c26a-3b5f-47fa-8695-5270086e1f7c.png" width="75%" height="75%">

Subsequently, we use photos that are collected from the internet, which were never seen by the model to test its performance. As results shown in figure 2, the buses were detected with a precise bounding box, even the bus on the left with fewer features were detected. Moreover, the couch was detected with a good bounding box position.


## Evaluation Results and prediction on Custom Labeled Product Dataset
The CenterNet HourGlass 104 model is trained on top of our model on VOC 2007. The new input will be the freezed ckpt-0 from our previous trained model. This practice can be useful when training multiple times with different datasets. 

The evaluation as shown on figure 3 conveys promising results. It also qualifies as a PoC with only 6-8 instances per class and 3000 training steps. We had achieved 62.5 for mAP@0.5, 51.3 for mAP and 65 for AR. The reason that APs, APm, ARs, ARm are all values of -1 is because of the lack of diversity of the dataset: all of the photos are taken with the object in the same depth. Therefore, only large objects had shown results in the chart.

For further improvement of the model, more annotated data are needed to identify products effectively. Furthermore, the training steps and batch size can also be optimized with more powerful GPU and RAMs.


<img src="https://user-images.githubusercontent.com/77212888/128619600-a5e7db8f-4903-4b87-8c37-e88745e3a343.png" width="75%" height="75%">

In figure 3, the men’s wallet was successfully detected with a precise bounding box. However, the edge of the keyboard on the right was not detected. While objects from other categories are being detected as well, the position and lighting need to be similar enough in order for the machine to reorganize the object, which is due to the insufficient instances and fewer training steps.

<img src="https://user-images.githubusercontent.com/77212888/128619607-dfbffde6-a13a-4f18-9fac-4d22f600cb5a.png" width="50%" height="50%">

In further steps of this project to be deployed on the online marketplace application, the product images that are already in the database can be leveraged along with the images that are already classified by the users as a part of the information for their product when listing. The only requirement is to annotate the bounding box for training. Plenty of APIs to really connect the link between the model and the application should also be considered as the next crucial step. Moreover, scripts on getting the product information from the database and calculating the recommended price (it can be a regression problem) or average price should also be considered in the development process. 

To summarize, we have obtained the proof of concept for the further development of the object detection mode, where it detects the object with precise bounding boxes even with minimum data points and minimum training. Leveraging the online marketplace application database and increasing training will strengthen the model. Furthermore, with the speed and mAP trade-off, we concluded that the CenterNet HourGlass104 has the best result with good speed.


Initial Performance        |  Improved Performance 
:-------------------------:|:-------------------------:
![ckpt-1 (1)](https://user-images.githubusercontent.com/77212888/128609076-dd7ff9a6-4470-4a12-bc3a-42c59075c1b4.gif)    |  ![ckpt-3](https://user-images.githubusercontent.com/77212888/128608971-e4af1c9f-2a8b-49ab-b2ab-40fb1bd4cada.gif)


