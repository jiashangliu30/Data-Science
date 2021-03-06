# Linear Regression: Real Estate House Prices Analysis & Prediction

## 1. Problem & Goals
Recently, Housing Market in Canada is more volatile due to Covid. There are more people looking for homes and condos, and prices continue to rise. However, there are plenty of instances of housing being more expensive than its market value. Therefore, It is beneficial to take a deep dive into a pipeline of housing price prediction from analyzing the selected features of a house for better home buying experience.

Real Estate market prices are impacted by various factors, to name a few, how old is the house, and how convenient it is to nearby stores. It is efficient for us to quickly evaluate the housing prices for agencies or compare Price/performance ratio for home buyers.

The Goal here is to explore the relationships between each factors in the dataset and train the model to predict the housing prices from the features

## 2. Dataset
The dataset is from the UCI Machine Learning Repository https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set. It is real data taken from New Taipei City, Taiwan, the original owner is Prof. I-Cheng Yeh. I used this dataset to test the example of correlation of each features and illustrate the underlying ideas. This pipeline could be applied to any current housing market to create an finetuned model according to the local market such as Toronto market.

The dataset includes 6 features "X", and labels "y" as the price of houses per unit area to better assess the price of a house

Features:

- Transaction date
- House age
- Distance to the nearest MRT station
- number of convenience stores
- latitude
- longitude

## 3. Data Preprocessing
This dataset is not cleaning heavy as there aren't many instances and features. However, data preprocessing should including checking and eliminating nans.

## 4. Data Exploration & Pattern Identification
Looking at data for patterns and comprehend the data before any training happens to get a deeper layer of understanding of the data or the feature fed in to the model.

In this case, we will investigate how each feature relates to the house prices and set an assuption before the training and prediction happens

- Transaction Date Vs Price

![image](https://user-images.githubusercontent.com/77212888/122659403-f3503280-d145-11eb-9a61-d8b7ff5c2d13.png)

- House Age Vs Price

![image](https://user-images.githubusercontent.com/77212888/122659421-137ff180-d146-11eb-9988-f77cbb38e994.png)

- Distance to MRT station Vs Price

![image](https://user-images.githubusercontent.com/77212888/122659462-796c7900-d146-11eb-9472-852ff243c523.png)

- Number of Convienience Stores Vs Price

![image](https://user-images.githubusercontent.com/77212888/122659477-91dc9380-d146-11eb-9edf-f307571cea83.png)

## 5. Result & Conclusion

The Metrics for regression models r2 score, in another word "Coefficient of determination", it measure how good y_pred fit with y_test
where accuracy is not suitable for any regression problems.
