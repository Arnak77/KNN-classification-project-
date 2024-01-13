#!/usr/bin/env python
# coding: utf-8

# <a class="anchor" id="0"></a>
# # **kNN Classifier Tutorial in Python**
# 
# 
# Hello friends,
# 
# kNN or k-Nearest Neighbours Classifier is a very simple and easy to understand machine learning algorithm. In this kernel, I build a k Nearest Neighbours classifier to classify the patients suffering from Breast Cancer.
# 
# So, let's get started.
# 

Table of Contents


1. [Introduction to k Nearest Neighbours Algorithm]
2. [k Nearest Neighbours intuition]
3. [How to decide the number of neighbours in kNN]
4. [Eager learners vs lazy learners]
5. [Import libraries]
6. [Import dataset]
7. [Exploratory data analysis]
8. [Data visualization]
9. [Declare feature vector and target variable]
10. [Split data into separate training and test set]
11.	[Feature engineering]
12.	[Feature scaling]
13.	[Fit Neighbours classifier to the training set]
14.	[Predict the test-set results]
15.	[Check the accuracy score]
16.	[Rebuild kNN classification model using different values of k]
17.	[Confusion matrix]
18.	[Classification metrices]
19. [Results and Conclusion]

# # **1. Introduction to k Nearest Neighbours algorithm** <a class="anchor" id="1"></a>
# 
# 
# 
# In machine learning, k Nearest Neighbours or kNN is the simplest of all machine learning algorithms. It is a non-parametric algorithm used for classification and regression tasks. Non-parametric means there is no assumption required for data distribution. So, kNN does not require any underlying assumption to be made. In both classification and regression tasks, the input consists of the k closest training examples in the feature space. The output depends upon whether kNN is used for classification or regression purposes.
# 
# -	In KNN classification, the output is a class membership. The given data point is classified based on the majority of type of its neighbours. The data point is assigned to the most frequent class among its k nearest neighbours. Usually k is a small positive integer. If k=1, then the data point is simply assigned to the class of that single nearest neighbour.
# 
# -	In kNN regression, the output is simply some property value for the object. This value is the average of the values of k nearest neighbours.
# 
# 
# kNN is a type of instance-based learning or lazy learning. Lazy learning means it does not require any training data points for model generation. All training data will be used in the testing phase. This makes training faster and testing slower and costlier. So, the testing phase requires more time and memory resources.
# 
# In kNN, the neighbours are taken from a set of objects for which the class or the object property value is known. This can be thought of as the training set for the kNN algorithm, though no explicit training step is required. In both classification and regression kNN algorithm, we can assign weight to the contributions of the neighbours. So, nearest neighbours contribute more to the average than the more distant ones.
# 
# 

# # **2. k Nearest Neighbours intuition** <a class="anchor" id="2"></a>
# 
# [Table of Contents](#0.1)
# 
# The kNN algorithm intuition is very simple to understand. It simply calculates the distance between a sample data point and all the other training data points. The distance can be Euclidean distance or Manhattan distance. Then, it selects the k nearest data points where k can be any integer. Finally, it assigns the sample data point to the class to which the majority of the k data points belong.
# 
# 
# Now, we will see kNN algorithm in action. Suppose, we have a dataset with two variables which are classified as `Red` and `Blue`.
# 
# 
# In kNN algorithm, k is the number of nearest neighbours. Generally, k is an odd number because it helps to decide the majority of the class. When k=1, then the algorithm is known as the nearest neighbour algorithm.
# 
# Now, we want to classify a new data point `X` into `Blue` class or `Red` class. Suppose the value of k is 3. The kNN algorithm starts by calculating the distance between `X` and all the other data points. It then finds the 3 nearest points with least distance to point `X`. 
# 
# 
# In the final step of the kNN algorithm, we assign the new data point `X` to the majority of the class of the 3 nearest points. If 2 of the 3 nearest points belong to the class `Red` while 1 belong to the class `Blue`, then we classify the new data point  as `Red`.
# 

# # **3. How to decide the number of neighbours in kNN** <a class="anchor" id="3"></a>
# 
# 
# 
# 
# While building the kNN classifier model, one question that come to my mind is what should be the value of nearest neighbours (k) that yields highest accuracy. This is a very important question because the classification accuracy depends upon our choice of k.
# 
# The number of neighbours (k) in kNN is a parameter that we need to select at the time of model building. Selecting the optimal value of k in kNN is the most critical problem. A small value of k means that noise will have higher influence on the result. So, probability of overfitting is very high. A large value of k makes it computationally expensive in terms of time to build the kNN model. Also, a large value of k will have a smoother decision boundary which means lower variance but higher bias.
# 
# The data scientists choose an odd value of k if the number of classes is even. We can apply the elbow method to select the value of k. To optimize the results, we can use Cross Validation technique. Using the cross-validation technique, we can test the kNN algorithm with different values of k. The model which gives good accuracy can be considered to be an optimal choice. It depends on individual cases and at times best process is to run through each possible value of k and test our result.

# # **4. Eager learners vs lazy learners** <a class="anchor" id="4"></a>
# 
# 
# Eager learners mean when giving training data points, we will construct a generalized model before performing prediction on given new points to classify. We can think of such learners as being ready, active and eager to classify new data points.
# 
# Lazy learning means there is no need for learning or training of the model and all of the data points are used at the time of prediction. Lazy learners wait until the last minute before classifying any data point. They merely store the training dataset and waits until classification needs to perform. Lazy learners are also known as instance-based learners because lazy learners store the training points or instances, and all learning is based on instances.
# 
# Unlike eager learners, lazy learners do less work in the training phase and more work in the testing phase to make a classification.

# # **5. Import libraries** <a class="anchor" id="5"></a>
# 

# In[129]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization purposes
import seaborn as sns # for data visualization
get_ipython().run_line_magic('matplotlib', 'inline')



# In[130]:


import warnings

warnings.filterwarnings('ignore')


# # **6. Import dataset** <a class="anchor" id="6"></a>
# 
# 

# In[131]:


df=pd.read_csv(r"D:\NIT\JANUARY\4 JAN(knn classification)\4th\projects\KNN\brest cancer.txt")


# # **7. Exploratory data analysis** <a class="anchor" id="7"></a>
# 
# 
# 
# 
# Now, I will explore the data to gain insights about the data. 

# In[132]:


# view dimensions of dataset

df.shape


# We can see that there are 699 instances and 11 attributes in the data set. 
# 
# 
# In the dataset description, it is given that there are 10 attributes and 1 `Class` which is the target variable. So, we have 10 attributes and 1 target variable.

# ### View top 5 rows of dataset

# In[133]:


# preview the dataset

df.head()


# ### Rename column names
# 
# We can see that the dataset does not have proper column names. The columns are merely labelled as 0,1,2.... and so on. We should give proper names to the columns. I will do it as follows:-

# In[134]:


col_names = ['Id', 'Clump_thickness', 'Uniformity_Cell_Size', 'Uniformity_Cell_Shape', 'Marginal_Adhesion', 
             'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class']

df.columns = col_names

df.columns


# We can see that the column names are renamed. Now, the columns have meaningful names.

# In[135]:


# let's agian preview the dataset

df.head()


# ### Drop redundant columns
# 
# 
# We should drop any redundant columns from the dataset which does not have any predictive power. Here, `Id` is the redundant column. So, I will drop it first.

# In[136]:


# drop Id column from dataset

df.drop('Id', axis=1, inplace=True)


# ### View summary of dataset
# 

# In[137]:


# view summary of dataset

df.info()


# We can see that the `Id` column has been removed from the dataset. 
# 
# We can see that there are 9 numerical variables and 1 categorical variable in the dataset. I will check the frequency distribution of values in the variables to confirm the same.

# ### Frequency distribution of values in variables

# In[138]:


for var in df.columns:
    
    print(df[var].value_counts())


# The distribution of values shows that data type of `Bare_Nuclei` is of type integer. But the summary of the dataframe shows that it is type object. So, I will explicitly convert its data type to integer.

# ### Convert data type of Bare_Nuclei to integer

# In[139]:


df.dtypes


# In[140]:


df['Bare_Nuclei'] = pd.to_numeric(df['Bare_Nuclei'], errors='coerce')


# ### Check data types of columns of dataframe

# In[141]:


df.dtypes


# Now, we can see that all the columns of the dataframe are of type numeric.

# ### Summary of variables
# 
# 
# - There are 10 numerical variables in the dataset.
# 
# 
# - All of the variables are of discrete type.
# 
# 
# - Out of all the 10 variables, the first 9 variables are feature variables and last variable `Class` is the target variable.
# 
# 
# 

# ### Explore problems within variables
# 
# 
# Now, I will explore problems within variables.
# 

# ### Missing values in variables

# In[142]:


# check missing values in variables

df.isnull().sum()


# We can see that the `Bare_Nuclei` column contains missing values. We need to dig deeper to find the frequency distribution of 
# values of `Bare_Nuclei`.

# In[143]:


# check `na` values in the dataframe

df.isna().sum()


# We can see that the `Bare_Nuclei` column contains 16 `nan` values.

# In[144]:


# check frequency distribution of `Bare_Nuclei` column

df['Bare_Nuclei'].value_counts()


# In[145]:


# check unique values in `Bare_Nuclei` column

df['Bare_Nuclei'].unique()


# We can see that there are `nan` values in the `Bare_Nuclei` column.

# In[146]:


# check for nan values in `Bare_Nuclei` column

df['Bare_Nuclei'].isna().sum()


# We can see that there are 16 `nan` values in the dataset. I will impute missing values after dividing the dataset into training and test set.

# ### check frequency distribution of target variable `Class`

# In[147]:


# view frequency distribution of values in `Class` variable

df['Class'].value_counts()


# ### check  percentage of frequency distribution of `Class`

# In[148]:


# view percentage of frequency distribution of values in `Class` variable

df['Class'].value_counts()/np.float64(len(df))


# We can see that the `Class` variable contains 2 class labels - `2` and `4`. `2` stands for benign and `4` stands for malignant cancer.

# ### Outliers in numerical variables

# In[149]:


# view summary statistics in numerical variables

print(round(df.describe(),2))


# kNN algorithm is robust to outliers.

# # **8. Data Visualization** <a class="anchor" id="8"></a>
# 
# 
# 
# Now, we have a basic understanding of our data. I will supplement it with some data visualization to get better understanding
# of our data.

# ### Univariate plots

# ### Check the distribution of variables
# 
# 
# Now, I will plot the histograms to check variable distributions to find out if they are normal or skewed. 

# In[150]:


# plot histograms of the variables
# plot histograms of the variables
plt.rcParams["figure.figsize"]=(30,25)
df.plot(kind='hist',bins=10,subplots=True,layout=(5,2))


# We can see that all the variables in the dataset are positively skewed. 

# ### Multivariate plots

# ### Estimating correlation coefficients
# 
# Our dataset is very small. So, we can compute the standard correlation coefficient (also called Pearson's r) between every pair of attributes. We can compute it using the `df.corr()` method as follows:-

# In[151]:


correlation = df.corr()


# Our target variable is `Class`. So, we should check how each attribute correlates with the `Class` variable. We can do it as follows:-

# In[152]:


correlation['Class'].sort_values


# In[153]:


correlation['Class'].sort_values(ascending=False)


# ### Interpretation 
# 
# - The correlation coefficient ranges from -1 to +1. 
# 
# - When it is close to +1, this signifies that there is a strong positive correlation. So, we can see that there is a strong positive correlation between `Class` and `Bare_Nuclei`, `Class` and `Uniformity_Cell_Shape`, `Class` and `Uniformity_Cell_Size`.
# 
# - When it is clsoe to -1, it means that there is a strong negative correlation. When it is close to 0, it means that there is no correlation. 
# 
# - We can see that all the variables are positively correlated with `Class` variable. Some variables are strongly positive correlated while some variables are negatively correlated.

# ### Discover patterns and relationships 
# 
# 
# An important step in EDA is to discover patterns and relationships between variables in the dataset. I will use the seaborn heatmap to explore the patterns and relationships in the dataset.
# 

# ### Correlation Heat Map

# In[154]:


plt.figure(figsize=(10,8))
plt.title("(Corr of Attributes with Class variable)")
H=sns.heatmap(correlation,square=True,annot=True)
          
plt.show()


# ### Interpretation
# 
# 
# From the above correlation heat map, we can conclude that :-
# 
# 1. `Class` is highly positive correlated with `Uniformity_Cell_Size`, `Uniformity_Cell_Shape` and `Bare_Nuclei`. (correlation coefficient = 0.82).
# 
# 2. `Class` is positively correlated with `Clump_thickness`(correlation coefficient=0.72), `Marginal_Adhesion`(correlation coefficient=0.70), `Single_Epithelial_Cell_Size)`(correlation coefficient = 0.68) and `Normal_Nucleoli`(correlation coefficient=0.71).
# 
# 3. `Class` is weekly positive correlated with `Mitoses`(correlation coefficient=0.42).
# 
# 4. The `Mitoses` variable is weekly positive correlated with all the other variables(correlation coefficient < 0.50).

# # Engineering missing values in variables

# In[155]:


df['Bare_Nuclei'].unique()


# In[156]:


df['Bare_Nuclei']=df['Bare_Nuclei'].fillna(np.mean(pd.to_numeric(df['Bare_Nuclei'])))


# In[157]:


df.isnull().sum()


# # **9. Declare feature vector and target variable** <a class="anchor" id="9"></a>
# 
# 

# In[158]:


X = df.drop(['Class'], axis=1)

y = df['Class']


# # **10. Split data into separate training and test set** <a class="anchor" id="10"></a>
# 
# 

# In[159]:


# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[160]:


# check the shape of X_train and X_test

X_train.shape, X_test.shape


# # **11. Feature Engineering** <a class="anchor" id="11"></a>
# 
# 
# 
# 
# **Feature Engineering** is the process of transforming raw data into useful features that help us to understand our model better and increase its predictive power. I will carry out feature engineering on different types of variables.
# 

# In[161]:


# check data types in X_train

X_train.dtypes


# In[162]:


df.isnull().sum()


# In[163]:


# check missing values in numerical variables in X_train

X_train.isnull().sum()


# In[164]:


# check missing values in numerical variables in X_test

X_test.isnull().sum()


# We can see that there are no missing values in X_train and X_test.

# In[165]:


X_train.head()


# In[166]:


X_test.head()


# We now have training and testing set ready for model building. Before that, we should map all the feature variables onto the same scale. It is called `feature scaling`. I will do it as follows.

# # **12. Feature Scaling** <a class="anchor" id="12"></a>
# 
# 

# In[167]:


cols = X_train.columns


# In[168]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# In[169]:


X_train = pd.DataFrame(X_train, columns=[cols])


# In[170]:


X_test = pd.DataFrame(X_test, columns=[cols])


# In[171]:


X_train.head()


# We now have `X_train` dataset ready to be fed into the Logistic Regression classifier. I will do it as follows.

# # **13. Fit K Neighbours Classifier to the training eet** <a class="anchor" id="13"></a>
# 
# 

# In[172]:


# import KNeighbors ClaSSifier from sklearn
from sklearn.neighbors import KNeighborsClassifier

# instantiate the model
knn = KNeighborsClassifier(n_neighbors=3)

# fit the model to the training set
knn.fit(X_train, y_train)


# # **14. Predict test-set results** <a class="anchor" id="14"></a>
# 
# 

# In[173]:


y_pred = knn.predict(X_test)

y_pred


# ### predict_proba method
# 
# 
# **predict_proba** method gives the probabilities for the target variable(2 and 4) in this case, in array form.
# 
# `2 is for probability of benign cancer` and `4 is for probability of malignant cancer.`

# In[174]:


# probability of getting output as 2 - benign cancer

knn.predict_proba(X_test)[:,0]


# In[175]:


# probability of getting output as 4 - malignant cancer

knn.predict_proba(X_test)[:,1]


# # **15. Check accuracy score** <a class="anchor" id="15"></a>
# 
# 

# In[178]:


from sklearn.metrics import accuracy_score 
ac = accuracy_score(y_test, y_pred)
print(ac) 
print('Model accuracy score:',ac)


# Here, **y_test** are the true class labels and **y_pred** are the predicted class labels in the test-set.

# ### Compare the train-set and test-set accuracy
# 
# 
# Now, I will compare the train-set and test-set accuracy to check for overfitting.

# ### Check for overfitting and underfitting

# In[182]:


bias = knn.score(X_train, y_train)
bias

variance = knn.score(X_test, y_test)
variance


# In[184]:


print(bias)
print(variance)


# The training-set accuracy score is 0.978494 while the test-set accuracy to be 0.964285 These two values are quite comparable. So, there is no question of overfitting. 
# 

# ### Compare model accuracy with null accuracy
# 
# 
# So, the model accuracy is 0.978494 But, we cannot say that our model is very good based on the above accuracy. We must compare it with the **null accuracy**. Null accuracy is the accuracy that could be achieved by always predicting the most frequent class.
# 
# So, we should first check the class distribution in the test set. 

# In[185]:


# check class distribution in test set

y_test.value_counts()


# We can see that the occurences of most frequent class is 85. So, we can calculate null accuracy by dividing 85 by total number of occurences.

# In[188]:


# check null accuracy score

null_accuracy = (85/(85+55))


print('Null accuracy score:',null_accuracy)


# We can see that our model accuracy score is 0.978494  but null accuracy score is 0.6071. So, we can conclude that our K Nearest Neighbors model is doing a very good job in predicting the class labels.

# # **16. Rebuild kNN Classification model using different values of k** <a class="anchor" id="16"></a>
# 
# 
# 
# I have build the kNN classification model using k=3. Now, I will increase the value of k and see its effect on accuracy.

# ### Rebuild kNN Classification model using k=5

# In[189]:


# instantiate the model with k=5
knn_5 = KNeighborsClassifier(n_neighbors=5)


# fit the model to the training set
knn_5.fit(X_train, y_train)


# predict on the test-set
y_pred_5 = knn_5.predict(X_test)


print('Model accuracy score with k=5 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_5)))


# ### Rebuild kNN Classification model using k=6

# In[190]:


# instantiate the model with k=6
knn_6 = KNeighborsClassifier(n_neighbors=6)


# fit the model to the training set
knn_6.fit(X_train, y_train)


# predict on the test-set
y_pred_6 = knn_6.predict(X_test)


print('Model accuracy score with k=6 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_6)))


# ### Rebuild kNN Classification model using k=7

# In[191]:


# instantiate the model with k=7
knn_7 = KNeighborsClassifier(n_neighbors=7)


# fit the model to the training set
knn_7.fit(X_train, y_train)


# predict on the test-set
y_pred_7 = knn_7.predict(X_test)


print('Model accuracy score with k=7 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_7)))


# ### Rebuild kNN Classification model using k=8

# In[192]:


# instantiate the model with k=8
knn_8 = KNeighborsClassifier(n_neighbors=8)


# fit the model to the training set
knn_8.fit(X_train, y_train)


# predict on the test-set
y_pred_8 = knn_8.predict(X_test)


print('Model accuracy score with k=8 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_8)))


# ### Rebuild kNN Classification model using k=9

# In[193]:


# instantiate the model with k=9
knn_9 = KNeighborsClassifier(n_neighbors=9)


# fit the model to the training set
knn_9.fit(X_train, y_train)


# predict on the test-set
y_pred_9 = knn_9.predict(X_test)


print('Model accuracy score with k=9 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_9)))


# Now, based on the above analysis we can conclude that our classification model accuracy is very good. Our model is doing a very good job in terms of predicting the class labels.
# 
# 
# But, it does not give the underlying distribution of values. Also, it does not tell anything about the type of errors our classifer is making. 
# 
# 
# We have another tool called `Confusion matrix` that comes to our rescue.

# # **17. Confusion matrix** <a class="anchor" id="17"></a>
# 
# 
# 
# 
# A confusion matrix is a tool for summarizing the performance of a classification algorithm. A confusion matrix will give us a clear picture of classification model performance and the types of errors produced by the model. It gives us a summary of correct and incorrect predictions broken down by each category. The summary is represented in a tabular form.
# 
# 
# Four types of outcomes are possible while evaluating a classification model performance. These four outcomes are described below:-
# 
# 
# **True Positives (TP)** – True Positives occur when we predict an observation belongs to a certain class and the observation actually belongs to that class.
# 
# 
# **True Negatives (TN)** – True Negatives occur when we predict an observation does not belong to a certain class and the observation actually does not belong to that class.
# 
# 
# **False Positives (FP)** – False Positives occur when we predict an observation belongs to a    certain class but the observation actually does not belong to that class. This type of error is called **Type I error.**
# 
# 
# 
# **False Negatives (FN)** – False Negatives occur when we predict an observation does not belong to a certain class but the observation actually belongs to that class. This is a very serious error and it is called **Type II error.**
# 
# 
# 
# These four outcomes are summarized in a confusion matrix given below.
# 

# In[194]:


# Print the Confusion Matrix with k =3 and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])


# The confusion matrix shows `83 + 53 = 136 correct predictions` and `2 + 2 = 4 incorrect predictions`.
# 
# 
# In this case, we have
# 
# 
# - `True Positives` (Actual Positive:1 and Predict Positive:1) - 83
# 
# 
# - `True Negatives` (Actual Negative:0 and Predict Negative:0) - 52
# 
# 
# - `False Positives` (Actual Negative:0 but Predict Positive:1) - 2 `(Type I error)`
# 
# 
# - `False Negatives` (Actual Positive:1 but Predict Negative:0) - 3 `(Type II error)`

# In[195]:


# Print the Confusion Matrix with k =7 and slice it into four pieces

cm_7 = confusion_matrix(y_test, y_pred_7)

print('Confusion matrix\n\n', cm_7)

print('\nTrue Positives(TP) = ', cm_7[0,0])

print('\nTrue Negatives(TN) = ', cm_7[1,1])

print('\nFalse Positives(FP) = ', cm_7[0,1])

print('\nFalse Negatives(FN) = ', cm_7[1,0])


# The above confusion matrix shows `83 + 54 = 137 correct predictions` and `2 + 1 = 4 incorrect predictions`.
# 
# 
# In this case, we have
# 
# 
# - `True Positives` (Actual Positive:1 and Predict Positive:1) - 83
# 
# 
# - `True Negatives` (Actual Negative:0 and Predict Negative:0) - 54
# 
# 
# - `False Positives` (Actual Negative:0 but Predict Positive:1) - 2 `(Type I error)`
# 
# 
# - `False Negatives` (Actual Positive:1 but Predict Negative:0) - 1 `(Type II error)`

# ### Comment
# 
# 
# So, kNN Classification model with k=7 shows more accurate predictions and less number of errors than k=3 model. Hence, we got performance improvement with k=7.

# In[196]:


# visualize confusion matrix with seaborn heatmap

plt.figure(figsize=(6,4))

cm_matrix = pd.DataFrame(data=cm_7, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


# # **18. Classification metrices** <a class="anchor" id="18"></a>
# 
# 

# ### Classification Report
# 
# 
# **Classification report** is another way to evaluate the classification model performance. It displays the  **precision**, **recall**, **f1** and **support** scores for the model. I have described these terms in later.
# 
# We can print a classification report as follows:-

# In[197]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_7))


# ### Classification accuracy

# In[198]:


TP = cm_7[0,0]
TN = cm_7[1,1]
FP = cm_7[0,1]
FN = cm_7[1,0]


# In[199]:


# print classification accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))


# ### Classification error

# In[200]:


# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))


# ### Precision
# 
# 
# **Precision** can be defined as the percentage of correctly predicted positive outcomes out of all the predicted positive outcomes. It can be given as the ratio of true positives (TP) to the sum of true and false positives (TP + FP). 
# 
# 
# So, **Precision** identifies the proportion of correctly predicted positive outcome. It is more concerned with the positive class than the negative class.
# 
# 
# 
# Mathematically, `precision` can be defined as the ratio of `TP to (TP + FP)`.
# 

# In[201]:


# print precision score

precision = TP / float(TP + FP)


print('Precision : {0:0.4f}'.format(precision))


# ### Recall
# 
# 
# Recall can be defined as the percentage of correctly predicted positive outcomes out of all the actual positive outcomes.
# It can be given as the ratio of true positives (TP) to the sum of true positives and false negatives (TP + FN). **Recall** is also called **Sensitivity**.
# 
# 
# **Recall** identifies the proportion of correctly predicted actual positives.
# 
# 
# Mathematically, `recall` can be given as the ratio of `TP to (TP + FN)`.
# 
# 

# In[202]:


recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))


# ### True Positive Rate
# 
# 
# **True Positive Rate** is synonymous with **Recall**.
# 

# In[203]:


true_positive_rate = TP / float(TP + FN)


print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))


# ### False Positive Rate

# In[204]:


false_positive_rate = FP / float(FP + TN)


print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))


# ### Specificity

# In[205]:


specificity = TN / (TN + FP)

print('Specificity : {0:0.4f}'.format(specificity))


# ### f1-score
# 
# 
# **f1-score** is the weighted harmonic mean of precision and recall. The best possible **f1-score** would be 1.0 and the worst 
# would be 0.0.  **f1-score** is the harmonic mean of precision and recall. So, **f1-score** is always lower than accuracy measures as they embed precision and recall into their computation. The weighted average of `f1-score` should be used to 
# compare classifier models, not global accuracy.
# 

# ### Support
# 
# 
# **Support** is the actual number of occurrences of the class in our dataset.

# ### Adjusting the classification threshold level

# In[206]:


# print the first 10 predicted probabilities of two classes- 2 and 4

y_pred_prob = knn.predict_proba(X_test)[0:10]

y_pred_prob


# ### Observations
# 
# 
# - In each row, the numbers sum to 1.
# 
# 
# - There are 2 columns which correspond to 2 classes - 2 and 4. 
# 
# 
#     - Class 2 - predicted probability that there is benign cancer.    
#     
#     - Class 4 - predicted probability that there is malignant cancer.
#         
#     
# - Importance of predicted probabilities
# 
#     - We can rank the observations by probability of benign or malignant cancer.
# 
# 
# - predict_proba process
# 
#     - Predicts the probabilities    
#     
#     - Choose the class with the highest probability    
#     
#     
# - Classification threshold level
# 
#     - There is a classification threshold level of 0.5.    
#     
#     - Class 4 - probability of malignant cancer is predicted if probability > 0.5.    
#     
#     - Class 2 - probability of benign cancer is predicted if probability < 0.5.    
#     
# 

# In[207]:


# store the probabilities in dataframe

y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - benign cancer (2)', 'Prob of - malignant cancer (4)'])

y_pred_prob_df


# In[208]:


# print the first 10 predicted probabilities for class 4 - Probability of malignant cancer

knn.predict_proba(X_test)[0:10, 1]


# In[209]:


# store the predicted probabilities for class 4 - Probability of malignant cancer

y_pred_1 = knn.predict_proba(X_test)[:, 1]


# In[210]:


# plot histogram of predicted probabilities


# adjust figure size
plt.figure(figsize=(6,4))


# adjust the font size 
plt.rcParams['font.size'] = 12


# plot histogram with 10 bins
plt.hist(y_pred_1, bins = 10)


# set the title of predicted probabilities
plt.title('Histogram of predicted probabilities of malignant cancer')


# set the x-axis limit
plt.xlim(0,1)


# set the title
plt.xlabel('Predicted probabilities of malignant cancer')
plt.ylabel('Frequency')


# ### Observations
# 
# 
# - We can see that the above histogram is positively skewed.
# 
# 
# - The first column tell us that there are approximately 80 observations with 0 probability of malignant cancer.
# 
# 
# - There are few observations with probability > 0.5.
# 
# 
# - So, these few observations predict that there will be malignant cancer.
# 

# ### Comments
# 
# 
# - In binary problems, the threshold of 0.5 is used by default to convert predicted probabilities into class predictions.
# 
# 
# - Threshold can be adjusted to increase sensitivity or specificity. 
# 
# 
# - Sensitivity and specificity have an inverse relationship. Increasing one would always decrease the other and vice versa.
# 
# 
# - Adjusting the threshold level should be one of the last step you do in the model-building process.

# # 19. Results and Conclusion
# 
# 
# 
# 
# 
# 1. In this project, I build a kNN classifier model to classify the patients suffering from breast cancer. The model yields very good performance as indicated by the model accuracy which was found to be 0.9571 with k=7.
# 
# 2. With k=3, the training-set accuracy score is 0.9784 while the test-set accuracy to be 0.9642. These two values are quite comparable. So, there is no question of overfitting. 
# 
# 3. I have compared the model accuracy score which is 0.9784 with null accuracy score which is 0.6071. So, we can conclude that our K Nearest Neighbors model is doing a very good job in predicting the class labels.
# 
# 
# 
# 4. kNN Classification model with k=7 shows more accurate predictions and less number of errors than k=3 model. Hence, we got performance improvement with k=7.
# 
# 
# 

# In[ ]:




