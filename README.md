# Diabetes data classifications
In this project, I want to use **Random Forest** and **XGBoost** to classify diabetes data \textcolor{red}{HHHHH}.
The dataset is available at this [Link](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008), which was uploaded by U of California, Irvine.
The objective in this project is to decide whether a patient will come back to the hospital within 30 days after visiting a specific doctor or not. 
## Data Preparation
At the first step, I read the data as follows:
![](Im1.png)
This file has a size of 101766x50, which indicates 101766 data samples with 50 features.

Based on the data description, we can see that only six features including:
```python
['gender','age','diag_1','diag_2','diag_3','readmitted']
```
are the most important ones, where ```'readmitted'``` is our target feature.

This data includes some irrelevant information, which was saved as "?". We have to transform them into 'Nan' data.
From the description below, we can see the number of "?" cells in the CSV file;

![](Im2.png)

Below, we can see the number of 'Nan' cells in the data.

![](Im3.png)

where we can see from 
```python
print((Data.iloc[:2, 3:8]))
```

where we can see ```"?"``` in ```weight``` is identified as non ```Nan``` cell, therefore, we have to transform all ```"?"``` cells into ```Nan```.
So, we use the following code to transform all ```?``` into ```Nan```:
```python
Data.replace("?",np.nan,inplace=True)
```
## Labeling
In this section, we have to create a label for each string-based data such as the diagnosis features ```diag_1, diag_2```, and ```diag_3```.
This is because we want to develop a classification model. Although the model itself can provide internal labels, we can perform this labeling, providing a mechanism to simplify further data evaluations.
Hence, for ```gender``` we have ```Female=1``` and ```Male=2```, and ```age``` we have ```"[0-10)"=1,...,"[90-100)"=10, and "other"=11```.
We have to do the same thing with ```diagnosis``` and ```readmitted``` features. The diagnosis feature would be encoded with ICD-9 text descriptions, for example ```"infectious"=1``` or ```"pregnancy"=11```. However, ```readmitted``` feature, on the other hand, is categorized by ```“NO”=0```, ```“<30”=1```, and ```“>30”=2```.

## Classification Models
Now, everything is prepared to train the learning models. Here, I used two models for the classification task, including 1) Random Forest and 2) XGBoost. Before defining the models, I define training and testing samples.

Before data splitting, based on the input and output features, we need to associate data, therefore, we have:
```python
Xin = Data[['gender', 'age', 'diag_1', 'diag_2', 'diag_3']]   # input variables
Yout = Data['readmitted']                                     # target
```
### Split data
In this step, we use ```train_test_split``` to split the data as follows:
```python
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xin, Yout, test_size=0.2, random_state=42, stratify=Yout)
```
where ```train_test_split``` is utilized to define training and testing samples. ```test_size=0.2``` means that 80% of the data is used for training and the rest, 20%, is reserved for testing the model. In addition, ```random_state=13``` is used to help us generate the same data in each run, I mean, the data would not be changed by the new run. It is notable to say that ```13``` can be changed to any number.
So, we have prepared data and we can train models.

### Random Forest
As you may know, Random Forest can be simply deployed from ```Scikit-Learn```package from Python. So, we need to install scikit-learng package before use it.
We have:
```python
from sklearn.ensemble import RandomForestClassifier
```

To define the model, we have:
```python
RanForModel_1 = RandomForestClassifier(n_estimators=100, random_state=42)
RanForModel_2 = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
```
where the first one does not use the balance between data and a class with a high population can impose bias in the decision. Accordingly, the second model creates a balance between data, which may increase the accuracy in Out-of-Distribution (OOD) data.

Random Forest models can then be trained by training samples as follows:
```python
RanForModel_1.fit(Xtrain, Ytrain)
RanForModel_2.fit(Xtrain, Ytrain)
```

Finally, we can get the prediction for testing samples as follows:
```python
Ypredicion_1 = RanForModel_1.predict(Xtest)
Ypredicion_2 = RanForModel_2.predict(Xtest)
```
where the accuracy of the model would be reached based on the following step:
```python
Ytest = Ytest.reset_index(drop=True)
Corrects=0
for i in range(len(Ypredicion_1)):
    if Ypredicion_1[i]==Ytest[i]:
        Corrects=Corrects+1
Precision=Corrects/len(Ypredicion_1)
print(f'The precision of the Random Forest is: {round(Precision,4)}')
```

### XGBoost
To use XGBoost, we need to use its package as follow:
```python
import xgboost as xgb
```
Then based on our objective, which is a binary response as ```Yes``` or ```No``` to the question regarding revisiting a patient within 30 days after the first evaluation, we use ```binary``` as our goal in xgboost as follows:
```python
XgbModel_1 = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    scale_pos_weight=(len(Yout) - sum(Yout)) / sum(Yout),
    n_estimators=100,
    random_state=42
)
XgbModel_2 = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    #scale_pos_weight=(len(Yout) - sum(Yout)) / sum(Yout),
    n_estimators=100,
    random_state=42
)
```
then, we can fit the model to training data as follows:
```python
XgbModel_1.fit(Xtrain, Ytrain)
XgbModel_2.fit(Xtrain, Ytrain)
```
Therefore, the prediction and accuracy would be checked as:
```python
Ypredicion_1 = XgbModel_1.predict(Xtest)
Ypredicion_2 = XgbModel_2.predict(Xtest)

Corrects=0
for i in range(len(Ypredicion_1)):
    if Ypredicion_1[i]==Ytest[i]:
        Corrects=Corrects+1
Precision=Corrects/len(Ypredicion_1)
print(f'The precision of the Random Forest is: {round(Precision,4)}')
```




