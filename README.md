# Diabetes data classifications
In this project, I want to use Random Forest and XGBoost to classify diabetes data.
The dataset is available at this [Link](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008), which was uploaded by U of California, Irvine.

## Data Preparation
At the first step, I read the data as follows:
![](Im1.png)
This file has a size of 101766x50, which indicates 101766 data samples with 50 features.

This data includes some irrelevant information, which was saved as "?". We have to transform them into 'Nan' data.
From the description below, we can see the number of "?" cells in the CSV file;

![](Im2.png)

Below, we can see the number of 'Nan' cells in data.

![](Im3.png)

where we can see from 
```python
print((Data.iloc[:2, 3:8]))
```

that "?" in ```python weight```  is another sss  s dsdf sdf df 

