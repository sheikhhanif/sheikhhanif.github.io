---
title: "Building Scikit-Learn Linear Model: Elasticnet"
header:
  teaser: /assets/images/elasticnet/output_14_1.png
date: 2019-01-19
tags: [scikit-learn, linear model]
excerpt: "In this notebook we will learn how to use scikit learn to build best 
linear regressor model. We used gapminder dataset which is related to 
population in different region and average life expectency. More about our 
dataset you will see below. We split our datasets into train and test set.
We build our model using scikit learn elasticnet linear model. We evaluate 
our model using test set. We used gridsearch cross validation to optimized our
model."
---
#### Introduction  [Go to Repository](https://github.com/sheikhhanif/ElasticNet-Scikit-Learn.git){:target="_blank"}
In this notebook we will learn how to use scikit learn to build best 
linear regressor model. We used gapminder dataset which is related to 
population in different region and average life expectency. More about our 
dataset you will see below. We split our datasets into train and test set.
We build our model using scikit learn elasticnet linear model. We evaluate 
our model using test set. We used gridsearch cross validation to optimized our
model.

### Importing Required Packages


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
```

## Reading the data in


```python
data = pd.read_csv('data/gm_2008_region.csv')
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>population</th>
      <th>fertility</th>
      <th>HIV</th>
      <th>CO2</th>
      <th>BMI_male</th>
      <th>GDP</th>
      <th>BMI_female</th>
      <th>life</th>
      <th>child_mortality</th>
      <th>Region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>34811059.0</td>
      <td>2.73</td>
      <td>0.1</td>
      <td>3.328945</td>
      <td>24.59620</td>
      <td>12314.0</td>
      <td>129.9049</td>
      <td>75.3</td>
      <td>29.5</td>
      <td>Middle East &amp; North Africa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>19842251.0</td>
      <td>6.43</td>
      <td>2.0</td>
      <td>1.474353</td>
      <td>22.25083</td>
      <td>7103.0</td>
      <td>130.1247</td>
      <td>58.3</td>
      <td>192.0</td>
      <td>Sub-Saharan Africa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>40381860.0</td>
      <td>2.24</td>
      <td>0.5</td>
      <td>4.785170</td>
      <td>27.50170</td>
      <td>14646.0</td>
      <td>118.8915</td>
      <td>75.5</td>
      <td>15.4</td>
      <td>America</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2975029.0</td>
      <td>1.40</td>
      <td>0.1</td>
      <td>1.804106</td>
      <td>25.35542</td>
      <td>7383.0</td>
      <td>132.8108</td>
      <td>72.5</td>
      <td>20.0</td>
      <td>Europe &amp; Central Asia</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21370348.0</td>
      <td>1.96</td>
      <td>0.1</td>
      <td>18.016313</td>
      <td>27.56373</td>
      <td>41312.0</td>
      <td>117.3755</td>
      <td>81.5</td>
      <td>5.2</td>
      <td>East Asia &amp; Pacific</td>
    </tr>
  </tbody>
</table>
</div>



### Data Exploration

Before we build our model Let's have a descriptive exploration on our data. We will visualize how each feature is co-related to each other. Also we will visualize some features vs life of our dataset. Visualization will give us better understanding of our data and will help us to select appropriate feature to build our model.


```python
# summerize our data
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>population</th>
      <th>fertility</th>
      <th>HIV</th>
      <th>CO2</th>
      <th>BMI_male</th>
      <th>GDP</th>
      <th>BMI_female</th>
      <th>life</th>
      <th>child_mortality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.390000e+02</td>
      <td>139.000000</td>
      <td>139.000000</td>
      <td>139.000000</td>
      <td>139.000000</td>
      <td>139.000000</td>
      <td>139.000000</td>
      <td>139.000000</td>
      <td>139.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.549977e+07</td>
      <td>3.005108</td>
      <td>1.915612</td>
      <td>4.459874</td>
      <td>24.623054</td>
      <td>16638.784173</td>
      <td>126.701914</td>
      <td>69.602878</td>
      <td>45.097122</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.095121e+08</td>
      <td>1.615354</td>
      <td>4.408974</td>
      <td>6.268349</td>
      <td>2.209368</td>
      <td>19207.299083</td>
      <td>4.471997</td>
      <td>9.122189</td>
      <td>45.724667</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.773150e+05</td>
      <td>1.280000</td>
      <td>0.060000</td>
      <td>0.008618</td>
      <td>20.397420</td>
      <td>588.000000</td>
      <td>117.375500</td>
      <td>45.200000</td>
      <td>2.700000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.752776e+06</td>
      <td>1.810000</td>
      <td>0.100000</td>
      <td>0.496190</td>
      <td>22.448135</td>
      <td>2899.000000</td>
      <td>123.232200</td>
      <td>62.200000</td>
      <td>8.100000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9.705130e+06</td>
      <td>2.410000</td>
      <td>0.400000</td>
      <td>2.223796</td>
      <td>25.156990</td>
      <td>9938.000000</td>
      <td>126.519600</td>
      <td>72.000000</td>
      <td>24.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.791973e+07</td>
      <td>4.095000</td>
      <td>1.300000</td>
      <td>6.589156</td>
      <td>26.497575</td>
      <td>23278.500000</td>
      <td>130.275900</td>
      <td>76.850000</td>
      <td>74.200000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.197070e+09</td>
      <td>7.590000</td>
      <td>25.900000</td>
      <td>48.702062</td>
      <td>28.456980</td>
      <td>126076.000000</td>
      <td>135.492000</td>
      <td>82.600000</td>
      <td>192.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Let's select some of the features to explore more
chosen_data = data[['population', 'fertility', 'HIV', 'CO2', 'GDP', 'child_mortality', 'life']]
chosen_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>population</th>
      <th>fertility</th>
      <th>HIV</th>
      <th>CO2</th>
      <th>GDP</th>
      <th>child_mortality</th>
      <th>life</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>34811059.0</td>
      <td>2.73</td>
      <td>0.1</td>
      <td>3.328945</td>
      <td>12314.0</td>
      <td>29.5</td>
      <td>75.3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>19842251.0</td>
      <td>6.43</td>
      <td>2.0</td>
      <td>1.474353</td>
      <td>7103.0</td>
      <td>192.0</td>
      <td>58.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>40381860.0</td>
      <td>2.24</td>
      <td>0.5</td>
      <td>4.785170</td>
      <td>14646.0</td>
      <td>15.4</td>
      <td>75.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2975029.0</td>
      <td>1.40</td>
      <td>0.1</td>
      <td>1.804106</td>
      <td>7383.0</td>
      <td>20.0</td>
      <td>72.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21370348.0</td>
      <td>1.96</td>
      <td>0.1</td>
      <td>18.016313</td>
      <td>41312.0</td>
      <td>5.2</td>
      <td>81.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# We can plot each of these features 
chosen_data.hist()
plt.show()
```


![png](/assets/images/elasticnet/output_8_0.png)


Let's now plot each of these features vs life, to their relation


```python
plt.scatter(chosen_data.HIV, chosen_data.life)
plt.xlabel('HIV')
plt.ylabel('Life')
plt.show()
```


![png](/assets/images/elasticnet/output_10_0.png)



```python
plt.scatter(chosen_data.GDP, chosen_data.life)
plt.xlabel('GDP')
plt.ylabel('Life')
plt.show()
```


![png](/assets/images/elasticnet/output_11_0.png)



```python
plt.scatter(chosen_data.child_mortality, chosen_data.life)
plt.xlabel('Child Mortality')
plt.ylabel('Life')
plt.show()
```


![png](/assets/images/elasticnet/output_12_0.png)



```python
plt.scatter(chosen_data.CO2, chosen_data.life)
plt.xlabel('CO2')
plt.ylabel('Life')
plt.show()
```


![png](/assets/images/elasticnet/output_13_0.png)



```python
sns.heatmap(data.corr(), square=True, cmap='RdYlGn')
```









![png](/assets/images/elasticnet/output_14_1.png)


### Data preprocessing
We will be using scikit learn to build our model as I mentioned earlier. We have to make sure that there is/are no missing value exist in our dataset. Also we we have to make sure all of our features data are numerical. As categorical data will not be entertained by scikit learn model. In the following we address and fix such issues. 


```python
# dertermining any missing value
data.count()
```




    population         139
    fertility          139
    HIV                139
    CO2                139
    BMI_male           139
    GDP                139
    BMI_female         139
    life               139
    child_mortality    139
    Region             139
    dtype: int64




```python
# determining andy null value
data.isna().sum()
```




    population         0
    fertility          0
    HIV                0
    CO2                0
    BMI_male           0
    GDP                0
    BMI_female         0
    life               0
    child_mortality    0
    Region             0
    dtype: int64




```python
# determing categorical value
data.Region.head()
```




    0    Middle East & North Africa
    1            Sub-Saharan Africa
    2                       America
    3         Europe & Central Asia
    4           East Asia & Pacific
    Name: Region, dtype: object




```python
# Converting categorical to numerical 
data = pd.get_dummies(data, drop_first=True)
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>population</th>
      <th>fertility</th>
      <th>HIV</th>
      <th>CO2</th>
      <th>BMI_male</th>
      <th>GDP</th>
      <th>BMI_female</th>
      <th>life</th>
      <th>child_mortality</th>
      <th>Region_East Asia &amp; Pacific</th>
      <th>Region_Europe &amp; Central Asia</th>
      <th>Region_Middle East &amp; North Africa</th>
      <th>Region_South Asia</th>
      <th>Region_Sub-Saharan Africa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>34811059.0</td>
      <td>2.73</td>
      <td>0.1</td>
      <td>3.328945</td>
      <td>24.59620</td>
      <td>12314.0</td>
      <td>129.9049</td>
      <td>75.3</td>
      <td>29.5</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>19842251.0</td>
      <td>6.43</td>
      <td>2.0</td>
      <td>1.474353</td>
      <td>22.25083</td>
      <td>7103.0</td>
      <td>130.1247</td>
      <td>58.3</td>
      <td>192.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>40381860.0</td>
      <td>2.24</td>
      <td>0.5</td>
      <td>4.785170</td>
      <td>27.50170</td>
      <td>14646.0</td>
      <td>118.8915</td>
      <td>75.5</td>
      <td>15.4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2975029.0</td>
      <td>1.40</td>
      <td>0.1</td>
      <td>1.804106</td>
      <td>25.35542</td>
      <td>7383.0</td>
      <td>132.8108</td>
      <td>72.5</td>
      <td>20.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21370348.0</td>
      <td>1.96</td>
      <td>0.1</td>
      <td>18.016313</td>
      <td>27.56373</td>
      <td>41312.0</td>
      <td>117.3755</td>
      <td>81.5</td>
      <td>5.2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



#### Creating train and test dataset
Train/Test Split involves splitting the dataset into training and testing sets respectively, which are mutually exclusive. After which, we will train with the training set and test with the testing set. 
This will provide a more accurate evaluation on out-of-sample accuracy because the testing dataset is not part of the dataset that have been used to train the data. It is more realistic for real world problems.

This means that we know the outcome of each data point in this dataset, making it great to test with! And since this data has not been used to train the model, the model has no knowledge of the outcome of these data points. So, in essence, it is truly an out-of-sample testing.


```python
# Choosing target feature
y = data[['life']]
print(y.head(3))

# Choosing input features
X = data.drop(data[['life']], axis=1)
print(X.head(3))
```

       life
    0  75.3
    1  58.3
    2  75.5
       population  fertility  HIV       CO2  BMI_male      GDP  BMI_female  \
    0  34811059.0       2.73  0.1  3.328945  24.59620  12314.0    129.9049   
    1  19842251.0       6.43  2.0  1.474353  22.25083   7103.0    130.1247   
    2  40381860.0       2.24  0.5  4.785170  27.50170  14646.0    118.8915   
    
       child_mortality  Region_East Asia & Pacific  Region_Europe & Central Asia  \
    0             29.5                           0                             0   
    1            192.0                           0                             0   
    2             15.4                           0                             0   
    
       Region_Middle East & North Africa  Region_South Asia  \
    0                                  1                  0   
    1                                  0                  0   
    2                                  0                  0   
    
       Region_Sub-Saharan Africa  
    0                          0  
    1                          1  
    2                          0  
    


```python
# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state=42)
```

### Building and Evaluating model
As mentioned at the beginning, we are using elasticnet and grid search Cross validation. You can find great explanation in scikit-learn documentation.


```python
# Create the hyperparameter grid
l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio': l1_space}

# Instantiate the ElasticNet regressor: elastic_net
elastic_net = ElasticNet()

# Setup the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)

# Fit it to the training data
gm_cv.fit(X_train, y_train)

# Predict on the test set and compute metrics
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))
```

   


    Tuned ElasticNet l1 ratio: {'l1_ratio': 0.0}
    Tuned ElasticNet R squared: 0.8697529413665848
    Tuned ElasticNet MSE: 9.837193188072188
    


