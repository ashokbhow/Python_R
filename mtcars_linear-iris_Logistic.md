

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
mtcars = pd.read_csv('mtcars.csv')
```


```python
mtcars.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>mpg</th>
      <th>cyl</th>
      <th>disp</th>
      <th>hp</th>
      <th>drat</th>
      <th>wt</th>
      <th>qsec</th>
      <th>vs</th>
      <th>am</th>
      <th>gear</th>
      <th>carb</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mazda RX4</td>
      <td>21.0</td>
      <td>6</td>
      <td>160.0</td>
      <td>110</td>
      <td>3.90</td>
      <td>2.620</td>
      <td>16.46</td>
      <td>0</td>
      <td>Manual</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mazda RX4 Wag</td>
      <td>21.0</td>
      <td>6</td>
      <td>160.0</td>
      <td>110</td>
      <td>3.90</td>
      <td>2.875</td>
      <td>17.02</td>
      <td>0</td>
      <td>Manual</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Datsun 710</td>
      <td>22.8</td>
      <td>4</td>
      <td>108.0</td>
      <td>93</td>
      <td>3.85</td>
      <td>2.320</td>
      <td>18.61</td>
      <td>1</td>
      <td>Manual</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hornet 4 Drive</td>
      <td>21.4</td>
      <td>6</td>
      <td>258.0</td>
      <td>110</td>
      <td>3.08</td>
      <td>3.215</td>
      <td>19.44</td>
      <td>1</td>
      <td>Automatic</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Hornet Sportabout</td>
      <td>18.7</td>
      <td>8</td>
      <td>360.0</td>
      <td>175</td>
      <td>3.15</td>
      <td>3.440</td>
      <td>17.02</td>
      <td>0</td>
      <td>Automatic</td>
      <td>3</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
mtcars.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cyl</th>
      <th>disp</th>
      <th>hp</th>
      <th>drat</th>
      <th>wt</th>
      <th>qsec</th>
      <th>vs</th>
      <th>gear</th>
      <th>carb</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>32.000000</td>
      <td>32.000000</td>
      <td>32.000000</td>
      <td>32.000000</td>
      <td>32.000000</td>
      <td>32.000000</td>
      <td>32.000000</td>
      <td>32.000000</td>
      <td>32.000000</td>
      <td>32.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>20.090625</td>
      <td>6.187500</td>
      <td>230.721875</td>
      <td>146.687500</td>
      <td>3.596563</td>
      <td>3.217250</td>
      <td>17.848750</td>
      <td>0.437500</td>
      <td>3.687500</td>
      <td>2.8125</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.026948</td>
      <td>1.785922</td>
      <td>123.938694</td>
      <td>68.562868</td>
      <td>0.534679</td>
      <td>0.978457</td>
      <td>1.786943</td>
      <td>0.504016</td>
      <td>0.737804</td>
      <td>1.6152</td>
    </tr>
    <tr>
      <th>min</th>
      <td>10.400000</td>
      <td>4.000000</td>
      <td>71.100000</td>
      <td>52.000000</td>
      <td>2.760000</td>
      <td>1.513000</td>
      <td>14.500000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>15.425000</td>
      <td>4.000000</td>
      <td>120.825000</td>
      <td>96.500000</td>
      <td>3.080000</td>
      <td>2.581250</td>
      <td>16.892500</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>2.0000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>19.200000</td>
      <td>6.000000</td>
      <td>196.300000</td>
      <td>123.000000</td>
      <td>3.695000</td>
      <td>3.325000</td>
      <td>17.710000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>2.0000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>22.800000</td>
      <td>8.000000</td>
      <td>326.000000</td>
      <td>180.000000</td>
      <td>3.920000</td>
      <td>3.610000</td>
      <td>18.900000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>4.0000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>33.900000</td>
      <td>8.000000</td>
      <td>472.000000</td>
      <td>335.000000</td>
      <td>4.930000</td>
      <td>5.424000</td>
      <td>22.900000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>8.0000</td>
    </tr>
  </tbody>
</table>
</div>




```python
mtcars.columns
```




    Index(['Unnamed: 0', 'mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs',
           'am', 'gear', 'carb'],
          dtype='object')




```python
sns.pairplot(mtcars)
```




    <seaborn.axisgrid.PairGrid at 0x2643ace0d68>




![png](output_5_1.png)



```python
sns.distplot(mtcars['mpg'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2643f6301d0>




![png](output_6_1.png)



```python
mtcars.corr()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cyl</th>
      <th>disp</th>
      <th>hp</th>
      <th>drat</th>
      <th>wt</th>
      <th>qsec</th>
      <th>vs</th>
      <th>gear</th>
      <th>carb</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mpg</th>
      <td>1.000000</td>
      <td>-0.852162</td>
      <td>-0.847551</td>
      <td>-0.776168</td>
      <td>0.681172</td>
      <td>-0.867659</td>
      <td>0.418684</td>
      <td>0.664039</td>
      <td>0.480285</td>
      <td>-0.550925</td>
    </tr>
    <tr>
      <th>cyl</th>
      <td>-0.852162</td>
      <td>1.000000</td>
      <td>0.902033</td>
      <td>0.832447</td>
      <td>-0.699938</td>
      <td>0.782496</td>
      <td>-0.591242</td>
      <td>-0.810812</td>
      <td>-0.492687</td>
      <td>0.526988</td>
    </tr>
    <tr>
      <th>disp</th>
      <td>-0.847551</td>
      <td>0.902033</td>
      <td>1.000000</td>
      <td>0.790949</td>
      <td>-0.710214</td>
      <td>0.887980</td>
      <td>-0.433698</td>
      <td>-0.710416</td>
      <td>-0.555569</td>
      <td>0.394977</td>
    </tr>
    <tr>
      <th>hp</th>
      <td>-0.776168</td>
      <td>0.832447</td>
      <td>0.790949</td>
      <td>1.000000</td>
      <td>-0.448759</td>
      <td>0.658748</td>
      <td>-0.708223</td>
      <td>-0.723097</td>
      <td>-0.125704</td>
      <td>0.749812</td>
    </tr>
    <tr>
      <th>drat</th>
      <td>0.681172</td>
      <td>-0.699938</td>
      <td>-0.710214</td>
      <td>-0.448759</td>
      <td>1.000000</td>
      <td>-0.712441</td>
      <td>0.091205</td>
      <td>0.440278</td>
      <td>0.699610</td>
      <td>-0.090790</td>
    </tr>
    <tr>
      <th>wt</th>
      <td>-0.867659</td>
      <td>0.782496</td>
      <td>0.887980</td>
      <td>0.658748</td>
      <td>-0.712441</td>
      <td>1.000000</td>
      <td>-0.174716</td>
      <td>-0.554916</td>
      <td>-0.583287</td>
      <td>0.427606</td>
    </tr>
    <tr>
      <th>qsec</th>
      <td>0.418684</td>
      <td>-0.591242</td>
      <td>-0.433698</td>
      <td>-0.708223</td>
      <td>0.091205</td>
      <td>-0.174716</td>
      <td>1.000000</td>
      <td>0.744535</td>
      <td>-0.212682</td>
      <td>-0.656249</td>
    </tr>
    <tr>
      <th>vs</th>
      <td>0.664039</td>
      <td>-0.810812</td>
      <td>-0.710416</td>
      <td>-0.723097</td>
      <td>0.440278</td>
      <td>-0.554916</td>
      <td>0.744535</td>
      <td>1.000000</td>
      <td>0.206023</td>
      <td>-0.569607</td>
    </tr>
    <tr>
      <th>gear</th>
      <td>0.480285</td>
      <td>-0.492687</td>
      <td>-0.555569</td>
      <td>-0.125704</td>
      <td>0.699610</td>
      <td>-0.583287</td>
      <td>-0.212682</td>
      <td>0.206023</td>
      <td>1.000000</td>
      <td>0.274073</td>
    </tr>
    <tr>
      <th>carb</th>
      <td>-0.550925</td>
      <td>0.526988</td>
      <td>0.394977</td>
      <td>0.749812</td>
      <td>-0.090790</td>
      <td>0.427606</td>
      <td>-0.656249</td>
      <td>-0.569607</td>
      <td>0.274073</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.heatmap(mtcars.corr(), annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2643d034f60>




![png](output_8_1.png)



```python
X = mtcars[['cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'carb']]
y =mtcars['mpg']
```


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
```


```python
from sklearn.linear_model import LinearRegression
```


```python
lm = LinearRegression()
```


```python
lm.fit(X_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)




```python
print(lm.intercept_)
```

    35.953154418
    


```python
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cyl</th>
      <td>-0.779906</td>
    </tr>
    <tr>
      <th>disp</th>
      <td>0.005502</td>
    </tr>
    <tr>
      <th>hp</th>
      <td>-0.013168</td>
    </tr>
    <tr>
      <th>drat</th>
      <td>0.574501</td>
    </tr>
    <tr>
      <th>wt</th>
      <td>-4.258394</td>
    </tr>
    <tr>
      <th>qsec</th>
      <td>0.078175</td>
    </tr>
    <tr>
      <th>carb</th>
      <td>-0.129095</td>
    </tr>
  </tbody>
</table>
</div>




```python
predictions = lm.predict(X_test)
```


```python
plt.scatter(y_test,predictions)
```




    <matplotlib.collections.PathCollection at 0x26440804278>




![png](output_18_1.png)



```python
sns.distplot((y_test-predictions))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x26440895a58>




![png](output_19_1.png)



```python
iris = pd.read_csv('iris_new.csv')
```


```python
iris.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Sepal.Length</th>
      <th>Sepal.Width</th>
      <th>Petal.Length</th>
      <th>Petal.Width</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>




```python
iris.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Sepal.Length</th>
      <th>Sepal.Width</th>
      <th>Petal.Length</th>
      <th>Petal.Width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>50.500000</td>
      <td>5.914000</td>
      <td>3.041000</td>
      <td>3.851000</td>
      <td>1.21300</td>
    </tr>
    <tr>
      <th>std</th>
      <td>29.011492</td>
      <td>0.856469</td>
      <td>0.439489</td>
      <td>1.785378</td>
      <td>0.74558</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>4.300000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.10000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>25.750000</td>
      <td>5.175000</td>
      <td>2.800000</td>
      <td>1.600000</td>
      <td>0.37500</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>50.500000</td>
      <td>5.850000</td>
      <td>3.000000</td>
      <td>4.500000</td>
      <td>1.40000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>75.250000</td>
      <td>6.500000</td>
      <td>3.300000</td>
      <td>5.100000</td>
      <td>1.80000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>100.000000</td>
      <td>7.900000</td>
      <td>4.400000</td>
      <td>6.900000</td>
      <td>2.50000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.pairplot(iris)
```




    <seaborn.axisgrid.PairGrid at 0x26440908198>




![png](output_23_1.png)



```python
iris.corr()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Sepal.Length</th>
      <th>Sepal.Width</th>
      <th>Petal.Length</th>
      <th>Petal.Width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Unnamed: 0</th>
      <td>1.000000</td>
      <td>0.735519</td>
      <td>-0.425542</td>
      <td>0.889291</td>
      <td>0.885842</td>
    </tr>
    <tr>
      <th>Sepal.Length</th>
      <td>0.735519</td>
      <td>1.000000</td>
      <td>-0.105661</td>
      <td>0.871159</td>
      <td>0.800274</td>
    </tr>
    <tr>
      <th>Sepal.Width</th>
      <td>-0.425542</td>
      <td>-0.105661</td>
      <td>1.000000</td>
      <td>-0.437935</td>
      <td>-0.382657</td>
    </tr>
    <tr>
      <th>Petal.Length</th>
      <td>0.889291</td>
      <td>0.871159</td>
      <td>-0.437935</td>
      <td>1.000000</td>
      <td>0.964567</td>
    </tr>
    <tr>
      <th>Petal.Width</th>
      <td>0.885842</td>
      <td>0.800274</td>
      <td>-0.382657</td>
      <td>0.964567</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.heatmap(iris.corr(), annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x26441610cf8>




![png](output_25_1.png)



```python
X = iris[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]
y = iris['Species']
```


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
```


```python
from sklearn.linear_model import LogisticRegression
```


```python
glm = LogisticRegression()
```


```python
glm.fit(X_train, y_train) # need help in this step on syntax
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)




```python
print(glm.intercept_)
```

    [ 0.20222326  0.44682296 -0.64273789]
    


```python
predictions = glm.predict(X_test)
```


```python
plt.scatter(y_test, predictions)
```




    <matplotlib.collections.PathCollection at 0x26442a04d68>




![png](output_34_1.png)

