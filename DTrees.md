

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

```


```python
balance_data = pd.read_csv(
'https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data',
                           sep= ',', header= None)
```


```python
balance_data.describe()
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
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>625.000000</td>
      <td>625.000000</td>
      <td>625.000000</td>
      <td>625.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.415346</td>
      <td>1.415346</td>
      <td>1.415346</td>
      <td>1.415346</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
print( "Dataset Lenght:: ", len(balance_data))
print ("Dataset Shape:: ", balance_data.shape)
```

    Dataset Lenght::  625
    Dataset Shape::  (625, 5)
    


```python
balance_data.head()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>R</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>R</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>R</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>R</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
balance_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 625 entries, 0 to 624
    Data columns (total 5 columns):
    0    625 non-null object
    1    625 non-null int64
    2    625 non-null int64
    3    625 non-null int64
    4    625 non-null int64
    dtypes: int64(4), object(1)
    memory usage: 24.5+ KB
    


```python
X = balance_data.values[:, 1:5]
Y = balance_data.values[:,0]
```


```python
X
```




    array([[1, 1, 1, 1],
           [1, 1, 1, 2],
           [1, 1, 1, 3],
           ..., 
           [5, 5, 5, 3],
           [5, 5, 5, 4],
           [5, 5, 5, 5]], dtype=object)




```python
Y
```




    array(['B', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
           'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'L',
           'B', 'R', 'R', 'R', 'B', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
           'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'L', 'L',
           'B', 'R', 'R', 'L', 'R', 'R', 'R', 'R', 'B', 'R', 'R', 'R', 'R',
           'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'L', 'L', 'L',
           'B', 'R', 'L', 'B', 'R', 'R', 'R', 'L', 'R', 'R', 'R', 'R', 'B',
           'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'L', 'L', 'L', 'L',
           'B', 'L', 'L', 'R', 'R', 'R', 'L', 'R', 'R', 'R', 'R', 'L', 'R',
           'R', 'R', 'R', 'B', 'R', 'R', 'R', 'R', 'L', 'B', 'R', 'R', 'R',
           'B', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
           'R', 'R', 'R', 'R', 'R', 'R', 'R', 'L', 'L', 'L', 'B', 'R', 'L',
           'B', 'R', 'R', 'R', 'L', 'R', 'R', 'R', 'R', 'B', 'R', 'R', 'R',
           'R', 'R', 'R', 'R', 'R', 'R', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
           'B', 'R', 'R', 'L', 'B', 'R', 'R', 'R', 'L', 'R', 'R', 'R', 'R',
           'L', 'R', 'R', 'R', 'R', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
           'B', 'R', 'L', 'L', 'R', 'R', 'R', 'L', 'B', 'R', 'R', 'R', 'L',
           'R', 'R', 'R', 'R', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
           'B', 'L', 'L', 'L', 'R', 'R', 'L', 'L', 'R', 'R', 'R', 'L', 'B',
           'R', 'R', 'R', 'L', 'L', 'B', 'R', 'R', 'L', 'R', 'R', 'R', 'R',
           'B', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
           'R', 'R', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'B', 'R', 'R', 'L',
           'B', 'R', 'R', 'R', 'L', 'R', 'R', 'R', 'R', 'L', 'R', 'R', 'R',
           'R', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'R', 'L', 'L',
           'B', 'R', 'R', 'L', 'L', 'R', 'R', 'R', 'L', 'R', 'R', 'R', 'R',
           'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
           'B', 'R', 'L', 'L', 'B', 'R', 'R', 'L', 'L', 'R', 'R', 'R', 'L',
           'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
           'B', 'L', 'L', 'L', 'R', 'R', 'L', 'L', 'B', 'R', 'R', 'L', 'L',
           'L', 'B', 'R', 'L', 'B', 'R', 'R', 'R', 'L', 'R', 'R', 'R', 'R',
           'B', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'L', 'L', 'L',
           'L', 'L', 'L', 'L', 'L', 'B', 'R', 'L', 'L', 'R', 'R', 'R', 'L',
           'B', 'R', 'R', 'R', 'L', 'R', 'R', 'R', 'R', 'L', 'L', 'L', 'L',
           'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'B', 'R', 'L', 'L',
           'B', 'R', 'R', 'L', 'L', 'R', 'R', 'R', 'L', 'L', 'L', 'L', 'L',
           'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
           'B', 'R', 'L', 'L', 'L', 'R', 'R', 'L', 'L', 'L', 'L', 'L', 'L',
           'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
           'B', 'L', 'L', 'L', 'B', 'R', 'L', 'L', 'L', 'L', 'B', 'L', 'L',
           'R', 'R', 'R', 'L', 'R', 'R', 'R', 'R', 'L', 'R', 'R', 'R', 'R',
           'B', 'R', 'R', 'R', 'R', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
           'L', 'B', 'L', 'L', 'L', 'R', 'R', 'L', 'L', 'R', 'R', 'R', 'L',
           'B', 'R', 'R', 'R', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
           'L', 'L', 'L', 'L', 'L', 'B', 'L', 'L', 'L', 'R', 'R', 'L', 'L',
           'B', 'R', 'R', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
           'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'B', 'L', 'L', 'L',
           'B', 'R', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
           'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'B'], dtype=object)




```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)
# random_state is like a seeding 
```


```python
from sklearn.tree import DecisionTreeClassifier
```


```python
dtree = DecisionTreeClassifier()
```


```python
dtree.fit(X_train,y_train)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best')




```python
predictions = dtree.predict(X_test)
```


```python
predictions
```




    array(['L', 'L', 'R', 'L', 'R', 'L', 'R', 'L', 'L', 'B', 'L', 'B', 'R',
           'L', 'R', 'R', 'L', 'L', 'R', 'L', 'R', 'L', 'B', 'L', 'R', 'L',
           'L', 'L', 'R', 'L', 'L', 'L', 'R', 'L', 'B', 'L', 'R', 'B', 'B',
           'B', 'R', 'L', 'R', 'L', 'R', 'R', 'L', 'R', 'R', 'L', 'B', 'R',
           'L', 'L', 'R', 'L', 'R', 'R', 'L', 'B', 'B', 'R', 'L', 'B', 'L',
           'L', 'R', 'R', 'R', 'L', 'L', 'R', 'R', 'L', 'R', 'L', 'L', 'R',
           'R', 'L', 'R', 'B', 'R', 'L', 'L', 'R', 'R', 'L', 'R', 'L', 'R',
           'R', 'L', 'L', 'L', 'R', 'R', 'L', 'B', 'L', 'R', 'L', 'R', 'R',
           'B', 'R', 'R', 'R', 'L', 'L', 'R', 'L', 'R', 'R', 'L', 'B', 'R',
           'R', 'R', 'R', 'L', 'R', 'R', 'L', 'L', 'L', 'L', 'L', 'L', 'R',
           'R', 'R', 'R', 'L', 'R', 'L', 'R', 'L', 'R', 'R', 'L', 'B', 'L',
           'R', 'L', 'L', 'R', 'L', 'R', 'R', 'L', 'L', 'L', 'R', 'R', 'R',
           'L', 'R', 'R', 'L', 'B', 'R', 'B', 'L', 'R', 'R', 'L', 'R', 'R',
           'R', 'B', 'L', 'B', 'R', 'B', 'R', 'L', 'L', 'R', 'R', 'R', 'R',
           'R', 'L', 'L', 'R', 'R', 'R'], dtype=object)




```python
from sklearn.metrics import classification_report,confusion_matrix
```


```python
print(confusion_matrix(y_test,predictions))
```

    [[ 0  6  7]
     [11 71  3]
     [10  2 78]]
    


```python
print(confusion_matrix(y_test,predictions, labels=['L', 'R', 'B']))
```

    [[71  3 11]
     [ 2 78 10]
     [ 6  7  0]]
    


```python
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#graph.create_png()
plt.show()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-1-c433c215d3c3> in <module>()
          4 import pydotplus
          5 dot_data = StringIO()
    ----> 6 export_graphviz(dtree, out_file=dot_data,  
          7                 filled=True, rounded=True,
          8                 special_characters=True)
    

    NameError: name 'dtree' is not defined



```python
y_test
```




    array(['L', 'L', 'R', 'L', 'R', 'B', 'R', 'L', 'L', 'R', 'L', 'L', 'R',
           'L', 'R', 'R', 'L', 'L', 'B', 'L', 'R', 'L', 'R', 'L', 'R', 'L',
           'L', 'L', 'R', 'L', 'L', 'L', 'R', 'L', 'L', 'L', 'R', 'L', 'L',
           'R', 'R', 'L', 'R', 'L', 'R', 'R', 'L', 'R', 'R', 'L', 'L', 'R',
           'L', 'L', 'R', 'L', 'R', 'R', 'L', 'L', 'R', 'R', 'L', 'R', 'B',
           'B', 'R', 'R', 'R', 'L', 'L', 'B', 'R', 'L', 'R', 'L', 'L', 'R',
           'R', 'L', 'R', 'L', 'L', 'L', 'B', 'B', 'R', 'L', 'R', 'L', 'R',
           'R', 'L', 'L', 'L', 'R', 'R', 'L', 'R', 'L', 'B', 'L', 'B', 'R',
           'L', 'R', 'R', 'R', 'L', 'L', 'R', 'L', 'R', 'R', 'L', 'L', 'B',
           'R', 'R', 'R', 'L', 'R', 'R', 'R', 'L', 'L', 'L', 'L', 'L', 'R',
           'R', 'R', 'R', 'L', 'R', 'R', 'R', 'L', 'L', 'R', 'L', 'L', 'L',
           'R', 'L', 'L', 'R', 'L', 'R', 'R', 'L', 'L', 'L', 'R', 'R', 'R',
           'L', 'R', 'B', 'L', 'R', 'R', 'R', 'L', 'R', 'R', 'L', 'R', 'R',
           'R', 'R', 'B', 'R', 'R', 'L', 'R', 'B', 'L', 'R', 'R', 'R', 'R',
           'R', 'L', 'L', 'L', 'R', 'R'], dtype=object)




```python
var=9
```


```python
print(var)
```

    9
    


```python
lst1 = [2,2,2,2,2,0,0,0,2,1,1,1,1]
```


```python
lstp = [0,0,2,2,2,0,0,2,2,1,1,0,1]
```


```python
print(confusion_matrix(lst1,lstp, labels=[0,1,2]))
```

    [[2 0 1]
     [1 3 0]
     [2 0 4]]
    


```python
cm = confusion_matrix(lst1, lstp)
import pylab as pl
print(cm)
pl.matshow(cm)
pl.colorbar()
pl.show()
```

    [[2 0 1]
     [1 3 0]
     [2 0 4]]
    


![png](output_26_1.png)



```python
cm = confusion_matrix(y_test, predictions)
import pylab as pl
print(cm)
pl.matshow(cm)
pl.colorbar()
pl.show()
```

    [[ 0  6  7]
     [11 71  3]
     [10  2 78]]
    


![png](output_27_1.png)

