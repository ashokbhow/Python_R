

```python
print("hello")
```

    hello
    


```python
print("everything works file")
```

    everything works file
    


```python
def printme( str ):
    "This prints a passed string into this function"
    return str
```


```python
printme ("I Am All That I Am")
```




    'I Am All That I Am'




```python
a = [1,3,5,7,9] 
b = [3,5,6,7,9]
a+b
```




    [1, 3, 5, 7, 9, 3, 5, 6, 7, 9]




```python
import numpy as np
```


```python
a = np.array([1,3,5,7,9]) 
b = np.array([3,5,6,7,9])
a+b
```




    array([ 4,  8, 11, 14, 18])




```python
import matplotlib.pyplot as plt
from pylab import *

xs = [1,2,3,4,5]
ys = [x**2 for x in xs]

plt.plot(xs, ys)
savefig('linechart')
plt.show() 
```


![png](output_7_0.png)



```python
import numpy as np
import matplotlib.pyplot as plt
# evenly sampled time at .2 intervals
t = np.arange(0., 10., 0.2)
print(t)
# red dashes, blue squares and green triangles 
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.axis([0, 10, 0, 150]) 
# x and y range of axis 
plt.show() 
```

    [ 0.   0.2  0.4  0.6  0.8  1.   1.2  1.4  1.6  1.8  2.   2.2  2.4  2.6  2.8
      3.   3.2  3.4  3.6  3.8  4.   4.2  4.4  4.6  4.8  5.   5.2  5.4  5.6  5.8
      6.   6.2  6.4  6.6  6.8  7.   7.2  7.4  7.6  7.8  8.   8.2  8.4  8.6  8.8
      9.   9.2  9.4  9.6  9.8]
    


![png](output_8_1.png)



```python
import numpy as np 
import matplotlib.pyplot as plt 
def f(t):     
    return np.exp(-t) * np.cos(2*np.pi*t)
t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)
plt.figure(1) # Called implicitly but can use               
# for multiple figures 
plt.subplot(231) # 1 rows, 3 column, 1st plot 
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')
plt.subplot(232)
plt.plot(t1, np.exp(-t), 'bo')
plt.subplot(233)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
#plt.show()
savefig('new')
show()
```


![png](output_9_0.png)



```python
f(t1)
```




    array([ 1.        ,  0.73202885,  0.25300172, -0.22892542, -0.54230031,
           -0.60653066, -0.44399794, -0.1534533 ,  0.13885029,  0.32892176,
            0.36787944,  0.26929836,  0.09307413, -0.08421696, -0.19950113,
           -0.22313016, -0.16333771, -0.05645231,  0.05108017,  0.12100355,
            0.13533528,  0.09906933,  0.03424006, -0.03098169, -0.07339237,
           -0.082085  , -0.06008859, -0.02076765,  0.01879134,  0.04451472,
            0.04978707,  0.03644557,  0.01259621, -0.01139753, -0.02699954,
           -0.03019738, -0.02210536, -0.00763999,  0.00691295,  0.01637605,
            0.01831564,  0.01340758,  0.00463389, -0.00419292, -0.00993258,
           -0.011109  , -0.00813211, -0.0028106 ,  0.00254313,  0.00602441])




```python
f(t2)
```




    array([  1.00000000e+00,   9.72469514e-01,   9.30604472e-01,
             8.75630519e-01,   8.08933021e-01,   7.32028848e-01,
             6.46537173e-01,   5.54149795e-01,   4.56601475e-01,
             3.55640759e-01,   2.53001717e-01,   1.50377027e-01,
             4.93927721e-02,  -4.84147297e-02,  -1.41619751e-01,
            -2.28925420e-01,  -3.09179223e-01,  -3.81385611e-01,
            -4.44715627e-01,  -4.98513513e-01,  -5.42300309e-01,
            -5.75774517e-01,  -5.98809920e-01,  -6.11450709e-01,
            -6.13904100e-01,  -6.06530660e-01,  -5.89832576e-01,
            -5.64440144e-01,  -5.31096756e-01,  -4.90642679e-01,
            -4.43997940e-01,  -3.92144618e-01,  -3.36108841e-01,
            -2.76942794e-01,  -2.15707024e-01,  -1.53453298e-01,
            -9.12082776e-02,  -2.99582306e-02,   2.93650179e-02,
             8.58967210e-02,   1.38850286e-01,   1.87526678e-01,
             2.31322066e-01,   2.69733663e-01,   3.02363730e-01,
             3.28921764e-01,   3.49224898e-01,   3.63196576e-01,
             3.70863602e-01,   3.72351659e-01,   3.67879441e-01,
             3.57751541e-01,   3.42350253e-01,   3.22126466e-01,
             2.97589828e-01,   2.69298364e-01,   2.37847734e-01,
             2.03860317e-01,   1.67974296e-01,   1.30832924e-01,
             9.30741301e-02,   5.53206168e-02,   1.81705854e-02,
            -1.78107837e-02,  -5.20989949e-02,  -8.42169556e-02,
            -1.13740680e-01,  -1.40303925e-01,  -1.63601736e-01,
            -1.83392873e-01,  -1.99501135e-01,  -2.11815608e-01,
            -2.20289859e-01,  -2.24940145e-01,  -2.25842697e-01,
            -2.23130160e-01,  -2.16987278e-01,  -2.07645925e-01,
            -1.95379578e-01,  -1.80497354e-01,  -1.63337714e-01,
            -1.44261943e-01,  -1.23647532e-01,  -1.01881560e-01,
            -7.93541795e-02,  -5.64523135e-02,  -3.35536502e-02,
            -1.10210171e-02,   1.08027864e-02,   3.15996377e-02,
             5.10801656e-02,   6.89872094e-02,   8.50986324e-02,
             9.92294691e-02,   1.11233400e-01,   1.21003555e-01,
             1.28472660e-01,   1.33612553e-01,   1.36433095e-01,
             1.36980520e-01,   1.35335283e-01,   1.31609437e-01,
             1.25943620e-01,   1.18503704e-01,   1.09477179e-01,
             9.90693315e-02,   8.74992915e-02,   7.49960195e-02,
             6.17942900e-02,   4.81307428e-02,   3.42400590e-02,
             2.03513176e-02,   6.68458480e-03,  -6.55222115e-03,
            -1.91661491e-02,  -3.09816865e-02,  -4.18428577e-02,
            -5.16149297e-02,  -6.01857154e-02,  -6.74664675e-02,
            -7.33923659e-02,  -7.79226074e-02,  -8.10401102e-02,
            -8.27508549e-02,  -8.30828852e-02,  -8.20849986e-02,
            -7.98251587e-02,  -7.63886668e-02,  -7.18761299e-02,
            -6.64012659e-02,  -6.00885870e-02,  -5.30710030e-02,
            -4.54873852e-02,  -3.74801315e-02,  -2.91927712e-02,
            -2.07676456e-02,  -1.23436981e-02,  -4.05440563e-03,
             3.97412302e-03,   1.16248571e-02,   1.87913428e-02,
             2.53789761e-02,   3.13060373e-02,   3.65044817e-02,
             4.09204810e-02,   4.45147201e-02,   4.72624505e-02,
             4.91533115e-02,   5.01909306e-02,   5.03923172e-02,
             4.97870684e-02,   4.84164062e-02,   4.63320685e-02,
             4.35950765e-02,   4.02744036e-02,   3.64455703e-02,
             3.21891905e-02,   2.75894937e-02,   2.27328489e-02,
             1.77063108e-02,   1.25962138e-02,   7.48683134e-03,
             2.45912132e-03,  -2.41042746e-03,  -7.05083223e-03,
            -1.13975255e-02,  -1.53931271e-02,  -1.89880715e-02,
            -2.21410873e-02,  -2.48195263e-02,  -2.69995426e-02,
            -2.86661253e-02,  -2.98129904e-02,  -3.04423382e-02,
            -3.05644854e-02,  -3.01973834e-02,  -2.93660348e-02,
            -2.81018201e-02,  -2.64417505e-02,  -2.44276606e-02,
            -2.21053558e-02,  -1.95237309e-02,  -1.67338738e-02,
            -1.37881698e-02,  -1.07394204e-02,  -7.63998984e-03,
            -4.54099275e-03,  -1.49153248e-03,   1.46199815e-03,
             4.27654592e-03,   6.91294868e-03,   9.33640353e-03,
             1.15168475e-02,   1.34292483e-02,   1.50538037e-02,
             1.63760504e-02,   1.73868839e-02,   1.80824928e-02,
             1.84642115e-02,   1.85382975e-02,   1.83156389e-02,
             1.78114004e-02,   1.70446155e-02,   1.60377324e-02,
             1.48161251e-02,   1.34075760e-02,   1.18417414e-02,
             1.01496075e-02,   8.36294774e-03,   6.51378771e-03,
             4.63388808e-03,   2.75425133e-03,   9.04660177e-04,
            -8.86746705e-04,  -2.59385622e-03,  -4.19291532e-03,
            -5.66281499e-03,  -6.98532112e-03,  -8.14525084e-03,
            -9.13059348e-03,  -9.93257663e-03,  -1.05456781e-02,
            -1.09675863e-02,  -1.11991104e-02,  -1.12440458e-02,
            -1.11089965e-02,  -1.08031605e-02,  -1.03380819e-02,
            -9.72737640e-03,  -8.98643413e-03,  -8.13210594e-03,
            -7.18237922e-03,  -6.15604815e-03,  -5.07238421e-03,
            -3.95081196e-03,  -2.81059519e-03,  -1.67053788e-03,
            -5.48704134e-04,   5.37839064e-04,   1.57325332e-03,
             2.54313170e-03,   3.43467091e-03,   4.23681143e-03,
             4.94034436e-03,   5.53798489e-03,   6.02441225e-03,
             6.39627712e-03,   6.65217733e-03,   6.79260381e-03,
             6.81985852e-03])




```python
import numpy as np
import matplotlib.pyplot as plt
t=np.arange(0,5,0.05)
f=2*np.pi*np.sin(2*np.pi*t)
plt.plot(t,f)
plt.grid()
plt.xlabel('‘x’')
plt.ylabel('‘y’')
plt.title('‘First Plot’')
plt.show()

```


![png](output_12_0.png)



```python
x=arange(0,2*pi,0.1)
plot(x,sin(2*x),marker='o',color='r', markerfacecolor='g',label='sin(2x)')
legend()#show the legend
savefig('sin')
show()
```


![png](output_13_0.png)



```python
print(sin(2*x))
print(x)
```

    [ 0.          0.19866933  0.38941834  0.56464247  0.71735609  0.84147098
      0.93203909  0.98544973  0.9995736   0.97384763  0.90929743  0.8084964
      0.67546318  0.51550137  0.33498815  0.14112001 -0.05837414 -0.2555411
     -0.44252044 -0.61185789 -0.7568025  -0.87157577 -0.95160207 -0.993691
     -0.99616461 -0.95892427 -0.88345466 -0.77276449 -0.63126664 -0.46460218
     -0.2794155  -0.0830894   0.1165492   0.31154136  0.49411335  0.6569866
      0.79366786  0.8987081   0.96791967  0.99854335  0.98935825  0.94073056
      0.85459891  0.7343971   0.58491719  0.41211849  0.22288991  0.02477543
     -0.17432678 -0.36647913 -0.54402111 -0.69987469 -0.82782647 -0.92277542
     -0.98093623 -0.99999021 -0.97917773 -0.91932853 -0.82282859 -0.69352508
     -0.53657292 -0.35822928 -0.16560418]
    [ 0.   0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.   1.1  1.2  1.3  1.4
      1.5  1.6  1.7  1.8  1.9  2.   2.1  2.2  2.3  2.4  2.5  2.6  2.7  2.8  2.9
      3.   3.1  3.2  3.3  3.4  3.5  3.6  3.7  3.8  3.9  4.   4.1  4.2  4.3  4.4
      4.5  4.6  4.7  4.8  4.9  5.   5.1  5.2  5.3  5.4  5.5  5.6  5.7  5.8  5.9
      6.   6.1  6.2]
    


```python
import numpy as np
from matplotlib import pyplot as plt
x=np.arange(0,100,10)
y=2.0*np.sqrt(x)
f=plt.figure()
ax=f.add_subplot(111)
line,=ax.plot(x,y)
#line.set_color('r')
line.set_linestyle('--')
line.set_marker('s')
plt.setp(line,markeredgecolor='b', color='r',markeredgewidth=2)
line.set_markersize(10)
savefig('sqrt')
show()
```


![png](output_15_0.png)



```python
from pylab import *
n_day1=[7,10,15,17,17,10,5,3,6,15,18,8]
n_day2=[5,6,6,12,13,15,15,18,16,13,10,6]
m=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec']
width=0.2
i=arange(len(n_day1))
r1=bar(i, n_day1,width, color='r',linewidth=1)
r2=bar(i+width,n_day2,width,color='b',linewidth=1)
xticks(i+width/2,m)
xlabel('Month'); ylabel('Rain Days'); title('Comparison')
legend((r1[0],r2[0]),('City1','City2'),loc=0)
savefig('barchart')
show()
```


![png](output_16_0.png)



```python
import numpy as np
import matplotlib.pyplot as plt

N = 6
men_means = (20, 35, 30, 35, 27, 17)
men_std = (2, 3, 4, 1, 2, 5)

ind = np.arange(N)  # the x locations for the groups
print(ind)
width = 0.25       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, men_means, width, color='r', yerr=men_std)

women_means = (25, 32, 34, 20, 25, 19)
women_std = (3, 5, 2, 3, 3, 4)
rects2 = ax.bar(ind + width, women_means, width, color='y', yerr=women_std)

# add some text for labels, title and axes ticks
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5', 'G6'))

ax.legend((rects1[0], rects2[0]), ('Men', 'Women'))


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
savefig('Special BarPlot')
plt.show()
```

    [0 1 2 3 4 5]
    


![png](output_17_1.png)



```python
import matplotlib.pyplot as plt

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs', 'Jogs'
sizes = [15, 30, 25, 20, 10]
explode = (0, 0.075, 0, 0.075, 0)  # only "explode" the 2nd and 4th slices (i.e. 'Hogs' and 'Logs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=30)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
savefig('piechart')
show()
```


![png](output_18_0.png)



```python
subplot(221)
pie(n_day1,labels=m,explode=[0,0,0,0.1,0.1,0,0,0,0,0,0.1,0],shadow=True, startangle = 90)
axis('equal')
title('City1')
subplot(222)
pie(n_day2,labels=m,explode=[0,0,0,0,0,0,0,0.1,0.1,0,0,0],shadow=True, startangle = 0)
axis('equal')
title('City2')
savefig('Pie Array')
show()
```


![png](output_19_0.png)



```python
from pylab import *
y = arange(1,10,0.2)
x = 1/(y**2)
hlines(x, -1, y, color='r', lw=3)
plt.title('Molecular Orbitals')
savefig('hlines')
show()
```


![png](output_20_0.png)



```python
from pylab import *
x = arange(0,10,0.2)
y = x**3
vlines(x, 0, y, color='b', lw=3)
savefig('vlines')#saves image as vlines
show()
```


![png](output_21_0.png)



```python
print(y)
```

    [  0.00000000e+00   8.00000000e-03   6.40000000e-02   2.16000000e-01
       5.12000000e-01   1.00000000e+00   1.72800000e+00   2.74400000e+00
       4.09600000e+00   5.83200000e+00   8.00000000e+00   1.06480000e+01
       1.38240000e+01   1.75760000e+01   2.19520000e+01   2.70000000e+01
       3.27680000e+01   3.93040000e+01   4.66560000e+01   5.48720000e+01
       6.40000000e+01   7.40880000e+01   8.51840000e+01   9.73360000e+01
       1.10592000e+02   1.25000000e+02   1.40608000e+02   1.57464000e+02
       1.75616000e+02   1.95112000e+02   2.16000000e+02   2.38328000e+02
       2.62144000e+02   2.87496000e+02   3.14432000e+02   3.43000000e+02
       3.73248000e+02   4.05224000e+02   4.38976000e+02   4.74552000e+02
       5.12000000e+02   5.51368000e+02   5.92704000e+02   6.36056000e+02
       6.81472000e+02   7.29000000e+02   7.78688000e+02   8.30584000e+02
       8.84736000e+02   9.41192000e+02]
    


```python
print(x)
```

    [ 0.   0.2  0.4  0.6  0.8  1.   1.2  1.4  1.6  1.8  2.   2.2  2.4  2.6  2.8
      3.   3.2  3.4  3.6  3.8  4.   4.2  4.4  4.6  4.8  5.   5.2  5.4  5.6  5.8
      6.   6.2  6.4  6.6  6.8  7.   7.2  7.4  7.6  7.8  8.   8.2  8.4  8.6  8.8
      9.   9.2  9.4  9.6  9.8]
    


```python
import matplotlib.pyplot as plt
from numpy.random import normal
gaussian_numbers = normal(size=1000)
#print(gaussian_numbers)
plt.hist(gaussian_numbers, edgecolor='black', linewidth=1.5)
plt.title("Gaussian Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
savefig('hist')#saves image as vlines
show()
```


![png](output_24_0.png)



```python
import numpy as np 
import matplotlib.pyplot as plt 

mu, sigma = 100, 15 
x = mu + sigma * np.random.randn(10000) 

print(x)
# the histogram of the data 
n, bins, patches = plt.hist(x, 50, normed=2, edgecolor='r',facecolor='y') 

plt.xlabel('Smarts') 
plt.ylabel('Probability') 
plt.title('Histogram of IQ') 
plt.text(60, .025, '$\mu=100,\ \sigma=15$') #TeX equations 
plt.axis([40, 160, 0, 0.03]) 
plt.grid(True) 
savefig('hist1')#saves image as vlines
#plt.
show()
```

    [  92.36099419   89.96201393   97.96522002 ...,  116.79021131  104.67086985
      111.00931069]
    


![png](output_25_1.png)



```python
x = [np.random.normal(0, std, 1000) for std in range(1, 10)]
# rectangular box plot
boxplot(x,vert=True,patch_artist=True)
savefig('boxplot')
show()
```


![png](output_26_0.png)

