# Pandas
# 认识Pandas
### 简介
Pandas 是 Python 的核心数据分析支持库，提供了快速、灵活、明确的数据结构，旨在简单、直观地处理关系型、标记型数据。
### 应用
Pandas 适用于处理以下类型的数据：

* 与 SQL 或 Excel 表类似的，含异构列的表格数据;
* 有序和无序（非固定频率）的时间序列数据;
* 带行列标签的矩阵数据，包括同构或异构型数据;
* 任意其它形式的观测、统计数据集, 数据转入 Pandas 数据结构时不必事先标记。
### 优势
* 处理浮点与非浮点数据里的缺失数据，表示为 NaN；
* 大小可变：插入或删除 DataFrame 等多维对象的列；
* 自动、显式数据对齐：显式地将对象与一组标签对齐，也可以忽略标签，在 Series、DataFrame 计算时自动与数据对齐；
* 强大、灵活的分组（group by）功能：拆分-应用-组合数据集，聚合、转换数据；
* 把 Python 和 NumPy 数据结构里不规则、不同索引的数据轻松地转换为 DataFrame 对象；
* 基于智能标签，对大型数据集进行切片、花式索引、子集分解等操作；
* 直观地合并（merge）、连接(join)数据集；
* 灵活地重塑（reshape）、透视(pivot)数据集；
* 轴支持结构化标签：一个刻度支持多个标签；
* 成熟的 IO 工具：读取文本文件（CSV 等支持分隔符的文件）、Excel 文件、数据库等来源的数据，利用超快的 HDF5 格式保存 / 加载数据；
* 时间序列：支持日期范围生成、频率转换、移动窗口统计、移动窗口线性回归、日期位移等时间序列功能。

# 数据结构
    import numpy as np
    import pandas as pd
## Series
Series 是带标签的一维数组，可以储存整数，浮点数，Python 对象等类型的数据。轴标签为 index
### 创建
调用pd.Series函数即可创建

    s = pd.Series(data, index=index)
data 可以是 
* Python 字典
* 多维数组 
* 标量值（如，5）
### 操作
Series 类似于多维数组，操作与ndarray类似，支持大多数numpy函数，还支持索引切片：
```
In [3]: s = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])

In [4]: s
Out[4]: 
a    0.469112
b   -0.282863
c   -1.509059
d   -1.135632
e    1.212112
dtype: float64

In [13]: s[0]
Out[13]: 0.4691122999071863

In [14]: s[:3]
Out[14]: 
a    0.469112
b   -0.282863
c   -1.509059
dtype: float64

In [15]: s[s > s.median()]
Out[15]: 
a    0.469112
e    1.212112
dtype: float64

In [16]: s[[4, 3, 1]]
Out[16]: 
e    1.212112
d   -1.135632
b   -0.282863
dtype: float64
```
Series 类似于字典，可以用索引标签取值或者设置值：
```
In [21]: s['a']
Out[21]: 0.4691122999071863

In [22]: s['e'] = 12.

In [23]: s
Out[23]: 
a     0.469112
b    -0.282863
c    -1.509059
d    -1.135632
e    12.000000
dtype: float64

In [24]: 'e' in s
Out[24]: True

In [25]: 'f' in s
Out[25]: False
```
矢量操作，对齐标签  
Series不用循环每个值，可直接进行运算。
```
In [28]: s + s
Out[28]: 
a     0.938225
b    -0.565727
c    -3.018117
d    -2.271265
e    24.000000
dtype: float64

In [29]: s * 2
Out[29]: 
a     0.938225
b    -0.565727
c    -3.018117
d    -2.271265
e    24.000000
dtype: float64
```
Series 和多维数组的主要区别是可以基于标签对齐数据。
```
In [31]: s[1:] + s[:-1]
Out[31]: 
a         NaN
b   -0.565727
c   -3.018117
d   -2.271265
e         NaN
dtype: float64
```
Pandas 用 NaN（Not a Number）表示缺失数据。
## DataFrame
是由多种类型的列构成的二维标签数据，类似于Excel，SQL表。支持多种类型输入
* 一维ndarray、列表、字典
* 二维numpy.ndarray
* 结构多维数组或记录多维数组
* Series
* DataFrame  
  
除了数据，还可以有选择的传递index（行标签）和column（列标签）参数。没有传递标签时，按常规输入数据进行创建。  
轴标签axis=0代表index，axis=1代表column
### 创建
使用Series字典  
生成的索引是每个Series索引的并集
```
In [37]: d = {'one': pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
   ....:      'two': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}
   ....: 

In [38]: df = pd.DataFrame(d)

In [39]: df
Out[39]: 
   one  two
a  1.0  1.0
b  2.0  2.0
c  3.0  3.0
d  NaN  4.0

In [40]: pd.DataFrame(d, index=['d', 'b', 'a'])
Out[40]: 
   one  two
d  NaN  4.0
b  2.0  2.0
a  1.0  1.0

In [41]: pd.DataFrame(d, index=['d', 'b', 'a'], columns=['two', 'three'])
Out[41]: 
   two three
d  4.0   NaN
b  2.0   NaN
a  1.0   NaN
```
使用多位数组字典、列表字典  
多维数组的长度必须相同。如果传递课索引参数，index的长度与数组需一致。没有传递参数索引生成结果是range(n)
```
In [44]: d = {'one': [1., 2., 3., 4.],
   ....:      'two': [4., 3., 2., 1.]}
   ....: 

In [45]: pd.DataFrame(d)
Out[45]: 
   one  two
0  1.0  4.0
1  2.0  3.0
2  3.0  2.0
3  4.0  1.0

In [46]: pd.DataFrame(d, index=['a', 'b', 'c', 'd'])
Out[46]: 
   one  two
a  1.0  4.0
b  2.0  3.0
c  3.0  2.0
d  4.0  1.0
```
使用列表字典生成
```
In [52]: data2 = [{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}]

In [53]: pd.DataFrame(data2)
Out[53]: 
   a   b     c
0  1   2   NaN
1  5  10  20.0

In [54]: pd.DataFrame(data2, index=['first', 'second'])
Out[54]: 
        a   b     c
first   1   2   NaN
second  5  10  20.0

In [55]: pd.DataFrame(data2, columns=['a', 'b'])
Out[55]: 
   a   b
0  1   2
1  5  10
```
### 操作
提取、添加、删除列  
就像是带索引的Series字典，提取、设置、删除列的操作与字典类似：
```
In [61]: df['one']
Out[61]: 
a    1.0
b    2.0
c    3.0
d    NaN
Name: one, dtype: float64

In [62]: df['three'] = df['one'] * df['two']

In [63]: df['flag'] = df['one'] > 2

In [64]: df
Out[64]: 
   one  two  three   flag
a  1.0  1.0    1.0  False
b  2.0  2.0    4.0  False
c  3.0  3.0    9.0   True
d  NaN  4.0    NaN  False

In [65]: del df['two']

In [66]: three = df.pop('three')

In [67]: df
Out[67]: 
   one   flag
a  1.0  False
b  2.0  False
c  3.0   True
d  NaN  False

In [68]: df['foo'] = 'bar'

In [69]: df
Out[69]: 
   one   flag  foo
a  1.0  False  bar
b  2.0  False  bar
c  3.0   True  bar
d  NaN  False  bar
```
使用现有的列创建新的列 assign()方法
```
In [74]: iris = pd.read_csv('data/iris.data')

In [75]: iris.head()
Out[75]: 
   SepalLength  SepalWidth  PetalLength  PetalWidth         Name
0          5.1         3.5          1.4         0.2  Iris-setosa
1          4.9         3.0          1.4         0.2  Iris-setosa
2          4.7         3.2          1.3         0.2  Iris-setosa
3          4.6         3.1          1.5         0.2  Iris-setosa
4          5.0         3.6          1.4         0.2  Iris-setosa

In [76]: (iris.assign(sepal_ratio=iris['SepalWidth'] / iris['SepalLength'])
   ....:      .head())
   ....: 
Out[76]: 
   SepalLength  SepalWidth  PetalLength  PetalWidth         Name  sepal_ratio
0          5.1         3.5          1.4         0.2  Iris-setosa     0.686275
1          4.9         3.0          1.4         0.2  Iris-setosa     0.612245
2          4.7         3.2          1.3         0.2  Iris-setosa     0.680851
3          4.6         3.1          1.5         0.2  Iris-setosa     0.673913
4          5.0         3.6          1.4         0.2  Iris-setosa     0.720000
```
还可以传递带参数的函数实现
```
In [77]: iris.assign(sepal_ratio=lambda x: (x['SepalWidth'] / x['SepalLength'])).head()
Out[77]: 
   SepalLength  SepalWidth  PetalLength  PetalWidth         Name  sepal_ratio
0          5.1         3.5          1.4         0.2  Iris-setosa     0.686275
1          4.9         3.0          1.4         0.2  Iris-setosa     0.612245
2          4.7         3.2          1.3         0.2  Iris-setosa     0.680851
3          4.6         3.1          1.5         0.2  Iris-setosa     0.673913
4          5.0         3.6          1.4         0.2  Iris-setosa     0.720000
```
索引，选择
* 选择列  df[col]	Series
* 用标签选择行	df.loc[label]	Series
* 用整数位置选择行	df.iloc[loc]	Series
* 行切片	df[5:10]	DataFrame
* 用布尔向量选择行	df[bool_vec]	DataFrame

```
In [83]: df.loc['b']
Out[83]: 
one              2
bar              2
flag         False
foo            bar
one_trunc        2
Name: b, dtype: object

In [84]: df.iloc[2]
Out[84]: 
one             3
bar             3
flag         True
foo           bar
one_trunc     NaN
Name: c, dtype: object
```  
支持布尔运算
```
In [97]: df1 = pd.DataFrame({'a': [1, 0, 1], 'b': [0, 1, 1]}, dtype=bool)

In [98]: df2 = pd.DataFrame({'a': [0, 1, 1], 'b': [1, 1, 0]}, dtype=bool)

In [99]: df1 & df2
Out[99]: 
       a      b
0  False  False
1  False   True
2   True  False

In [100]: df1 | df2
Out[100]: 
      a     b
0  True  True
1  True  True
2  True  True

In [101]: df1 ^ df2
Out[101]: 
       a      b
0   True   True
1   True  False
2  False   True

In [102]: -df1
Out[102]: 
       a      b
0  False   True
1   True  False
2  False  False
```
# 基础用法

### head与tail
```
In [4]: long_series = pd.Series(np.random.randn(1000))

In [5]: long_series.head()
Out[5]: 
0   -1.157892
1   -1.344312
2    0.844885
3    1.075770
4   -0.109050
dtype: float64

In [6]: long_series.tail(3)
Out[6]: 
997   -0.289388
998   -1.020544
999    0.589993
dtype: float64
```
### 描述性统计
Series 与 DataFrame 支持大量计算描述性统计的方法与操作。这些方法大部分都是 sum()、mean()、quantile() 等聚合函数，其输出结果比原始数据集小；此外，还有输出结果与原始数据集同样大小的 cumsum() 、 cumprod() 等函数。  

* count	统计非空值数量
* sum	汇总值
* mean	平均值
* mad	平均绝对偏差
* median	算数中位数
* min	最小值
* max	最大值
* mode	众数
* abs	绝对值
* prod	乘积
* std	贝塞尔校正的样本标准偏差
* var	无偏方差
* sem	平均值的标准误差
* skew	样本偏度 (第三阶)
* kurt	样本峰度 (第四阶)
* quantile	样本分位数 (不同 % 的值)
* cumsum	累加
* cumprod	累乘
* cummax	累积最大值
* cummin	累积最小值