import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# from pandas.tools import plotting
from scipy import stats
plt.style.use("ggplot")

import warnings
warnings.filterwarnings("ignore")
import os
os.system("pwd")
# print(os.listdir("/Users/yungi/Documents/Hello_Atom/Data_anaylsis2/input"))

for dirname, _, filenames in os.walk("/Users/yungi/Documents/Hello_Atom/Data_anaylsis2/input"):
    for filename in filenames:
        data_path = os.path.join(dirname, filename)

data = pd.read_csv(data_path)
print(data.head())


print(data.columns)

data = data.drop(['Unnamed: 32', 'id'], axis=1) # delete selected columns (Unnamed: 32 & id cols) along the axis 1
print(data.head())
print(data.shape)
print(data.columns)

## Histogram (히스토그램)
# How many times each value appears in dataset.
# - This description is called 'the distribution of variable'.
# Most common way to represent distribution of variable 
# is Histogram!! that is graph which shows frequency of each value.
# Frequency = number of times each value appears.
# Ex) [1,1,1,1,2,2,2]. Frequency of 1 is four 
# and frequency of 2 is three.

# diagnosis가 M인 데이터셋들의 radius_mean (variable)의 value별 frequnecy를 보기 위한 histo이다.
# Malignant : 악성의, 극히 해로운 (악성종양할 때 악성)
m = plt.hist(data[data["diagnosis"]=="M"].radius_mean, bins=30, fc=(1,0,0,0.5), label="Malignant")
# m[0] : (plt.hist) 히스토그램에 대한 frequency for each value 를 np.array로 담고 있다.
# m[1] : (plt.hist) 히스토그램에 대한 bin 경계값들에 대한 정보를 np.array로 담고 있다.
#         - 이 경계값은 bin의 앞쪽 값을 가리킨다.
# m[2] : (plt.hist) <BarContainer object of 30 artists> 

b = plt.hist(data[data["diagnosis"]=="B"].radius_mean, bins=30, fc=(0,1,0,0.5), label="Bening")
'''
plt.legend()
plt.xlabel("Radius Mean Values")
plt.ylabel("Frequency")
plt.title("Histogram of Radius Mean for Bening and Malignant Tumors")
plt.show()
'''

frequent_malignant_radius_mean = m[0].max() # 가장 큰 frequency 값 (m[0])
print("type(m[0]) : ", type(m[0]) ) # <class 'numpy.ndarray'>
print("m[0].max() : ", m[0].max())
print("list(m[1]) : ", list(m[1]))
print("np.where(m[0]==frequent_malignant_radius_mean) : ", np.where(m[0]==frequent_malignant_radius_mean))

# numpy.ndarray에는 .index라는 메소드(원하는 원소값의 인덱스를 반환해주는 메소드)가 없다. 
# 따라서 이 메뉴얼에서는 list(m[0])로 list로 먼저 형변환 후 .index() 메소드를 활용한다.
# numpy.ndarray object에 리스트.index와 같은 메소드를 활용하려면 np.where(myarr==given_val)을 쓰면 된다.
# 다만!!! 찾고자 하는 값이 여러 개 존재할 때!!
# list의 .index 메소드는 가장 먼저 오는 (첫 번째) 값에 대한 인덱스만 반환하고,
# numpy의 np.where()함수는 모든 해당 값에 대한 인덱스를 모두 찾아준다.
# 메뉴얼로 시작
index_frequent_malignant_radius_mean = list(m[0]).index(frequent_malignant_radius_mean)
most_frequent_malignant_radius_mean = m[1][index_frequent_malignant_radius_mean]
print("Most frequent malignant radius mean is : ", most_frequent_malignant_radius_mean)

# Lets look at other conclusions
# From this graph you can see that radius mean of malignant tumors are bigger 
# than radius mean of bening tumors mostly. 
# 즉, 시각화를 활용한 탐색적 분석(Exploratory Data Analysis; EDA)를 통해 
# 위와 같은 대략적인 서술 분석(Description analysis)을 수행할 수 있다.
# malignant tumor(악성종양)의 평균 크기가 bening tumor(일반적인 종양)보다 크다는 사실.

# The bening distribution is approximately bell-shaped that is shaped of normal distribution.
# 또한, 일반적인 종양 크기의 분포는 대략적으로 '종 모양'을 하고 있으며, 이는 정규분포와 비슷함을 알 수 있었다.

# Also you can find result like that most frequent malignant radius is 20.101999999
# 또한, 가장 많이 측정되는 악성 종양의 반지름 크기는 20.1019999임도 알 수 있었다.


## Outliers (이상치)
# While looking histogram as you can see, there are rare values in bening distribution.
# (사실 malignant distribution에도 outlier가 있는거 같긴 하다.)
# Those values can be errors or rare events. (이는 에러일 수도 있고, 그냥 희귀한 실제 경우일 수도 있다.)
# Calculating outliers :
# - first we need to calculate first quartile (Q1)(25%)
# - then find IQR(inter-quartile range) = Q3 - Q1 (Q2에서 양쪽으로 각각 %25씩의 범위)
# - finally compute Q1 - 1.5 * IQR and Q3 + 1.5 * IQR 
# - Anything outside this range is an "outlier(이상치)"
# - lets write the code for bening tumor distribution for feature radius mean.

data_bening = data[data["diagnosis"]=="B"] # Bening인 tumor들에 대한 것만 뽑은 data frame.
# print("type(data_bening) : ", type(data_bening)) # <class 'pandas.core.frame.DataFrame'>
data_malignant = data[data["diagnosis"]=="M"]
desc = data_bening.radius_mean.describe()
print("data_bening.radius_mean.describe() : ", data_bening.radius_mean.describe())
# describe()메소드 : 데이터 갯수(count), mean, std, min, Q1, Q2, Q3, max 을 보여준다.
Q1 = desc[4] # describe()메소드로 출력된 결과는 desc[1~8] 로 값을 따로 조회할 수 있다. desc[4]는 Q1이다.
Q3 = desc[6] # Q3
IQR = Q3 - Q1 # inter-quartile range
lower_bound = Q1 - 1.5*IQR # 이 이하의 값들은 outlier라 할거다.
upper_bound = Q3 + 1.5*IQR # 이 이상의 값들은 outlier라 할거다.
print("Anything outside this range is an outlier : (", lower_bound, ", ", upper_bound, ")")
print("Outliers : ", data_bening[np.logical_or(data_bening.radius_mean < lower_bound, data_bening.radius_mean > upper_bound)].radius_mean.values)
# np.logical_or을 사용하였다.
# 아래와 같이 bitwise 연산자를 활용해도 좋다.
print("Outliers : ", data_bening[(data_bening.radius_mean < lower_bound) | (data_bening.radius_mean > upper_bound)].radius_mean.values)
# | : bitwise OR , & : bitwise AND, ~ : bitwise NOT, ^ : bitwise XOR


## Box Plot 
# You can see outliers also from box plots (위와 같이 analytic하게 푸는 방법도 있고, box plots을 활용하여 이상치를 보는 방법도 있다.)
# We found 3 outliers in bening radius mean and in box plot there are also 3 outliers.
melted_data = pd.melt(data, id_vars="diagnosis", value_vars=["radius_mean", "texture_mean"])
print("melted_data : \n", melted_data)
plt.figure(figsize=(15,10))
# visualtion tool : seaborn (as sns)
sns.boxplot(x="variable", y="value", hue="diagnosis", data=melted_data)
# box plot 을 그릴땐 melt시킨 걸로!! x축에 variable들을, y축엔 value를, hue는 각 variable에 대해 hue의 종류들로 나눠서 표시하라는 뜻!
plt.show()
# bening tumor인 것들 중에 texture_mean의 이상치가 좀 많다는 것을 알 수 있다.

## Summary Statistics (아주아주 기초적인 통계)
# Mean : 산술평균
# Variance : 분산 (spread of distribution; 분포의 퍼짐 정도)
# Standard deviation (square root of variance; 표준편차(분산의 제곱근))
# Lets look at summary statistics of bening tumor radiance mean
print("mean : ", data_bening.radius_mean.mean())
print("variance : ", data_bening.radius_mean.var())
print("standard deviation (std) : ", data_bening.radius_mean.std())
print("describe method : ", data_bening.radius_mean.describe())

