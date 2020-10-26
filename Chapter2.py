### Chapter 2
# 1. CDF (Cumulative Distribution Function)
# 2. Effect Size (feat. pooled & unpooled variance)
# 3. Relationship Between Variables (두 변수 사이의 관계)


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
data_bening = data[data['diagnosis']=="B"]
data_malignant = data[data['diagnosis']=="M"]

## CDF (Cumulative Distribution Function : 누적분포함수)
# CDF is the probability that the variable takes a value less than or equal to x. P(X <= x)
# CDF(누적분포함수)는 '랜덤변수가 특정 값(x)보다 작거나 같을 확률'이다.
# Cumulative(누적)이라는 이름에서도 알 수 있듯이 '특정 값보다 작은 값들의 확률을 모두 누적!해서 구한다'는 의미에서 
# 붙여진 이름이다.
# => 특정 값보다 작거나 같을 확률
# Q. 어떤 값 x이하의(작거나 같을) 확률은? -> A. P(X<=x)라 표기한다. 

# Lets explain in cdf graph of bening radius mean
# In graph, what is P(X<=12)? The answer is 0.5
# The probability that the variable takes a values less than or equal to 12 is 0.5

# You can plot cdf with two different method.
'''
plt.hist(data_bening.radius_mean, bins=50, fc=(0,1,0,0.5), label="Bening", density=True , cumulative=True)
# normed=True caused AttributeError : The reason why it's happened is the normed option is now deprecated(권장X), 
# recent in maplotlib, they recommend you to use "density=True" option as normed=True.
sorted_data = np.sort(data_bening.radius_mean) # 오름차순 정렬 
y = np.arange(len(sorted_data))/float(len(sorted_data)-1)  # I don't understand this formula.
plt.plot(sorted_data,y,color='red')
plt.title("CDF of bening tumor radius mean")
plt.show()
'''

## Effect Size (효과크기)
# : standardized measure of the size of the mean difference among the study groups
# It describes size of an effect. It is simple way of quantifying the difference between two groups.
# It is similar with p-value of Inferential statistics(추론 통계학).
# p-value of Inferential Statistics strongly depends on the number of samples.
# 또한 p-value는 표본수가 엄청나게 많은 상황이고, 두 집단간의 연관성이 부족해도,
# 몇 개의 표본으로 인한 overpower문제로 p<0.05인 현상이 벌어져 맞지않은 결과를 내게 된다.
# 반면, 효과크기는 실제 관찰된 데이터에서 비교하려는 집단 사이의 차이를 '직접적으로(있는 그대로)' 기술한다는 장점이 있다.
# Effect size emphasises the size of the 'difference'.

# Use 'Cohen Effect size'. (Officially, it is called "Cohen's d".)
# Cohen defined the Effect size to be standardized as Small (d=0.2), Medium (d=0.5), Large (d=0.8).
# Lets compare size of the effect size between bening radius mean and malignant radius mean.
# Effect size is 2.2 that is too big and says that 
# the two groups (bening & malignant radius means) are very different from each other as we expect.
# Because our groups are bening radius mean and malignant radius mean that are different from each other.
mean_diff = data_malignant.radius_mean.mean() - data_bening.radius_mean.mean() # 두 집단의 평균 차이 
var_bening = data_bening.radius_mean.var() # bening그룹의 분산(variance)
var_malignant = data_malignant.radius_mean.var() # malignant그룹의 분산(variance)
var_pooled = (len(data_bening)*var_bening + len(data_malignant)*var_malignant) / float(len(data_bening)+len(data_malignant))
# pooled varaince (합동분산?) : 간단하게 개념을 말하자면, 
# 두 집단의 variance를 pooling하여 1개의 분산으로 표현한다는 의미이다.
# pooled varaince 는 언제나 양수(+)이다. 
# unpooled varaince도 있는데, 
# pooled을 쓸지, unpooled 를 쓸지는, 두 집단의 분산이 거의 2배이상 차이가 난다면, unpooled variance를 쓰는게 좋다. (고 한다.)
# 반면, 두 집단의 분산이 거의 비슷하거나 같다면, pooled variance 를 사용하는게 좋다. (고 한다.)
# pooled(or unpooled) variance를 구하는 방법은
# 기본적으로 '산술평균'을 떠올리면 된다.
# 각 집단의 표본수(the number of samples)를 고려하여 가중한다.
# ex) pooled variance = ( N * var_1 + M * var_2 ) / N + M

# 단위에 구속받지 않는 standardized effect size를 구하기 위해
# 두 mean 차이에 standard deviation (표준편차; pooled variance 에서 비롯됨)를 나눠준다.
print("mean of the bening : ", data_bening.radius_mean.mean())
print("mean of the malignant : ", data_malignant.radius_mean.mean())
print("diff between the means : ", mean_diff)
effect_size = mean_diff / np.sqrt(var_pooled)
print("Effect size : ", effect_size) # Cohen's d => 2.2 ((really) Large) : 이 정도면 연관성이 거의 없다고 봐야지.

## Relationship Between Variables
# Scatter Plot
# For example, price and distance. If you go long distance with taxi, you will pay more.
# Therefore we can say that price and distance are positively related with each other.

# Lets look at relationship between radius mean and area mean.
# In scatter plot you can see that when radius mean increases, area mean also increases.
# Therefore, they are positively correlated with each other.
'''
plt.figure(figsize=(15,10))
sns.jointplot(data.radius_mean, data.area_mean, kind="reg")
plt.show()
'''

print(data.loc[:,["radius_mean", "area_mean", "fractal_dimension_se"]]) # type of data frame.
print("type of this : ", type(data.loc[:,["radius_mean", "area_mean", "fractal_dimension_se"]]))

## Correlation (연관성)
# Strength of the relationship between two variables.
# The correlation range are -1 to +1
# Meaning of +1 is two variables are positively correlated with each other. (ex. radius mean & area mean)
# Meaning of 0 is there is 'no correlation' between two variables (ex. radius mean & fractal dimension se)
# Meaning of -1 is two variables are negatively correlated with each other. (ex. radius mean & fractal dimension)

# Lets look at correlation between all features.
# Using Heatmap!
print("data.corr() : \n", data.corr())
f, ax = plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(), annot=True, linewidths=0.5, fmt=".1f", ax=ax)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title("Correlation Map")
plt.show()

## Covariance (공분산)
# Covariance is measure of the tendency of two variables to vary together.
# ( 공분산은 두 변수가 변화할 때, 함께 변화하는 경향에 대한 값이다. )
# Cov(X, Y) : Dependence of realizations of 2 (or more) different RV(Random Variable)s.
# 공분산의 수학적 정의는 Cov(X, Y) = E{(X - mean_X)(Y - mean_Y)}
# 이를 해석해보면, 랜덤변수 X의 편차(deviation)와 랜덤변수 Y의 편차 곱한 결과의 평균이다.
# - 공분산은 랜덤변수 간 서로에 대한 의존성을 나타낸다.
# - X가 커지면, Y도 커진다 또는 X가 작아지면, Y도 작아진다. (vice versa Y, X) : 이 때 Covariance 는 양수값을 갖는다.
#   - 서로에 대한 의존성이 클 수록 큰 양수값을 갖는다.
# - 랜덤변수 간 서로에 대한 의존성이 낮다면, Covariance 는 0에 가까운 값을 갖는다.
# - 그렇다면 Covariance(X,Y)가 0이라면, 두 변수는 서로 독립변수인가? : 꼭 그렇지는 않다!
# 


