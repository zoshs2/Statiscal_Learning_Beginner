### Chapter 3
# 10. Pearson Correlation = (Cov(A,B)) / (Std(A)*Std(B)) or .corr(method='pearson') method!
# 11. Spearman's Rank Correlation () : Pearson 은 outlier영향을 받음. 이 때 Spearman correlation은 그러한 영향이 덜 하기에
#                                        Spearman's (rank) correlation은 robust하다고 말할 수 있다. (data.rank())를 통해 rank화
# 12. Mean VS. Median : Median avoids outliers. (Median활용을 통해 outlier의 영향을 회피할 수 있다.)
# 13. Hypothesis Testing
# 14. Normal(Gaussian) Distribution & z-score

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

## Pearson Correlation (피어슨 상관관계) - 키,몸무게 같은 연속형 자료(continuous data)에 적용가능
# 키, 몸무게와 같이 분석하고자 하는 두 변수가 모두 '연속형 자료(continuous data)'일 때,
# 두 변수간 '선형적인 상관관계의 크기'를 '모수적인(parametric)' 방법으로 나타내는 값이다.
# '피어슨의 적률 상관계수(Pearson's product moment correlation coefficient)'
# '피어슨의 r(Pearson's r)', 'r', 'R' 등은 모두 피어슨 상관계수를 나타내는 다른 용어이다.
# -1 ~ +1 의 값을 가지며, +1과 -1에 가까울 수록 각각 양 또는 음의 상관관계에 있다고 말한다.
# 0 에 가까울 수록(일반적으로 -0.1 < r < 0.1) 상관관계가 없다고 한다.

# Lets look at person correlation between radius mean & area mean.
# First  lets use .corr() pandas method that we used actually at correlation part.
# - 두 변수간 correlation(상관관계)를 분석할 때, 이 .corr() 메소드를 활용한다.
# - 앞으로 배울 스피어만 순위 상관관계(Spearman Rank Correlation)과 켄달 상관계수(Kendall Correlation)
# - 모두 -1에서 +1 사이의 값을 가지며, .corr() 메소드로 구현가능하다. (아마도)
# p1 and p2 is the same. 
# In p1, we use .corr() method, in p2 we apply definition of pearson correlation.
# -> Pearson Correlation Def. : (Cov(A,B)) / (Std(A) * Std(B))
p1 = data.loc[:, ["area_mean", "radius_mean"]].corr(method="pearson") # Data frame 타입으로 출력함
p2 = data.radius_mean.cov(data.area_mean) / (data.radius_mean.std()*data.area_mean.std())
print("Pearson Correlation(p1) : \n", p1)
print("type of .corr(method='pearson') : ", type(p1)) # <class 'pandas.core.frame.Dataframe'>
print("Pearson Correlation(p2) : ", p2)


## Spearman's Rank Correlation (스피어만의 순위 상관관계)
# Pearson's correlation works well if the relationship between variables are linear.
# But! it is not robust, if there are outliers.
# -> 그렇다. Pearson은 연속형 자료에 대한 수치분석을 통해 산출되는 값인데,
#     이는 특정 outlier들의 영향을 크게 받는다. 따라서 이 경우 rank를 매겨 Spearman rank correlation을 사용한다면,
#     outlier에 대한 영향을 최소화할 수 있다.
# To compute spearman's (rank) correlation, first we need to compute 'rank' of each value.
ranked_data = data.rank() # data가 가진 attribute들에서 모두 rank를 매김!! rank로 변환!
print("ranked_data : \n ", ranked_data.head())
spearman_corr = ranked_data.loc[:, ["area_mean", "radius_mean"]].corr(method="spearman")
print("Spearman's correlation : \n", spearman_corr)

# Spearman's rank correlation is little higher than pearson correlation.
# - If relationship between two distributions are non-linear,
#   Spearman's correlation tends to better estimate the strength of relationship.
# - Pearson can be affected by outliers. (Pearson correlation 방법은 outlier영향을 받는다.)
#   Spearman's correlation is more robust. (Spearman correlation 방법은 이러한 영향을 덜 받기에 더욱 robust하다.) -> 확실하다?


## Mean vs. Median (평균 vs. 중간값)
# Sometimes instead of mean, we need to use 'median'. Lets explain why we need to use median with an ex.
# Lets think that there are 10 people who work in a company.
# Boss of the company will make raise in their salary if their mean of salary is smaller than 5.
salary = [1,4,3,2,5,4,2,3,1,500]
print("Mean of salary : ", np.mean(salary)) # Mean of salary : 52.5 

# Mean of Salary is 52.5 so the boss thinks that "Ohh~~ Damnn! I gave a lot of salary to my employees."
# And do not makes raise in their salaries. However, as you know this is not fair and 500(salary) is outlier
# for this salary list.
# Median avoids outliers. 
print("Median of Salary : ", np.median(salary))


## Hypothesis Testing (가설 검정)
# Classical Hypothesis Testing (고전적인 가설검정 기법)
# We want to answer this question : 
# "Given a sample and an apparent effect, what is the probability of seeing such an effect by chance?"
# (by chance : 우연히, 뜻밖에)

# 가설(Hypothesis)을 어떻게 정의할 수 있을까?
# 가설이란 [진실이라고 확증할 수는 없지만, "아마도 그럴 것이다"라는 잠정적인 주장]이라고 정의할 수 있다.
# 연구자(researcher)들은 연구하고자 하는 대상이 나타내는 현상을 관찰한 후, 그 현상을 설명하는 가설을 설정한다.
# 그리고 그 가설(Hypothesis)를 통계적인 방식으로 검정(Testing)한다.
# 우리는 이를 '통계적 가설검정(Hypothesis Testing)'이라고 부른다.
# 통계적 가설검정은 어떠한 큰 이론을 제안하는 가설이 아니다!!
# 통계에서 쓰이는 가설은 우리가 알고 싶어하는 "어떤 모집단의 모수(ex. 평균, 분산, 등)에 대한 잠정적인 주장"이다.
# 따라서 통계적 가설은 일정한 형식을 따라야 한다.
# 그 형식이 바로 "귀무가설(Null Hypothesis; H_0)"과 "대립가설(Alternative Hypothesis, H_1)"이다.
# 즉, 통계적 가설 검정을 실시하려면, 우선 두 가지 형식적 가설(H_0, H_1)을 설정해야한다.
# 그리고 어떠한 가설을 채택할 지는 '확률적으로' 따져보고 둘 중 하나를 채택한다.

# 귀무가설 (Null Hypothesis, H_0)
# - 사전적 정의는 "모집단의 특성에 대해 옳다고 제안하는 잠정적인 주장"이다.
# 쉽게 말해, '모집단의 모수는 ??와 같다.' 또는 '모집단의 모수는 ??와 차이가 없다'라고 가정하는 가설을 말한다.
# ex.1 ) 전국 20세 이상의 평균 키가 170cm라는 주장을 통계적으로 검정한다면,
#        이에 대한 귀무가설(H_0)은 "20세 이상의 성인 남자의 평균 키는 170cm와 같다(or 차이가 없다)."가 될 것이다.
# ex.2 ) 제약회사에서 개발한 신약의 효과를 검정한다면, 
#        이에 대한 귀무가설(H_0)은 "개발한 신약은 효과가 없다(or 차이가 없다)."가 된다.
# 즉, 귀무가설은 '~와 차이가 없다.', '~의 효과는 없다.', '~와 같다.'라는 형식으로 설정된다는 것을 알 수 있다.

# 대립가설 (Alternative Hypothesis, H_1)
# - "귀무가설(H_0)이 거짓이라면, 대안적으로 참이 되는 가설"이다.
# 쉽게 말해, '귀무가설이 틀렸다고 판단했을 때(기각 되었을 때), 대안적으로 선택되는(채택되는) 가설'을 말한다.
# 앞서 설명한 귀무가설의 예시를 대응시킨다면, 대립가설은 
# '모집단의 모수는 ??와 다르다.' 또는 '모집단의 모수는 ??와 차이가 있다.' 정도로 가정하는 것을 말한다.
# 즉, 대립가설은 '~와 차이가 있다.', '~의 효과는 있다.', '~와 다르다.'라는 형식으로 설정된다.

# 자! 이렇게 두 가지 가설을 세웠다면, 우리가 수집한 데이터를 바탕으로 통계적 분석을 수행하여
# 귀무가설(H_0)이 옳아서 accept(채택)할지 또는 reject(기각)할지 판단해야 한다.
# 이를 '귀무가설의 유의성 검정(Null Hypothesis Significance Testing; NHST)'이라고 한다.
# 우리는 표본(sample)을 추출하고, 그 표본으로부터 얻은 정보에 기초하여 귀무가설이 참인지 거짓인지 판정하게 된다.
# 따라서 표본조사라는 본질로 인해, 항상 오류의 가능성은 존재한다.
# 표본(sample)을 추출할 때마다 통계치가 매번 달라질 것이기 떄문이다.
# 따라서 연구자는 수집한 표본을 바탕으로, 
# "귀무가설이 참이라고 가정했을 때, 표본으로부터 얻어지는 통계치(ex. 표본 평균)가 나타날(관측될) 확률"을 계산한다.
# 이 때 계산된 확률값을 p-value라고 한다.
