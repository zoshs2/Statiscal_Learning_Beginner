# Central Limit Theorem (중심극한정리) 
# 표본 수가 적당히 크다면(generally, n>=30), 표본 평균의 분포는 
# 모집단의 특성(평균이 mu, 표준편차가 sigma)을 가진 정규분포(평균 mu, 표준편차 sigma/sqrt(n))를 이룬다. 
# 추출한 표본의 분포 형태가 어떤 분포모양을 갖던 그건 상관없다.
# 그러한 표본 분포들의 평균들의 분포가 모집단의 특성을 가진 정규분포를 이룬다는 것이 중요한 것이다.
# -> Even when a sample is not normally distributed, if you draw multiple samples and take each of their
#    averages, these averages will represent a normal distribution.

# 1. 지수분포를 이루는 표본이라는 상황가정에서 출발
#     - 표본 수가 30개라고 특정
#     - 이 30개의 표본들을 여러 번(times) 추출해서 각각 평균들이 모여 이루는 분포가 정규분포인지 확인해보자.
from matplotlib.pyplot import xlabel
import numpy as np
from scipy.stats import expon, zscore, norm
# norm(mu, sigma).pdf(x) : 평균mu, 표준편차 sigma를 가지는 정규분포(norm)를 가지는 배열을 반환
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import os
for dirname, _, filenames in os.walk("/Users/yungi/Documents/Hello_Atom/Data_anaylsis2/input"):
    for filename in filenames:
        data_path = os.path.join(dirname, filename)

data = pd.read_csv(data_path)

# times = 30
# l = 10 
# loc = 0
# lambda가 10인 지수분포를 랜덤으로 30개 생성해서 그 평균을 구하여 30번 반복한 뒤 histogram을 생성해볼 것이다.
# exponential distribution (scipy.stats.expon(loc, lambda)):
# f(x; lambda) = lambda * exp(-lambda * x) for x >= loc
#              = 0 for x < loc

def test(times):
    t = times # 표본 반복추출 횟수
    loc = 0
    lamb = 10
    samples_mean = []

    for _ in range(times):
        samples_mean.append(np.mean(expon(loc, lamb).rvs(size=30))) # sizes : 샘플(표본) 수 

    z = zscore(samples_mean) # normalization ? 표준화 -> 평균이 0, 표준편차가 1인 표준정규분포(standard normal distribution)로 만듬
    bins = int(6 * np.log10(t))

    fig, axes = plt.subplots(1,2,figsize=(12,6))
    fig.suptitle(r"Histogram of Random Exponential with 30 rvs size ($\lambda = 10, times = $"+str(t)+")")
    axes[0].hist(samples_mean, bins=bins, facecolor='wheat')
    axes[0].set(xlabel='samples_mean', ylabel='frequency')
    axes[0].set_title("Samples_mean's distribution")
    # rvs = random variates 
    axes[0].grid()

    x = np.linspace(-3, 3, 101)

    axes[1].hist(z, bins=bins, density=True, facecolor='skyblue')
    axes[1].plot(x, norm(0,1).pdf(x), 'r--')
    axes[1].set(xlabel='normalized distribution of samples_mean', ylabel='density')
    axes[1].set_title("Standardily Normalized by zscore(samples_mean)")
    axes[1].grid()

    plt.show()

# print(test(30)) # 추출 반복횟수가 너무 적어서, 30개따리 표본평균들의 분포가 정규분포가 되는지 알아보기 쉽지 않다.
# print(test(1000)) # 1000회 정도 반복추출해보니, 표본평균분포가 정규분포모습을 점점더 선명히 보이고 있다.
# print(test(10000)) # (100000회 반복추출부터 컴퓨터가 좀 힘들어하네) 확실한 정규분포모습을 보인다.



# 표본 수를 일반적인 30개로 설정하여, 추출해본 결과
# 표본평균분포가 정규분포모습을 가진다는 것을 확인해볼 수 있었다.
# 2. 그렇다면 과연 표본정규분포가 정말 모집단의 모수 특성을 가지고 있을까?
#    - bening and malignant dataset으로 한번 해보자
# pop : population (모집단), sam : sample (표본)
# bening과 malignant의 분포는 서로 다르지만, 모집단의 분포모양은 어차피 상관이 없어서 그냥 합쳐서 진행하기로 함.
print(data.radius_mean.describe()) # 569개의 모집단 수
pop_radius_mean = data[["radius_mean"]]
pop_avg = np.mean(pop_radius_mean)
# 오호! cmd + / 하면, 해당 커서줄 주석처리 on/off
print(type(pop_radius_mean)) # data frame

sampleMeans = []
sam_n = 100
for _ in range(10000):
    samples = pop_radius_mean["radius_mean"].sample(n=sam_n) # 표본 수 100개 랜덤 추출
    sampleMean = np.mean(samples)
    sampleMeans.append(sampleMean)

sam_avg = np.mean(sampleMeans)
print("radius_mean - 모집단 평균 : ", pop_avg)
print("radius_mean - 표본평균분포의 평균 : ", sam_avg) 

sam_std = np.std(sampleMeans)
pop_std = np.std(pop_radius_mean)
print("radius_mean - 모집단 표준편차 : ", pop_std)
print("radius_mean - 표본평균분포의 표준편차 : ", sam_std)

# 표본평균분포의 표준편차 = 모집단의 표준편차 / sqrt(n) ( where n is the number of sample)
resulting_std = pop_std / np.sqrt(sam_n)
print("모집단의 표준편차 / sqrt(n) = ", resulting_std)
print("표본평균분포의 표준편차 = ", sam_std)