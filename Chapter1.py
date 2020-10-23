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
b = plt.hist(data[data["diagnosis"]=="B"].radius_mean, bins=30, fc=(0,1,0,0.5), label="Bening")
plt.legend()
plt.xlabel("Radius Mean Values")
plt.ylabel("Frequency")
plt.title("Histogram of Radius Mean for Bening and Malignant Tumors")
plt.show()
frequent_malignant_radius_mean = m[0].max() # ?


