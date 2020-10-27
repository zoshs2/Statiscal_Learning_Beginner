### Chapter 3
# 10. Pearson Correlation
# 11. Spearman's Rank Correlation
# 12. Mean VS. Median
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

