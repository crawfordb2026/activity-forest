import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12,6)

train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')


print (test_df.isnull().sum().loc[lambda x: x > 0])
