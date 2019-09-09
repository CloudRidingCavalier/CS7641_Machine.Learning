
# coding: utf-8

# In[17]:

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import pandas as pd
import numpy as np
import random
import os

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

random.seed(2019)


# # Loading dataset

# In[18]:

#Loading dataset
wine = pd.read_csv('./winequality-red.csv')

#Making binary classificaion for the response variable.
#Dividing wine as good and bad by giving the limit for the quality
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)

#Now lets assign a labels to our quality variable
label_quality = LabelEncoder()

#Bad becomes 0 and good becomes 1 
wine['quality'] = label_quality.fit_transform(wine['quality'])

print(wine['quality'].value_counts())

#Now seperate the dataset as response variable and feature variabes
X = wine.drop('quality', axis = 1)
y = wine['quality']

#Applying Standard scaling to get optimized result
sc = StandardScaler()
X  = sc.fit_transform(X)
dataset = pd.DataFrame(X)

#Combine X and y again
transformedData = pd.concat([dataset, y], axis=1, ignore_index=True)
print(transformedData.shape)

#Split
msk = np.random.rand(len(transformedData)) < 0.8
train_test = transformedData[msk]
validate   = transformedData[~msk]
msk = np.random.rand(len(train_test)) < 0.8
train = train_test[msk]
test  = train_test[~msk]

print(train.shape)
print(test.shape)
print(validate.shape)

os.mkdir("./data")
train.to_csv("./data/wine_train.csv", index=False, header=False)
test.to_csv("./data/wine_test.csv", index=False, header=False)
validate.to_csv("./data/wine_validate.csv", index=False, header=False)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



