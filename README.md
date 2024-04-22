## EXNO-3-DS

## AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

## ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

## FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

## Methods Used for Data Transformation:
 #### 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
 #### 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

## CODING AND OUTPUT:

DEVELOPED BY : NAVEEN RAJA N.R

REG NO : 212222230093
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
df=pd.read_csv('/content/Encoding Data.csv')
df
```


![image](https://github.com/LATHIKESHWARAN/EXNO-3-DS/assets/119393556/69912819-7283-407e-91dd-a98bacd10907)


```
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

![image](https://github.com/LATHIKESHWARAN/EXNO-3-DS/assets/119393556/149fd26a-608f-4b4f-bb17-d3b1cfb2c5ed)

```p
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

![image](https://github.com/LATHIKESHWARAN/EXNO-3-DS/assets/119393556/834dcaa5-ff95-49d1-8e6f-557b31a30c88)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```


![image](https://github.com/LATHIKESHWARAN/EXNO-3-DS/assets/119393556/9b0c7865-75e9-4ccf-88f8-d495700488ac)

```
on=OneHotEncoder(sparse=False)
df2=df.copy()
en=pd.DataFrame(on.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,en],axis=1)
pd.get_dummies(df2,columns=["nom_0"])
```

![image](https://github.com/LATHIKESHWARAN/EXNO-3-DS/assets/119393556/7bb2cdb6-9a1f-48f0-afef-9873a0a816d9)

```
from category_encoders import BinaryEncoder
fd=pd.read_csv('/content/data.csv')
fd
```

![image](https://github.com/LATHIKESHWARAN/EXNO-3-DS/assets/119393556/8904b3e1-787a-4c11-ad13-3196e2a87819)

```
be=BinaryEncoder()
nd=be.fit_transform(fd['Ord_2'])
fd
```

![image](https://github.com/LATHIKESHWARAN/EXNO-3-DS/assets/119393556/490ab4ea-fd32-4fdf-bdf5-be349dd6caa1)

```
dfb=pd.concat([fd,nd],axis=1)
dfb=fd.copy()
dfb
```

![image](https://github.com/LATHIKESHWARAN/EXNO-3-DS/assets/119393556/14f0b0ed-6cc3-450e-a3d3-b272ad4144a0)

```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=fd.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```


![image](https://github.com/LATHIKESHWARAN/EXNO-3-DS/assets/119393556/24e5c303-4275-45f2-be66-a2949b9354cb)

```
from scipy import stats
import numpy as np
ab=pd.read_csv('/content/Data_to_Transform.csv')
ab
```

![image](https://github.com/LATHIKESHWARAN/EXNO-3-DS/assets/119393556/870be182-a1ce-41dc-862f-53bd6a49acbc)

```
ab.skew()
```
![image](https://github.com/LATHIKESHWARAN/EXNO-3-DS/assets/119393556/04c6ef56-caee-41ef-9228-64728a0f5351)

```
np.log(ab['Highly Positive Skew'])
```

![image](https://github.com/LATHIKESHWARAN/EXNO-3-DS/assets/119393556/edd7a723-f153-448d-8c38-0890eec0cdc7)

```
np.reciprocal(ab["Moderate Negative Skew"])
```

![image](https://github.com/LATHIKESHWARAN/EXNO-3-DS/assets/119393556/134955fc-c21e-41ca-abe8-444d6bb79c6d)

```
np.sqrt(ab["Highly Negative Skew"])
```

![image](https://github.com/LATHIKESHWARAN/EXNO-3-DS/assets/119393556/ab0039c8-bfb6-4dc8-9739-aea4ece8a86f)

```
np.square(ab["Highly Positive Skew"])
```

![image](https://github.com/LATHIKESHWARAN/EXNO-3-DS/assets/119393556/c20568ec-d2a9-4574-be7e-1e9d9ae4a339)

```
ab['Highly Positive Skew_boxcox'], parameters=stats.boxcox(ab['Highly Positive Skew'])
ab
```

![image](https://github.com/LATHIKESHWARAN/EXNO-3-DS/assets/119393556/3b9e770a-e94c-4e0e-9ff3-8e98bf2ed9df)

```
ab.skew()
```

![image](https://github.com/LATHIKESHWARAN/EXNO-3-DS/assets/119393556/701906f2-7a18-4935-b358-235eca41ead1)

```
ab['Moderate Negative Skew_yeojohnson'], parameters=stats.yeojohnson(ab['Moderate Negative Skew'])
ab
```

![image](https://github.com/LATHIKESHWARAN/EXNO-3-DS/assets/119393556/25c7303c-6ff7-44c4-a4ad-8c215723ec5e)

```
ab.skew()

```

![image](https://github.com/LATHIKESHWARAN/EXNO-3-DS/assets/119393556/bca201b5-35f9-4004-a86f-e91a1f5d675e)


```
ab['Highly Negative Skew_yeojohnson'], parameters=stats.yeojohnson(ab['Highly Negative Skew'])
ab
```

![image](https://github.com/LATHIKESHWARAN/EXNO-3-DS/assets/119393556/60545326-ed1f-43a4-9b26-d7a448aa5295)


```
ab.skew()
```

![image](https://github.com/LATHIKESHWARAN/EXNO-3-DS/assets/119393556/80e2b38f-d248-4280-9eee-73a6d6e981f2)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
ab["Moderate Negative Skew_1"]=qt.fit_transform(ab[["Moderate Negative Skew"]])
ab
```

![image](https://github.com/LATHIKESHWARAN/EXNO-3-DS/assets/119393556/257c87c3-96ab-4202-ae85-f9845f8ea1d6)


```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(ab["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/LATHIKESHWARAN/EXNO-3-DS/assets/119393556/d7c76ad6-2ab6-4e03-bb3b-b55981a179e2)

```
sm.qqplot(np.reciprocal(ab["Moderate Negative Skew"]),line='45')
```

![image](https://github.com/LATHIKESHWARAN/EXNO-3-DS/assets/119393556/85682389-ed02-4aa8-be54-b9ceb372855a)


```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
ab["Moderate Negative Skew"]=qt.fit_transform(ab[["Moderate Negative Skew"]])
sm.qqplot(ab["Moderate Negative Skew"],line="45")
plt.show()
```
![image](https://github.com/LATHIKESHWARAN/EXNO-3-DS/assets/119393556/789a3105-3ffe-43e9-bd6d-a1652ddcfef4)


## RESULT:
       Hence performing Feature Encoding and Transformation process is Successful.
