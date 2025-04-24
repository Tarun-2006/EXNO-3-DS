## EXNO-3-DS
## Register Number:212223040226
# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:


```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/47391cc9-2ae1-4374-b6f3-9bdda6fb1a9d)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/e84b9bb5-9e3d-4586-bd17-c90b4893a92a)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/63b08a90-7212-4dc7-b505-3f3dad20c4ab)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/55295efa-cb26-4026-bf98-5081461ebaec)
```
from sklearn.preprocessing import OneHotEncoder
df
```
 ![image](https://github.com/user-attachments/assets/b94d74b5-f9af-49a6-ad05-f6515455d3c9)
 ```
ohe=OneHotEncoder(sparse_output=False)
df=df.copy()

enc=pd.DataFrame(ohe.fit_transform(df[['nom_0']]))
enc
```
![image](https://github.com/user-attachments/assets/298f7678-2a06-4ddb-ba56-b82789a6477f)
```
df2=pd.concat([df,enc],axis=1)
df2
```
 ![image](https://github.com/user-attachments/assets/15540fca-f4a2-4b94-98b7-249c0569320f)
```
from category_encoders import BinaryEncoder
df=pd.read_csv(r"C:\Users\admin\Downloads\data (1).csv")
df
```
![image](https://github.com/user-attachments/assets/0c8bcc9c-82a0-4d9f-9714-dabd6abcb3c4)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/4e727019-2647-44a4-b670-81a4fbc77558)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/user-attachments/assets/a1484c9c-add9-4011-967b-811304bc647d)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv(r"C:\Users\admin\Downloads\Data_to_Transform.csv")
df.skew()
```
![image](https://github.com/user-attachments/assets/22a7fbab-66eb-4358-ae34-a0d72380a880)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/3c403ff1-90ea-461f-8643-bd96c020992f)
```
np.reciprocal(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/846b883d-1bcf-44a3-84d8-cc1953f6af3c)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/89e3c66d-7b87-4f75-94dd-86b1b7e8834e)
```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/830baaed-5811-4cbe-8f50-b521d780fa35)
```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/eafc031b-9f3d-4855-9b0d-2e1914d3c346)
```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/6d53194a-4c56-4598-bad8-d386a4429a01)
```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df['Moderate Negative Skew'])
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/00b27c1d-fa97-4110-8476-e80b5bca7192)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/f4343f78-2dd6-4890-860d-7807a34f2523)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/a1fae4e3-3a7f-40df-8583-61051228abb4)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/a5bfdfae-6118-4458-91eb-c9971c072a09)

# RESULT:
Thus the code for Data Transformation is executed successfully.
