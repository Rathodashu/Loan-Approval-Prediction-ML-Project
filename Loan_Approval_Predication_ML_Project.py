#!/usr/bin/env python
# coding: utf-8

# ## Loan Approval Predication Machine Learning End to End Project

# In[1]:


### Important libraries Install
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data imported

# In[2]:


df = pd.read_csv(r"C:\Users\Ashok Rathod\Downloads\Dataset.csv")


# ## Data Understanding 

# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


print("The number of the column =",df.shape[1])
print("the numer of the a row's =", df.shape[0])


# In[6]:


df.columns


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[9]:


### Checking the outliers

plt.figure(figsize=(12,8))
sns.boxplot(data = df)
plt.show()


# ## Data Preprocessing and cleaning

# In[10]:


### Fill the null values of numerical datatype

df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mean())


# In[11]:


## Fill the null values of object datatype
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])


# In[12]:


df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])


# In[13]:


df.isnull().sum()


# ## EDA 

# In[14]:


print('Number of people who took loan by gender')
print(df['Gender'].value_counts())
sns.countplot(x = 'Gender', data = df, palette='Set1')
plt.show()


# In[15]:


print("Number of People who took loan bby Married")
print(df['Married'].value_counts())
sns.countplot(x = 'Married', data = df, palette='Set1')
plt.show()


# In[16]:


print("Number of People who took loan bby Education")
print(df['Education'].value_counts())
sns.countplot(x = 'Education', data = df, palette='Set1')
plt.show()


# In[17]:


df.head()


# ## Corr-relation of the numerical data

# In[18]:


corr = df.select_dtypes(include=['int64', 'float64']).corr()


# In[19]:


plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='BuPu')
plt.show()


# ## Feature Engineering

# In[20]:


## Total Applicant Income

df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df.head()


# In[21]:


## Apply Log Transformation

df['ApplicantIncomelog'] = np.log(df['ApplicantIncome']+1)
sns.distplot(df['ApplicantIncomelog'])
plt.show()


# In[22]:


df.head()


# In[23]:


df['LoanAmountlog'] = np.log(df['LoanAmount']+1)
sns.distplot(df['LoanAmountlog'])
plt.show()


# In[24]:


df['Loan_Amount_Term_log'] = np.log(df['Loan_Amount_Term']+1)
sns.distplot(df['Loan_Amount_Term_log'])
plt.show()


# In[25]:


df['Total_Income_log'] = np.log(df['Total_Income']+1)
sns.distplot(df['Total_Income_log'])
plt.show()


# In[26]:


df.head()


# ## Dropping Unnecessary Columns 

# In[27]:


### Drop Unnecessary Columns

cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Total_Income','Loan_ID']
df = df.drop(columns = cols, axis = 1)
df.head()


# ## Encoding : Apply Lable Encoding Method

# In[28]:


## Encoding Technique : Label Encoding, One Hot Encoding

from sklearn.preprocessing import LabelEncoder
cols = ['Gender', 'Married','Education','Dependents','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])


# In[29]:


df.head()


# In[30]:


df.dtypes


# In[31]:


## Split Independent and Dependent Features

x = df.drop(columns = ['Loan_Status'],axis = 1)
y = df['Loan_Status']


# In[32]:


x


# In[33]:


y


# ## Apply Model Algorithms 

# In[34]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25,random_state = 42)


# In[36]:


## Logistic Regression

model1 = LogisticRegression()
model1.fit(X_train,y_train)
y_pred_model1 = model1.predict(X_test)
accuracy = accuracy_score(y_test,y_pred_model1)


# In[37]:


accuracy*100


# In[38]:


## Accuracy : The ratio of the correctly predicted avalues to the total values


# In[39]:


score = cross_val_score(model1,x,y,cv = 5)
score


# In[40]:


np.mean(score)*100


# In[41]:


## Decision Tree Classifier 

model2 = DecisionTreeClassifier()
model2.fit(X_train, y_train)
y_pred_model2 = model2.predict(X_test)
accuracy = accuracy_score(y_test,y_pred_model2)
print('Accuracy score of Decision Tree model: ', accuracy*100)


# In[42]:


score = cross_val_score(model2,x,y,cv = 5)
print('Cross validation score of Decision Tree : ', np.mean(score)*11)


# In[43]:


## Random Forest Classifier

model3 = RandomForestClassifier()
model3.fit(X_train,y_train)
y_pred_model3 = model3.predict(X_test)
accuracy = accuracy_score(y_test,y_pred_model3)
print('Accuracy score of Random Forest model :', accuracy*100)


# In[44]:


## KNeighbor model

model4 = KNeighborsClassifier(n_neighbors=3)
model4.fit(X_train,y_train)
y_pred_model4 = model4.predict(X_test)
accuracy = accuracy_score(y_test,y_pred_model4)
print('Accuracy score of KNeighbor model:', accuracy*100)


# In[45]:


score = cross_val_score(model4,x,y,cv=3)
print('Cross validation score of KNeighbor model:',np.mean(score)*100)


# ## Accuracy Reports

# In[46]:


from sklearn.metrics import classification_report

def generate_classification_report(model_name, y_test, y_pred):
    report = classification_report(y_test,y_pred)
    print(f"Classification Report For {model_name}:\n{report}\n")

generate_classification_report(model1,y_test,y_pred_model1)
generate_classification_report(model2,y_test,y_pred_model2)
generate_classification_report(model3,y_test,y_pred_model3)
generate_classification_report(model4,y_test,y_pred_model4)
    


# In[47]:


df['Loan_Status'].value_counts()


# ## Resampling 

# In[48]:


from imblearn.over_sampling import RandomOverSampler


# In[49]:


oversample = RandomOverSampler(random_state=42)
x_resampled, y_resampled = oversample.fit_resample(x,y)

df_resample = pd.concat([pd.DataFrame(x_resampled,columns=x.columns),pd.Series(y_resampled,name='Loan_status')],axis =1)


# In[50]:


x_resampled


# In[51]:


y_resampled


# In[52]:


y_resampled.value_counts()


# In[53]:


x_resampled_train, x_resampled_test, y_resampled_train, y_resampled_test = train_test_split(x_resampled,y_resampled,test_size = 0.25,random_state=42)


# In[54]:


## Logistic Regression resampled
model1 = LogisticRegression()
model1.fit(x_resampled_train,y_resampled_train)
y_pred_model1 = model1.predict(x_resampled_test)
accuracy = accuracy_score(y_resampled_test,y_pred_model1)
accuracy*100


# In[55]:


## Decision Tree Classifier Resampled

model2 = DecisionTreeClassifier()
model2.fit(x_resampled_train, y_resampled_train)
y_pred_model2 = model2.predict(x_resampled_test)
accuracy = accuracy_score(y_resampled_test,y_pred_model2)
print('Accuracy score of Decision Tree model: ', accuracy*100)


# In[56]:


## Random Forest Classifier Resampled

model3 = RandomForestClassifier()
model3.fit(x_resampled_train,y_resampled_train)
y_pred_model3 = model3.predict(x_resampled_test)
accuracy = accuracy_score(y_resampled_test,y_pred_model3)
print('Accuracy score of Random Forest model :', accuracy*100)


# In[57]:


## KNeighbor model resampled

model4 = KNeighborsClassifier(n_neighbors=3)
model4.fit(x_resampled_train,y_resampled_train)
y_pred_model4 = model4.predict(x_resampled_test)
accuracy = accuracy_score(y_resampled_test,y_pred_model4)
print('Accuracy score of KNeighbor model:', accuracy*100)


# In[58]:


from sklearn.metrics import classification_report

def generate_classification_report(model_name, y_test, y_pred):
    report = classification_report(y_test,y_pred)
    print(f"Classification Report For {model_name}:\n{report}\n")

generate_classification_report(model1,y_resampled_test,y_pred_model1)
generate_classification_report(model2,y_resampled_test,y_pred_model2)
generate_classification_report(model3,y_resampled_test,y_pred_model3)
generate_classification_report(model4,y_resampled_test,y_pred_model4)


# In[59]:


## The good predication of the Random Forest Classification model


# In[60]:


import joblib
joblib.dump(df, 'final_model_Loan_Approval_Predication.pkl')

