#!/usr/bin/env python
# coding: utf-8

# In[94]:


#Details from the dataset 

#Education: 1 'Below College', 2 'College', 3 'Bachelor', 4 'Master', 5 'Doctor'
#EnvironmentSatisfaction: 1 'Low', 2 'Medium', 3 'High', 4 'Very High'
#JobInvolvement: 1 'Low', 2 'Medium', 3 'High', 4 'Very High'
#JobSatisfaction: 1 'Low', 2 'Medium', 3 'High', 4 'Very High'
#PerformanceRating: 1 'Low', 2 'Good', 3 'Excellent', 4 'Outstanding'
#RelationshipSatisfaction: 1 'Low', 2 'Medium', 3 'High', 4 'Very High'
#WorkLifeBalance: 1 'Bad', 2 'Good', 3 'Better', 4 'Best'


# In[128]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# In[96]:


data = pd.read_csv('HR Employee Attrition.csv')


# In[97]:


data.head()


# In[98]:


data=data.drop('EmployeeNumber', axis = 1)


# In[99]:


data.shape


# In[100]:


#checking for null values
data.isnull().sum()


# In[101]:


data.info()


# In[102]:


# variable(s) with constant value
val = data.apply(lambda x: len(x.unique()))
val[val ==1 ].index


# In[103]:


data.drop(['EmployeeCount', 'Over18', 'StandardHours'], axis = 1, inplace = True)


# In[104]:


data.hist(figsize = (15,15))
plt.tight_layout()
plt.show()


#      Following are some inferences from the histogram plots. 
# Most distributions are right-skewed (Monthly Income, Total Working Years, Year at Company, Distance From Home, etc.  
# The age feature is a little right-skewed, and most of the employees have ages between 25–40 years. 

# In[105]:


sns.kdeplot(data.loc[data['Attrition'] == 'No', 'Age'], label = 'Active Employee')
sns.kdeplot(data.loc[data['Attrition'] == 'Yes', 'Age'], label = 'Ex-Employees')

plt.legend()
plt.show()


# Ex-employees have an average age of 30 years, while the current employees have 36 years. A younger employee is more likely to leave a company, and the education and marital status parameters are potential support.

# In[106]:


sns.countplot(data['Department'])


# counting the number of employees in each department, we can see that Research & Development has the biggest number of employees

# In[107]:


f, ax = plt.subplots(figsize=(10,7))
sns.swarmplot(x= data["TotalWorkingYears"], y= data["Department"]).set( title= 'TotalWorkingYears to Department')
plt.show()


# Seeing the total working years for employees of each department, we can see that Research & Development also takes the lead, and the human resources has the least amount of working years for employees

# In[108]:


sns.histplot(data=data,x='YearsAtCompany',hue='Attrition')
plt.show()


# Seeing the total working years for employees and the rate of attrition, we can see that the more working years the employees have, the less likely they are to leave the job. Also, employees who have worked for 1 and 10 years have the greater likelihood to leave the job..

# In[109]:


sns.boxplot(y=data["MonthlyIncome"], x=data["JobRole"])
plt.grid(True, alpha=1)
plt.tight_layout()
plt.show()


# Managers and Research directors earn a relatively large sum. Intuitively, the attrition must have an inverse relationship with the monthly income parameter. 
# Research Scientist, Lab. Technicians and Sales Representatives positions are not well paid. Such factors would lead to attrition in these departments.

# In[110]:


# Check numeric variables
numeric_data=data.select_dtypes(include=['int64'])


# In[111]:


def outlier_summary(data):
    print(f'Count of outliers:')
    outlier_var = 0
    for val in data.columns:
        Q1,Q3 = np.percentile(data[val].sort_values(), [25,75])
        IQR = Q3 - Q1
        min_ = Q1 - (1.5 * IQR)
        max_ = Q3 + (1.5 * IQR)

        tagging = data[val].apply(lambda x: 1 if (x < min_ or x > max_) else 0)
        count = tagging[tagging == 1].sum()
        if count > 0: outlier_var = outlier_var + 1
        print(f'  {val} -- Count: {count} Percentage: {int(round(count/data.shape[0]*100, 0))}%')
    print(f'Total number of variables with outliers: {outlier_var}')
    
outlier_summary(numeric_data)


# In[112]:


# get categorical variables
data.select_dtypes(include=['object'])


# In[113]:


data.head()


# In[114]:


corr=data.corr()


# In[115]:


mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, cmap=cmap, mask=mask, vmax=.5, linewidths=.2, )
plt.show()


# In[116]:


# label encode the target variable
le = LabelEncoder()
y = pd.Series(le.fit_transform(data.Attrition))


# In[118]:


label_summary = pd.DataFrame(zip(y, data.Attrition), columns = ['label_encode', 'Attrition'])    .groupby(['label_encode', 'Attrition']).size().to_frame('count').reset_index()
label_summary['%'] = round(label_summary['count']/label_summary['count'].sum()*100, 1)
label_summary


# In[120]:


X_train, X_test, y_train, y_test = train_test_split(data.drop('Attrition', axis = 1), y, test_size = 0.2,
                                                    random_state = 778, shuffle = True, stratify = y)

# check distribution
pd.DataFrame({'Count - Train': y_train.value_counts(), '% - Train': round(y_train.value_counts(1)*100, 1),
              'Count - Test': y_test.value_counts(), '% - Test': round(y_test.value_counts(1)*100, 1)})


# In[ ]:





# In[121]:


from sklearn.compose import ColumnTransformer


# In[122]:


nominal = X_train.select_dtypes(include=['object']).columns
ohe = ColumnTransformer([('encoder', OneHotEncoder(), nominal)], remainder='passthrough')


# In[123]:


ohe.fit(X_train)
X_train_new = ohe.transform(X_train)
X_test_new = ohe.transform(X_test)


# In[140]:


models = [DecisionTreeClassifier(), RandomForestClassifier(), LogisticRegression()]
models_name = [ 'Decision Tree', 'Random Forest', 'Logistic Regression']

for i in range(0, len(models)):
    cv_results = cross_validate(models[i], X_train_new, y_train,
                            cv=StratifiedKFold(n_splits= 5, shuffle=True, random_state=3),
                            scoring = ['f1_macro', 'accuracy'])
    accuracy = round(cv_results['test_accuracy'].mean(), 2)
    f1_score = round(cv_results['test_f1_macro'].mean(), 2)
    print(f'Model: {models_name[i]} - Accuracy: {accuracy} - F1 Score: {f1_score}')

