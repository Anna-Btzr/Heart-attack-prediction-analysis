#!/usr/bin/env python
# coding: utf-8

# Heart attack analysis and prediction

# 
# 
# 
# 
# 

# Attributes
# 
#     1. age: age of the patient in years 
#     
#     2. sex: sex of the patient 
#         •	0=female
#         •	1=male
#         
#     3. cp: chest pain type 
#         •	Value 0: asymptomatic (ASY)
#         •	Value 1: typical agina (TA)
#         •	Value 2: atypical angina (ATA)
#         •	Value 3: non-anginal pain (NAP)
# 
#     4. trtbps: Resting blood pressure (in mm Hg (millimeter of mercury) on admission to the hospital
#     
#     5. chol: Serum cholesterol in milligrams per deciliter (mg/dl) of blood, fetched by BMI sensor
#     
#     6. fbs: fasting blood sugar > 120 mg/dl 
#         •	0=false
#         •	1=true
# 
#     7. restecg: resting electrocardiographic results
#         •	Value 0: normal
#         •	Value 1:having ST-T wave abnormality (Twave intersions and/or ST elevation or depression of > 0.05 mV)
#         •	Value 2: showing probable or definite left ventricular hypertrophy by Estes’ criteria
# 
#     8. exng: exercise induced angina 
#         •	0=no
#         •	1=yes
# 
#     9. oldpeak: ST depression induced by exercise relative to rest 
# 
#     10. thalachh: maximum heart rate achieved 
# 
#     11. slp: the slope of the peak exercise ST segment
#         •	0=unsloping
#         •	1=flat
#         •	2=downsloping
# 
#     12. caa: number of major vessels colored by fluoroscopy (0-3) 
# 
#     13. thall: Thallium stress test
#         •	0=null
#         •	1=fixed defect
#         •	2=normal
#         •	3=reversable
# 
#     14. output: target value 
#         •	0=less chance of heart attack (<50% diameter narrowing)
#         •	1=more chance of a heart attack (>50% diameter narrowing)
# 

# Loading The Statistics Dataset / Required Python Libraries

# In[135]:


import numpy as np #linear algebra
import pandas as pd #data processing, CSV file

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("heart.csv") #loading the dataset
df.head()


# Initial analysis on the dataset

# In[136]:


print ("Shape of Dataset:", df.shape)


# In[137]:


df.info()


# Examining Missing values

# In[138]:


df.isnull()


# In[139]:


df.isnull().sum()


# In[140]:


isnull_number = []
for i in df.columns:
    x = df[i].isnull().sum()
    isnull_number.append(x)

pd.DataFrame(isnull_number, index = df.columns, columns = ["Total Missing Values"])


# In[141]:


get_ipython().system('pip install missingno')
import missingno
missingno.bar(df, color = "b")


# Examining Unique Values

# In[142]:


df.head()


# In[143]:


df["cp"].value_counts()


# In[144]:


df["cp"].value_counts().count()


# In[145]:


df["cp"].value_counts().sum()


# In[146]:


unique_number = []
for i in df.columns:
    x = df[i].value_counts().count()
    unique_number.append(x)

pd.DataFrame(unique_number, index = df.columns, columns = ["Total Unique Values"])


# Separating attributes (numeric or categorical)

# In[147]:


df.head()


# In[148]:


numeric_var = ["age", "trtbps", "chol", "thalachh", "oldpeak"]
categorical_var = ["sex", "cp", "fbs", "restecg", "exng", "slp", "caa", "thall", "output"]


# Examining statistics of variables

# In[149]:


df[numeric_var].describe()


# In[150]:


sns.distplot(df["age"])


# In[151]:


sns.distplot(df["age"], hist_kws = dict(linewidth = 1, edgecolor = "k"));


# In[152]:


sns.distplot(df["trtbps"], hist_kws = dict(linewidth = 1, edgecolor = "k"), bins = 20);


# In[153]:


sns.distplot(df["chol"], hist = False);


# In[154]:


x, y = plt.subplots(figsize = (8, 6))
sns.distplot(df["thalachh"], hist = False, ax = y)
y.axvline(df["thalachh"].mean(), color = "r", ls = "--");


# In[155]:


x, y = plt.subplots(figsize = (8, 6))
sns.distplot(df["oldpeak"], hist_kws = dict(linewidth = 1, edgecolor = "k"), bins = 20, ax = y)
y.axvline(df["oldpeak"].mean(), color = "r", ls = "--");


# Exploratory Data Analysis (EDA)- Uni-variate Analysis

# Numeric variables analysis with distplot

# In[156]:


numeric_var 


# In[250]:


numeric_axis_name = ["Age of the patient", "Resting blood pressure (in mm Hg)", "Serum Cholesterol in mg/dl", "Maximum heart rate achieved", "ST depression"]


# In[251]:


list(zip(numeric_var, numeric_axis_name))


# In[252]:


title_font = {"family" : "arial", "color" : "darkred", "weight" : "bold", "size" : 15}
axis_font = {"family" : "arial", "color" : "darkblue", "weight" : "bold", "size" : 13}

for i, z in list(zip(numeric_var, numeric_axis_name)):
    plt.figure(figsize = (8, 6), dpi = 80)
    sns.distplot(df[i], hist_kws = dict(linewidth = 1, edgecolor = "k"), bins = 20)
    
    plt.title(i, fontdict = title_font)
    plt.xlabel(z, fontdict = axis_font)
    plt.ylabel("Density", fontdict = axis_font)
    
    plt.tight_layout()
    plt.show()


# Categoric variables analysis with pie chart

# In[160]:


categorical_var


# In[198]:


categorical_axis_name = ["Sex assigned at birth", "Chest pain type", "Fasting blood sugar", "Resting electrocardiographic results", "Exercise induced angina", "The slope of ST segment", "Number of major vesels", "Thallium stress test", "Target value"]


# In[199]:


list(zip(categorical_var, categorical_axis_name))


# In[200]:


df["cp"].value_counts()


# In[201]:


list(df["cp"].value_counts())


# In[202]:


list(df["cp"].value_counts().index)


# In[203]:


title_font = {"family" : "arial", "color" : "darkred", "weight" : "bold", "size" : 15}
axis_font = {"family" : "arial", "color" : "darkblue", "weight" : "bold", "size" : 13}

for i, z in list(zip(categorical_var, categorical_axis_name)):
    fig, ax = plt.subplots(figsize = (8, 6))
    
    observation_values = list(df[i].value_counts().index)
    total_observation_values = list(df[i].value_counts())
    
    ax.pie(total_observation_values, labels= observation_values, autopct = '%1.1f%%', startangle = 110, labeldistance = 1.1)
    ax.axis("equal") # Equal aspect ratio ensures that pie is drawn as a circle.
    
    plt.title((i + "(" + z + ")"), fontdict = title_font) # Naming Pie Chart Titles
    plt.legend()
    plt.show()


# Examining the Missing Data According to the Analysis Result
# 

# In[204]:


df[df["thall"] == 0]


# In[205]:


df["thall"] = df["thall"].replace(0, np.nan)


# In[206]:


df.loc[[48, 281], :]


# In[207]:


isnull_number = []
for i in df.columns:
    x = df[i].isnull().sum()
    isnull_number.append(x)
    
pd.DataFrame(isnull_number, index = df.columns, columns = ["Total Missing Values"])


# In[208]:


df["thall"].fillna(2, inplace = True)


# In[209]:


df.loc[[48, 281], :]


# In[210]:


df


# In[211]:


df["thall"] = pd.to_numeric(df["thall"], downcast = "integer")


# In[212]:


df.loc[[48, 281], :]


# In[213]:


isnull_number = []
for i in df.columns:
    x = df[i].isnull().sum()
    isnull_number.append(x)
    
pd.DataFrame(isnull_number, index = df.columns, columns = ["Total Missing Values"])


# In[177]:


df["thall"].value_counts()


# Bi-variate Analysis

# Numerical Variables - Target Variable(Analysis with FaceGrid)

# In[253]:


numeric_var


# In[254]:


numeric_var.append("output")


# In[255]:


numeric_var


# In[256]:


title_font = {"family" : "arial", "color" : "darkred", "weight" : "bold", "size" : 15}
axis_font = {"family" : "arial", "color" : "darkblue", "weight" : "bold", "size" : 13}

for i, z in list(zip(numeric_var, numeric_axis_name)):
    graph = sns.FacetGrid(df[numeric_var], hue = "output", height = 5, xlim = ((df[i].min() - 10), (df[i].max() + 10)))
    graph.map(sns.kdeplot, i, shade = True)
    graph.add_legend()
    
    plt.title(i, fontdict = title_font)
    plt.xlabel(z, fontdict = axis_font)
    plt.ylabel("Density", fontdict = axis_font)
    
    plt.tight_layout()
    plt.show()


# In[257]:


df[numeric_var].corr()


# In[258]:


df[numeric_var].corr().iloc[:, [-1]]


# Categorical Variables - Target Variable(Analysis with Count Plot)

# In[259]:


title_font = {"family" : "arial", "color" : "darkred", "weight" : "bold", "size" : 15}
axis_font = {"family" : "arial", "color" : "darkblue", "weight" : "bold", "size" : 13}

for i, z in list(zip(categorical_var, categorical_axis_name)):
    plt.figure(figsize = (8, 5))
    sns.countplot(i, data = df[categorical_var], hue = "output")
    
    plt.title(i + " - output", fontdict = title_font)
    plt.xlabel(z, fontdict = axis_font)
    plt.ylabel("output", fontdict = axis_font)
    
    plt.tight_layout()
    plt.show()


# In[266]:


df[categorical_var].corr()


# In[267]:


df[categorical_var].corr().iloc[:, [-1]]


# Examining Numeric Variables Among Themselves (Analysis with pair plot)

# In[262]:


numeric_var


# In[263]:


numeric_var.remove("output")


# In[264]:


df[numeric_var].head()


# In[265]:


graph = sns.pairplot(df[numeric_var], diag_kind = "kde")
graph.map_lower(sns.kdeplot, levels = 4, color = ".2")
plt.show()


#  Feature Scaling with the RobustScaler Method

# In[229]:


from sklearn.preprocessing import RobustScaler


# In[230]:


robust_scaler = RobustScaler()


# In[231]:


scaled_data = robust_scaler.fit_transform(df[numeric_var])


# In[232]:


scaled_data


# In[233]:


type(scaled_data)


# In[234]:


df_scaled = pd.DataFrame(scaled_data, columns = numeric_var)
df_scaled.head()


# Creating a New DataFrame with the Melt() Function

# In[236]:


df_new = pd.concat([df_scaled, df.loc[:, "output"]], axis = 1)


# In[237]:


df_new.head()


# In[238]:


melted_data = pd.melt(df_new, id_vars = "output", var_name = "variables", value_name = "value")


# In[239]:


melted_data


# In[241]:


plt.figure(figsize = (8, 5))
sns.swarmplot(x = "variables", y = "value", hue = "output", data = melted_data)
plt.show()


# Numerical Variables - Categorical Variables (Analysis with Swarm Plot)

# In[242]:


axis_font = {"family" : "arial", "color" : "black", "weight" : "bold", "size" : 14}
for i in df[categorical_var]:
    df_new = pd.concat([df_scaled, df.loc[:, i]], axis = 1)
    melted_data = pd.melt(df_new, id_vars = i, var_name = "variables", value_name = "value")
    
    plt.figure(figsize = (8, 5))
    sns.swarmplot(x = "variables", y = "value", hue = i, data = melted_data)
    
    plt.xlabel("variables", fontdict = axis_font)
    plt.ylabel("value", fontdict = axis_font)
    
    plt.tight_layout()
    plt.show()


# Numerical Variables - Categorical Variables (Analysis with Box Plot)

# In[243]:


axis_font = {"family" : "arial", "color" : "black", "weight" : "bold", "size" : 14}
for i in df[categorical_var]:
    df_new = pd.concat([df_scaled, df.loc[:, i]], axis = 1)
    melted_data = pd.melt(df_new, id_vars = i, var_name = "variables", value_name = "value")
    
    plt.figure(figsize = (8, 5))
    sns.boxplot(x = "variables", y = "value", hue = i, data = melted_data)
    
    plt.xlabel("variables", fontdict = axis_font)
    plt.ylabel("value", fontdict = axis_font)
    
    plt.tight_layout()
    plt.show()


# Relationships between variables(Analysis with Heatmap)

# In[244]:


df_scaled


# In[246]:


df_new2 = pd.concat([df_scaled, df[categorical_var]], axis = 1)


# In[247]:


df_new2


# In[248]:


df_new2.corr()


# In[249]:


plt.figure(figsize = (15, 10))
sns.heatmap(data = df_new2.corr(), cmap = "Spectral", annot = True, linewidths = 0.5)


# Preparation for Modeling

# Dropping Columns with Low Correlation

# In[268]:


df.head()


# In[270]:


df.drop(["chol", "fbs", "restecg"], axis = 1, inplace = True)


# In[271]:


df.head()


# Struggling Outliers

# Visualizing outliers

# In[272]:


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize = (20, 6))

ax1.boxplot(df["age"])
ax1.set_title("age")

ax2.boxplot(df["trtbps"])
ax2.set_title("trtbps")

ax3.boxplot(df["thalachh"])
ax3.set_title("thalachh")

ax4.boxplot(df["oldpeak"])
ax4.set_title("oldpeak")

plt.show()


# Dealing with outliers

# Trtbps Variable

# In[273]:


from scipy import stats
from scipy.stats import zscore
from scipy.stats.mstats import winsorize 


# In[274]:


z_scores_trtbps = zscore(df["trtbps"])
for threshold in range(1, 4):
    print("Threshold Value: {}".format(threshold))
    print("Number of Outliers: {}".format(len(np.where(z_scores_trtbps > threshold)[0])))
    print("-------------------")


# In[275]:


df[z_scores_trtbps > 2][["trtbps"]]


# In[276]:


df[z_scores_trtbps > 2].trtbps.min()


# In[277]:


df[df["trtbps"] < 170].trtbps.max()


# In[278]:


winsorize_percentile_trtbps = (stats.percentileofscore(df["trtbps"], 165)) / 100
print(winsorize_percentile_trtbps)


# In[279]:


1 - winsorize_percentile_trtbps


# In[280]:


trtbps_winsorize = winsorize(df.trtbps, (0, (1 - winsorize_percentile_trtbps)))


# In[281]:


plt.boxplot(trtbps_winsorize)
plt.xlabel("trtbps_winsorize", color = "b")
plt.show()


# In[282]:


df["trtbps_winsorize"] = trtbps_winsorize


# In[283]:


df.head()


# Thalachh Variable

# In[284]:


def iqr(df, var):
    q1 = np.quantile(df[var], 0.25)
    q3 = np.quantile(df[var], 0.75)
    diff = q3 - q1
    lower_v = q1 - (1.5 * diff)
    upper_v = q3 + (1.5 * diff)
    return df[(df[var] < lower_v) | (df[var] > upper_v)]


# In[286]:


thalachh_out = iqr(df, "thalachh")


# In[287]:


thalachh_out


# In[288]:


df.drop([272], axis = 0, inplace = True)


# In[290]:


df["thalachh"][270:275]


# In[292]:


plt.boxplot(df["thalachh"]);


# Oldpeak Variable

# In[293]:


def iqr(df, var):
    q1 = np.quantile(df[var], 0.25)
    q3 = np.quantile(df[var], 0.75)
    diff = q3 - q1
    lower_v = q1 - (1.5 * diff)
    upper_v = q3 + (1.5 * diff)
    return df[(df[var] < lower_v) | (df[var] > upper_v)]


# In[294]:


iqr(df, "oldpeak")


# In[295]:


df[df["oldpeak"] < 4.2].oldpeak.max()


# In[296]:


winsorize_percentile_oldpeak = (stats.percentileofscore(df["oldpeak"], 4)) / 100
print(winsorize_percentile_oldpeak)


# In[297]:


oldpeak_winsorize = winsorize(df.oldpeak, (0, (1 - winsorize_percentile_oldpeak)))


# In[298]:


plt.boxplot(oldpeak_winsorize)
plt.xlabel("oldpeak_winsorize", color = "b")
plt.show()


# In[299]:


df["oldpeak_winsorize"] = oldpeak_winsorize


# In[300]:


df.head()


# In[301]:


df.drop(["trtbps", "oldpeak"], axis = 1, inplace = True)


# In[302]:


df.head()


# Determining Distributions of Numeric Variables
# 

# In[304]:


df.head()


# In[306]:


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize = (20, 6))

ax1.hist(df["age"])
ax1.set_title("age")

ax2.hist(df["trtbps_winsorize"])
ax2.set_title("trtbps_winsorize")

ax3.hist(df["thalachh"])
ax3.set_title("thalachh")

ax4.hist(df["oldpeak_winsorize"])
ax4.set_title("oldpeak_winsorize")

plt.show()


# In[307]:


df[["age", "trtbps_winsorize", "thalachh", "oldpeak_winsorize"]].agg(["skew"]).transpose()


# Transformation Operations on Unsymmetrical Data

# In[308]:


df["oldpeak_winsorize_log"] = np.log(df["oldpeak_winsorize"])
df["oldpeak_winsorize_sqrt"] = np.sqrt(df["oldpeak_winsorize"])


# In[309]:


df.head()


# In[310]:


df[["oldpeak_winsorize", "oldpeak_winsorize_log", "oldpeak_winsorize_sqrt"]].agg(["skew"]).transpose()


# In[311]:


df.drop(["oldpeak_winsorize", "oldpeak_winsorize_log"], axis = 1, inplace = True)


# In[312]:


df.head()


# Applying One Hot Encoding Method to Categorical Variables
# 

# In[313]:


df_copy = df.copy()


# In[314]:


df_copy.head()


# In[320]:


categorical_var


# In[322]:



categorical_var.remove("restecg")


# In[324]:


categorical_var


# In[325]:


df_copy = pd.get_dummies(df_copy, columns = categorical_var[:-1], drop_first = True)


# In[326]:


df_copy.head()


# Feature Scaling with the RobustScaler Method for Machine Learning Algorithms

# In[327]:


new_numeric_var = ["age", "thalachh", "trtbps_winsorize", "oldpeak_winsorize_sqrt"]


# In[328]:


robus_scaler = RobustScaler()


# In[329]:


df_copy[new_numeric_var] = robust_scaler.fit_transform(df_copy[new_numeric_var])


# In[330]:


df_copy.head()


# Separating Data into Test and Training Set

# In[331]:


from sklearn.model_selection import train_test_split


# In[332]:


X = df_copy.drop(["output"], axis = 1)
y = df_copy[["output"]]


# In[334]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 3)


# In[335]:


X_train.head()


# In[336]:


y_train.head()


# In[337]:


print(f"X_train: {X_train.shape[0]}")
print(f"X_test: {X_test.shape[0]}")
print(f"y_train: {y_train.shape[0]}")
print(f"y_test: {y_test.shape[0]}")


# Modelling

# Logistic Regression Algorithm

# In[338]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[339]:


log_reg = LogisticRegression()
log_reg


# In[340]:


log_reg.fit(X_train, y_train)


# In[341]:


y_pred = log_reg.predict(X_test)


# In[342]:


y_pred


# In[343]:


accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy: {}".format(accuracy))


# Cross Validation

# In[344]:


from sklearn.model_selection import cross_val_score


# In[345]:


scores = cross_val_score(log_reg, X_test, y_test, cv = 10)
print("Cross-Validation Accuracy Scores", scores.mean())


# Decision Tree Algorithm

# In[346]:


from sklearn.tree import DecisionTreeClassifier


# In[347]:


dec_tree = DecisionTreeClassifier(random_state = 5)


# In[348]:


dec_tree.fit(X_train, y_train)


# In[349]:


y_pred = dec_tree.predict(X_test)


# In[350]:


print("The test accuracy score of Decision Tree is:", accuracy_score(y_test, y_pred))


# In[351]:


scores = cross_val_score(dec_tree, X_test, y_test, cv = 10)
print("Cross-Validation Accuracy Scores", scores.mean())


# In[ ]:





# Support Vector Machine Algorithm

# In[355]:


from sklearn.svm import SVC


# In[356]:


svc_model = SVC(random_state = 5)


# In[357]:


svc_model.fit(X_train, y_train)


# In[358]:


y_pred = svc_model.predict(X_test)


# In[359]:


print("The test accuracy score of SVM is:", accuracy_score(y_test, y_pred))


# In[360]:


scores = cross_val_score(svc_model, X_test, y_test, cv = 10)
print("Cross-Validation Accuracy Scores", scores.mean())


# In[ ]:





# Random Forest Algorithm

# In[362]:


from sklearn.ensemble import RandomForestClassifier


# In[363]:


random_forest = RandomForestClassifier(random_state = 5)


# In[364]:


random_forest.fit(X_train, y_train)


# In[365]:


y_pred = random_forest.predict(X_test)


# In[366]:


print("The test accuracy score of Random Forest is", accuracy_score(y_test, y_pred))


# In[367]:


scores = cross_val_score(random_forest, X_test, y_test, cv = 10)
print("Cross-Validation Accuracy Scores", scores.mean())


# In[ ]:





# In[ ]:




