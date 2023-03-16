#!/usr/bin/env python
# coding: utf-8

# # Airline Passengers Satisfaction Prediction
# 
# ### By Ludmila Emília Mucavele

# ## Content
# 
# <ul>
# <li><a href="#1"> Dataset Introduction</a></li>
# <li><a href="#2">Data Gathering and Wrangling</a></li>
# <li><a href="#3">Pre-Processing</a></li>
# <li><a href="#4">Model Evaluation</a></li>
# <li><a href="#5">Satisfaction Prediction</a></li>
# </ul>

# <a id = 1></a>
# ## Dataset Introduction
# The dataset contains Customer satisfaction scores from 120,000+ airline passengers, including additional information about each passenger, their flight, and type of travel, as well as their evaluation of different factors like cleanliness, comfort, service, and overall experience.
# 
# The dataset can be found <a href="https://www.kaggle.com/datasets/mysarahmadbhat/airline-passenger-satisfaction">here</a>.
# 
# The aim of the project is to develop a predictive system for passenger satisfaction based on passengers' flights information.
# 
# More details are shown in the dictionary below.
# 
# * __ID:__	Unique passenger identifier
# * __Gender:__	Gender of the passenger (Female/Male)
# * __Age:__	Age of the passenger
# * __Customer Type:__	Type of airline customer (First-time/Returning)
# * __Type of Travel:__	Purpose of the flight (Business/Personal)
# * __Class Travel:__ class in the airplane for the passenger seat
# * __Flight Distance:__	Flight distance in miles
# * __Departure Delay:__	Flight departure delay in minutes
# * __Arrival Delay:__	Flight arrival delay in minutes
# * __Departure and Arrival Time Convenience:__	Satisfaction level with the convenience of the flight departure and arrival times from 1 (lowest) to 5 (highest) - 0 means "not applicable"
# * __Ease of Online Booking:__	Satisfaction level with the online booking experience from 1 (lowest) to 5 (highest) - 0 means "not applicable"
# * __Check-in Service:__	Satisfaction level with the check-in service from 1 (lowest) to 5 (highest) - 0 means "not applicable"
# * __Online Boarding:__	Satisfaction level with the online boarding experience from 1 (lowest) to 5 (highest) - 0 means "not applicable"
# * __Gate Location:__	Satisfaction level with the gate location in the airport from 1 (lowest) to 5 (highest) - 0 means "not applicable"
# * __On-board Service:__	Satisfaction level with the on-boarding service in the airport from 1 (lowest) to 5 (highest) - 0 means "not applicable"
# * __Seat Comfort:__	Satisfaction level with the comfort of the airplane seat from 1 (lowest) to 5 (highest) - 0 means "not applicable"
# * __Leg Room Service:__	Satisfaction level with the leg room of the airplane seat from 1 (lowest) to 5 (highest) - 0 means "not applicable"
# * __Cleanliness:__	Satisfaction level with the cleanliness of the airplane from 1 (lowest) to 5 (highest) - 0 means "not applicable"
# * __Food and Drink:__	Satisfaction level with the food and drinks on the airplane from 1 (lowest) to 5 (highest) - 0 means "not applicable"
# * __In-flight Service:__	Satisfaction level with the in-flight service from 1 (lowest) to 5 (highest) - 0 means "not applicable"
# * __In-flight Wifi Service:__	Satisfaction level with the in-flight Wifi service from 1 (lowest) to 5 (highest) - 0 means "not applicable"
# * __In-flight Entertainment:__	Satisfaction level with the in-flight entertainment from 1 (lowest) to 5 (highest) - 0 means "not applicable"
# * __Baggage Handling:__	Satisfaction level with the baggage handling from the airline from 1 (lowest) to 5 (highest) - 0 means "not applicable"
# * __Satisfaction:__	Overall satisfaction level with the airline (Satisfied/Neutral or unsatisfied)
# 
# 
# 
# #### Packages

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(0)


# <a id = 2></a>
# ## Data Gathering and Cleaning
# Let's start with collecting and cleaning the data for training and testing in the model.

# In[2]:


df = pd.read_csv('airline_passenger_satisfaction.csv')
df.head()


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


# Check for null values in Arrival Delay column
df[df['Arrival Delay'].isna()].head()


# In[6]:


# Fix null values in Arrival Delay column
df.loc[df['Arrival Delay'].isna(), 'Arrival Delay'] = 0
df['Arrival Delay'].astype(int)
assert df['Arrival Delay'].isna().sum() == 0


# In[7]:


df.duplicated().sum()


# In[8]:


df.nunique()


# <a id = 3></a>
# ## Pre-Processing

# In[9]:


# Turn Satisfaction column to numeric categorical feature
dict = ({'Neutral or Dissatisfied' : 0, 'Satisfied' : 1})
df.Satisfaction = df.Satisfaction.map(dict)
df.head()


# In[10]:


# Transforming non numeric categorical features
df = pd.get_dummies(data = df, columns = ['Gender', 'Customer Type', 'Type of Travel', 
                                         'Class'], drop_first=True)
df.head()


# Let's have a look at the correlation of the data features and the target.

# In[11]:


plt.figure(figsize = (25, 20))

# Generate a mask for the upper triangle
mask = np.zeros_like(df.corr())
mask[np.triu_indices_from(mask)] = True
# plot the Heat map
sns.heatmap(df.corr(),  annot=True, mask = mask, square=True, linewidths=.5);


# With the plot above, it was possible to examine the correlations between the features. There are features that are negatively correlated with all the features, and that will e of no help to the model:
# * ID
# * Age
# * Flight DIstance
# * Arrival Delay
# 
# **Note:** Arrival Delay feature feauture will be dropped to avoiding multicorrelation problem, once there is a significant correlation between Departure Delay and Arrival Delay.

# In[12]:


# Drop unnecessary features
df.drop(['ID', 'Age', 'Flight Distance'], axis = 1, inplace = True)

# Drop Arrival Delay feature for avoiding multicorrelation problem
df.drop(['Arrival Delay'], axis = 1, inplace = True)

df


# In[13]:


columns = list(df.columns)
for i in columns:
    df[i] = pd.to_numeric(df[i], errors = 'coerce')
df


# In[14]:


# creating predictor and target dataset
x = df.drop('Satisfaction', axis = 1)
y = pd.DataFrame(df['Satisfaction'])
print('Predictor features: '  + str(list(x.columns)))
print('Target feature: ' + str(list(y.columns)))


# Let's split it into train and test data for the model.

# In[15]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 0)
y_train, y_test = np.ravel(y_train), np.ravel(y_test)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#y_train, y_test = y_train.values.flatten(), y_test.values.flatten()


# Data preprocessed and splitted, let's move to the model selection and comparison.

# <a id = 4></a>
# ## Model Evaluation

# In[16]:


from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

logreg = LogisticRegression(solver = 'lbfgs', max_iter = 300, penalty = 'l2', C = 0.1)
knn = KNeighborsClassifier(n_neighbors = 1, weights = 'uniform')
dectree = DecisionTreeClassifier()
randomf = RandomForestClassifier()

classifiers = [logreg, knn, dectree, randomf]
classifiers_name = ['Logistic Regression', 'K - Nearest Neighbors', 'Decision Trees', 'Random Forest']


# In[17]:


# Function for model training and testing
def run_model(model, name):
    model.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    print(f'Model: {name}')
    # Percentage of correct predictions in relation with the total predictions
    print(f'Trainning Score: {model.score(x_train, y_train)}')
    print(f'Test Score: {model.score(x_test, y_test)}')
    # Verify accuracy of predictions for train and test data:
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    print(f'Acc Train: {(acc_train)}')
    print(f'Acc Test: {(acc_test)}') 
    print ("--------------------------------------")


# In[18]:


from scipy import stats

# Using Logistic Regression CLassifier
run_model(logreg, 'Logistic Regression')

# Using Logistic Decision Trees CLassifier
run_model(dectree, 'Decision Trees')

# Using Random Forest CLassifier
run_model(randomf, 'Random Forest')

# Using K - Nearest Neighbors CLassifier
run_model(knn, 'K - Nearest Neighbors')


# * Using plot roc curves for model evaluations:

# In[19]:


from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
line = np.linspace(0,1)


sns.set(font_scale = 1.0)
for classifier, ax, name in zip(classifiers, axes.flatten(), classifiers_name):
    # Compute the false positive rate and true positive rate
    y_pred = classifier.predict(x_test)
    auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    # Plot the ROC curve using RocCurveDisplay
    ax.plot(fpr, tpr)
    ax.plot(line, line, color = 'blue', linestyle = 'dashed')
    ax.title.set_text(name)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend([f'{name} (AUC = {auc:.2f})'], loc = 'lower right')
fig.tight_layout(pad = 1.0);


# It is clear that the best suited model to this occasion is the Random Forest with 0.96 of proportion of correct predictions over the total number of predictions.
# 
# Finnaly, let us try predicting one passenger overall satisfaction based in their rating.
# 
# <a id = 5></a>
# ### Satisfaction Prediction

# In[20]:


def satisfaction(rates):
    satisfaction = randomf.predict(rates)
    if (satisfaction == 0):
        print("Not Satisfied (0)")
    else:
        print("Satisfied (1)")


# In[21]:


rates = {'Departure Delay' : 20,
        'Departure and Arrival Time Convenience' : 0,
        'Ease of Online Booking' : 3,
        'Check-in Service' : 5,
        'Online Boarding' : 5,
        'Gate Location' : 2,
        'On-board Service' : 4, 
        'Seat Comfort' : 3,
        'Leg Room Service' : 3,
        'Cleanliness' : 2,
        'Food and Drink ': 3, 
        'In-flight Service' : 3, 
        'In-flight Wifi Service' : 3, 
        'In-flight Entertainment' : 2, 
        'Baggage Handling' : 3,
        'Gender_Male' : 0,
        'Customer Type_Returning' : 0, 
        'Type of Travel_Personal' : 1,
        'Class_Economy' : 1,
        'Class_Economy Plus' : 0 }
rates = np.array(list(rates.values())).reshape(1, -1)
satisfaction(rates)


# In[ ]:




