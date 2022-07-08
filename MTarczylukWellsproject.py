import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_curve, roc_auc_score


###LOADING DATA SETS###

cross_data = pd.read_csv("oasis_cross-sectional.csv")
long_data = pd.read_csv("oasis_longitudinal.csv")

print(cross_data.info())
print(long_data.info())

print(cross_data.head())
print(long_data.head())


###DATA PROCESSING###

# Longitudinal study: use data from 1st visit only

long_data = long_data.loc[long_data['Visit']==1]
long_data = long_data.reset_index(drop=True)

# Longitudinal study: replace Male / Female with 0 / 1

long_data['M/F'] = long_data['M/F'].replace(['F','M'], [0,1])

# Longitudinal study: replace CDR score with Demented / Nondemented

long_data['CDR'] = long_data['CDR'].replace([0, 0.5, 1, 2], [0, 1, 1, 1])

# Longitudinal study: drop columns I won't be using

long_data.drop(columns=['Subject ID', 'MRI ID', 'Group', 'Visit', 'MR Delay', 'Hand'], inplace=True)

# Longitudinal study: rename EDUC column to Educ to match cross study

long_data = long_data.rename(columns={'EDUC':'Educ'})

# Longitudinal study: rename CDR column to Diagnosis

long_data = long_data.rename(columns={'CDR':'Diagnosis'})

print(long_data.info())
print(long_data.head())

# Cross-sectional study: replace Male / Female with 0 / 1

cross_data['M/F'] = cross_data['M/F'].replace(['F','M'], [0,1])

# Cross-sectional study: drop columns I won't be using

cross_data.drop(columns=['ID', 'Delay', 'Hand'], inplace=True)

# Cross-sectional study: drop any NA values from CDR column since it's the one I'll be predicting

cross_data.dropna(subset = ["CDR"], axis = 0, inplace = True)

# Cross-sectional study: replace CDR score with Demented / Nondemented

cross_data['CDR'] = cross_data['CDR'].replace([0, 0.5, 1, 2], [0, 1, 1, 1])

# Cross-sectional study: rename CDR column to Diagnosis

cross_data = cross_data.rename(columns={'CDR':'Diagnosis'})

print(cross_data.info())
print(cross_data.head())

# Append both data sets

AD_data = pd.concat([cross_data, long_data])
print(AD_data.head())
print(AD_data.info())

###EXPLORATORY DATA ANALYSIS###

# Correlation

def AD_corr(data):
    corr = AD_data.corr()
    plt.figure(figsize=(12,6))
    sns.heatmap(corr, annot=True, vmin=-1)
    plt.show()

AD_corr(AD_data)

# Gender vs dementia

def plot_gender(data):
    demented = AD_data[AD_data['Diagnosis']==1]['M/F'].value_counts()
    demented = pd.DataFrame(demented)
    demented.index=['Male', 'Female']
    demented.plot(kind='bar', figsize=(8,6))
    plt.title('Gender vs Dementia', size=16)
    plt.xlabel('Gender', size=14)
    plt.ylabel('Patients with Dementia', size=14)
    plt.xticks(rotation=0)
    plt.show()

plot_gender(AD_data)

# Age vs Normalized Whole Brain Volume

def plot_age(data):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x='Age', y='nWBV', data=AD_data, hue='Diagnosis')
    plt.title('Age vs Normalized Whole Brain Volume', size=16)
    plt.xlabel('Age', size=14)
    plt.ylabel('Normalized Whole Brain Volume', size=14)
    plt.show()

plot_age(AD_data)

# Diagnosis vs Normalized Whole Brain Volume

def plot_diagnosis(data):
    plt.figure(figsize=(12,8))
    sns.catplot(x='Diagnosis',y='nWBV',data=AD_data, hue='M/F')
    plt.show()

plot_diagnosis(AD_data)

# Diagnosis vs MMSE

def plot_MMSE(data):
    plt.figure(figsize=(12,8))
    sns.catplot(x='Diagnosis',y='MMSE',data=AD_data, hue='M/F')
    plt.show()

plot_MMSE(AD_data)    

# ASF vs eTIV

def plot_ASF(data):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x='ASF', y='eTIV', data=AD_data, hue='Diagnosis')
    plt.title('ASF vs eTIV', size=16)
    plt.xlabel('ASF', size=14)
    plt.ylabel('eTIV', size=14)
    plt.show()

plot_ASF(AD_data)

# Since ASF and eTIV correlation is almost 1:1, drop ASF

AD_data.drop(columns=['ASF'], inplace=True)
print(AD_data.info())

###DATA MODELLING###

# Imputation

AD_data.isna().sum()

# Since SES is a class value of 1,2,3,4 or 5, we will use most frequent element

impute = SimpleImputer (missing_values = np.nan, strategy = 'most_frequent')
impute.fit(AD_data[['SES']])
AD_data[['SES']] = impute.fit_transform(AD_data[['SES']])

pd.isnull(AD_data['SES']).value_counts()

# Splitting data set

Y = AD_data['Diagnosis'].values
X = AD_data[['M/F', 'Age', 'Educ', 'SES', 'MMSE', 'eTIV', 'nWBV']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

X_train.hist(bins=30, figsize=(20,15))
plt.show()

# Scaling

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)


# Machine Learning algorithms

"""Program to analyse the AD_dataset and print accuracy and F1 scores"""

models = {'            Logistic Regression': LogisticRegression(),
          '   Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
          'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
          '                            KNN': KNeighborsClassifier(),
          '       Decision Tree Classifier': DecisionTreeClassifier(),
          '       Random Forest Classifier': RandomForestClassifier(),
          '   Gradient Boosting Classifier': GradientBoostingClassifier()}

def fit_models(models):
    """Fits machine learning algorith from models dictionary to training data"""
    for name, model in models.items():
        model.fit(X_train_scaled, Y_train)
        print(name + ' trained.')


# Model Results

def print_results(models):
    """Predicts response variable for test data and prints accuracy result for each machine learning algorith from models dictionary"""
    for name, model in models.items():
        y_prediction = model.predict(X_test_scaled)
        acc = accuracy_score(Y_test, y_prediction)
        print(name + ' Accuracy: {:.2%}'.format(acc)) 
    
# F1-score

def print_F1_score(models):
    """Predicts response variable for test data and prints F1 score for each machine learning algorith from models dictionary"""
    for name, model in models.items():
        y_prediction = model.predict(X_test_scaled)
        f1 = f1_score(Y_test, y_prediction, pos_label=1)
        print(name + ' F1-Score: {:.5}'.format(f1))


def main():
    """The main function"""
    fit_models(models)
    print_results(models)
    print_F1_score(models)
    
main()


###FURTHER EVALUATION OF THE BEST MODEL - LOGICAL REGRESSION###

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train_scaled, Y_train)

Y_pred = logreg.predict(X_test_scaled)


# Confusion Matrix

cnf_matrix = confusion_matrix(Y_test, Y_pred)
cnf_matrix

# visualize the Confusion Matrix

def plot_matrix(cnf_matrix):
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cnf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cnf_matrix.flatten()/np.sum(cnf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    plt.figure()
    sns.heatmap(cnf_matrix, annot=labels, fmt='', cmap='Blues')
    plt.show()

plot_matrix(cnf_matrix)

# Confusion Matrix evaluation

print(f"Accuracy: {accuracy_score(Y_test, Y_pred):.5}")
print(f"Precision: {precision_score(Y_test, Y_pred):.5}")
print(f"Recall: {recall_score(Y_test, Y_pred):.5}")
print(f"F1-Score: {f1_score(Y_test, Y_pred, pos_label=1):.5}")

# ROC Curve

def plot_ROC(X_test_scaled, Y_test):
    y_pred_proba = logreg.predict_proba(X_test_scaled)[::,1]
    fpr, tpr, _ = roc_curve(Y_test,  y_pred_proba)
    auc = roc_auc_score(Y_test, y_pred_proba)

    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()

plot_ROC(X_test_scaled, Y_test)