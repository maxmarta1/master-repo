import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn import feature_selection
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from IPython.display import display

def analyze_feature(X, y, feature_name):
    # Separate data by class label
    feature_0 = X[feature_name][y == -1]
    feature_1 =  X[feature_name][y == 1]

    # Plot the distributions
    fig, axs = plt.subplots(2,1, sharex = True)
    axs[0].hist(feature_1, color='red', alpha=0.5, label='Class 1')
    axs[0].legend(loc='upper right')
    axs[0].set_title('Distribution of Feature {0} by Class Label'.format(feature_name))
    axs[0].set_xlabel('Feature Value')
    axs[0].set_ylabel('Frequency')

    axs[1].hist(feature_0, color='blue', alpha=0.5, label='Class 0')
    axs[1].legend(loc='upper right')
    axs[1].set_xlabel('Feature Value')
    axs[1].set_ylabel('Frequency')
    
    plt.waitforbuttonpress()
    plt.show()

def analyze_model(svm_model):
    # make a scales + model pipeline
    pipeline = make_pipeline(StandardScaler(), svm_model)
    # 5 fold cross validation
    scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    print('5-fold cross validation scores: ')
    print(scores)
    return scores.mean()
    # pipeline.fit(X_train, y_train)
    # y_pred = pipeline.predict(X_test)

    # print(f1_score(y_test, y_pred))


# Import first part of features 
filepath_1 = os.getcwd() + '\TextureAnalysisMetrics_Pooled.csv'
data_1 = pd.read_csv(filepath_1, header=0)
print(data_1.head())
print('Number of features: {0}'.format(data_1.shape[1]))

# Import second part of features
filepath_2 = os.getcwd() + r'\features.csv'
data_2 = pd.read_csv(filepath_2, header=0)
print(data_2.head())
print('Number of features: {0}'.format(data_2.shape[1]))

data_2['Filename'] = data_2['Filename'].str.replace('.bmp','.dcm')

# Merge data on filename
data = pd.merge(data_1, data_2, on='Filename')

print('Feature names: ' + data.columns)

# Delete outliers (empty images)
data = data[data['Filename']!="dnm (106).dcm"]
data = data[data['Filename']!="dnm (108).dcm"]
print('Size of the dataset: {0}'.format(data.shape[0]))

# Split into X and y and drop non numerical columns
X = data.drop(['Subject_isHealthy','Subject_id', 'Filename', 'isRegular'], axis= 1)
y = data['isRegular']*2-1

# Check null values
null_count = data.isnull().sum()
print('Number of features with at least one null value: {0}'.format(null_count[null_count>0]))

# Check balance of the dataset
balance = y.value_counts()
print(balance.keys)
print('Defect: {0}, Regular: {1}'.format(balance[-1], balance[1]))

# Save a part of the dataset for final evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# Feature analysis
info_gains = feature_selection.mutual_info_classif(X_train, y_train)
df = pd.DataFrame({'Feature': X.columns, 'Information Gain': info_gains})
df_sorted = df.sort_values(by='Information Gain', ascending=False)

print('Features sorted by information gain:')
print(df_sorted)

# Plot distributions of feature with highest information gain
analyze_feature(X_train, y_train, 'SecondOrderStats_cprom')

# Plot distributions of feature with one of the lowest information gain
analyze_feature(X_train, y_train, 'SecondOrderStats_denth')
eval = pd.DataFrame(columns = ['Model', 'Description', 'Score'])

# Linear classifier
clf_linear = svm.LinearSVC()
cross_val_mean = analyze_model(clf_linear)
eval.loc[len(eval.index)] = ['Linear', 'Linear, l2 penalty, squared hinge', cross_val_mean]

# Linear classifier
clf_linear = svm.LinearSVC(penalty = 'l1', dual = False)
cross_val_mean = analyze_model(clf_linear)
eval.loc[len(eval.index)] = ['Linear', 'Linear, l1 penalty, squared hinge', cross_val_mean]

clf_linear = svm.LinearSVC(loss='hinge')
cross_val_mean = analyze_model(clf_linear)
eval.loc[len(eval.index)] = ['Linear', 'Linear, l2 penalty, hinge', cross_val_mean]

costs = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]

for cost in costs:
    clf = svm.LinearSVC(C = cost)
    cross_val_mean = analyze_model(clf)
    eval.loc[len(eval.index)] = ['Linear', 'Linear with cost {0}'.format(cost), cross_val_mean]

costs = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]

for cost in costs:
    clf_rbf = svm.SVC(kernel = 'rbf', C = cost)
    cross_val_mean = analyze_model(clf_rbf)
    eval.loc[len(eval.index)] = ['RBF', 'RBF with cost {0}'.format(cost), cross_val_mean]

costs = [0.1, 1, 10, 100, 1000]

for cost in costs:
    clf_p = svm.SVC(kernel = 'poly', C = cost)
    cross_val_mean = analyze_model(clf_p)
    eval.loc[len(eval.index)] = ['Poly', 'Poly with cost {0}'.format(cost), cross_val_mean]

display(eval)