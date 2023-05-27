import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.metrics import confusion_matrix, f1_score
from sklearn import feature_selection
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from IPython.display import display
import plotly.express as px
import plotly.figure_factory as ff

def plot_cm(cm):
    z = cm
    x = ['defect', 'regular']
    y = ['defect', 'regular']

    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in z]

    # set up figure 
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')

    # add title
    fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',
                    #xaxis = dict(title='x'),
                    #yaxis = dict(title='x')
                    )

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=-0.35,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig['data'][0]['showscale'] = True
    fig.show()

def analyze_feature(df, feature_name):    
    fig = px.histogram(df, x = feature_name, color = 'isRegular', marginal="rug")
    fig.show()

def analyze_model(svm_model):
    # make a scaler + model pipeline
    pipeline = make_pipeline(StandardScaler(), svm_model)
    # 5 fold cross validation
    
    scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    print('5-fold cross validation scores: ')
    print(scores)
    pipeline.fit(X_train, y_train)
    return scores.mean()

def grid_analyze_model(param_grid):
     # make a scales + model pipeline
    pipeline = make_pipeline(StandardScaler(), svm.SVC())
    # 5 fold cross validation
    grid = GridSearchCV(pipeline, param_grid, refit = True, cv = 5)
    
    grid.fit(X_train, y_train)
    display(grid.cv_results_['mean_test_score'])
    eval.loc[len(eval.index)] = [grid.best_params_['svc__kernel'], grid.best_params_['svc__kernel'] + ' with cost {0} and gamma {1}'.format(grid.best_params_['svc__C'], grid.best_params_['svc__gamma']), grid.best_score_, grid.best_estimator_.named_steps["svc"].n_support_[0], grid.best_estimator_.named_steps["svc"].n_support_[1]]
    if(grid.best_params_['svc__kernel']== 'poly'):
        eval.loc[len(eval.index)-1,'Description']+=str(grid.best_params_['svc__degree'])
    return grid.best_score_

def plot_2d(df, feature_x, feature_y):
    df["isRegular"] = df["isRegular"].astype(str)
    fig = px.scatter(df, x=feature_x, y=feature_y, color="isRegular", hover_data=['Filename'], log_x=True, log_y=True, color_discrete_sequence=["red", "blue"])
    fig.show()


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
ig = pd.DataFrame({'Feature': X.columns, 'Information Gain': info_gains})
ig_sorted = ig.sort_values(by='Information Gain', ascending=False)

display(ig_sorted)

plot_2d(data, ig_sorted['Feature'].iloc[0], ig_sorted['Feature'].iloc[1])

corr = X_train.corrwith(y_train).abs()
corr_sorted = corr.sort_values(ascending=False)
plot_2d(data, corr_sorted.index[0], corr_sorted.index[1])

display(corr_sorted)

# Plot distributions of feature with highest information gain
analyze_feature(data,  ig_sorted['Feature'].iloc[0])

# Plot distributions of feature with highest correlation
analyze_feature(data, corr.index[0])

# Plot distributions of feature with one of the lowest information gain
analyze_feature(data, ig_sorted['Feature'].iloc[80])

# Plot distributions of feature with lowest correlation
analyze_feature(data, corr.index[80])

eval = pd.DataFrame(columns = ['Model', 'Description', 'Score', 'Number of support vectors class 0', 'Number of support vectors class 1'])

# Linear classifier
costs = [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 100, 1000, 10000, 100000]

for cost in costs:
    clf = svm.SVC(kernel = 'linear', C = cost)
    cross_val_mean = analyze_model(clf)
    eval.loc[len(eval.index)] = ['Linear', 'Linear with cost {0}'.format(cost), cross_val_mean, clf.n_support_[0], clf.n_support_[1]]
  
# defining parameter range
param_grid = {'svc__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1e6, 1e7], 
              'svc__gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
              'svc__kernel': ['rbf']} 

cross_val_mean = grid_analyze_model(param_grid)

# defining parameter range
param_grid = {'svc__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 
              'svc__gamma': [10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
              'svc__kernel': ['poly'],
              'svc__degree': [3],
              'svc__coef0':[0,1,10,100]} 

cross_val_mean = grid_analyze_model(param_grid)

display(eval)

final_model = make_pipeline(StandardScaler(), svm.SVC(kernel = 'linear', C = 1))
final_model.fit(X_train,y_train)
y_pred= final_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
plot_cm(cm)
