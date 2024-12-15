import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from imblearn.under_sampling import RandomUnderSampler #undersampler
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import mlflow
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("df.csv")


X = df.drop(['dangerous'],axis = 1)
y = df['dangerous']


#function StandardScaler before and after
def aftstd(data):
    std_scaler = StandardScaler()
    df_std = std_scaler.fit(data)
    df_std = df_std.transform(data)
    return df_std





to_standardize = X
#Before standardization
# standardization
X = pd.DataFrame(aftstd(to_standardize))


#Using feature selector (mutual info classifier) to find relevant data for the target
importances = mutual_info_classif(X,y)
feat_importances = pd.Series(importances, X.columns[0:len(X.columns)])


#creating new df importances
column_names = X.columns.values.tolist()
importances_values = importances.tolist()
df_importances = pd.DataFrame([importances_values], columns = column_names)


#Finding values that hits set criteria
criteria = 0.05
relevant = []
for i in df_importances:
    value = df_importances[i]
    if df_importances[i].values > criteria:
        relevant.append(i) 
        


#Updating X values
X_sorted = X[relevant].values
print(f'\nNew X values: \n\n{X_sorted}')

rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, train_size=0.75, random_state=102)

# Outlier Handling
print('Initial shape of the training dataset', X_train.shape, y_train.shape)
# identify outliers in the training dataset
iso = IsolationForest(contamination=0.1)
x_out = iso.fit_predict(X_train)
# select all rows that are not outliers
mask = (x_out != -1)
X_train, y_train = X_train.iloc[mask, :], y_train.iloc[mask]
# summarize the shape of the updated training dataset
print('Shape of the training dataset after removing outliers', X_train.shape, y_train.shape)

# Outlier Handling
print('Initial shape of the training dataset', X_train.shape, y_train.shape)
# identify outliers in the training dataset
lof = LocalOutlierFactor(contamination=0.1)
x_out = lof.fit_predict(X_train)
# select all rows that are not outliers
mask = (x_out != -1)
X_train, y_train = X_train.iloc[mask, :], y_train.iloc[mask]
# summarize the shape of the updated training dataset
print('Shape of the training dataset after removing outliers', X_train.shape, y_train.shape)

batch_size = 1000
num_batches = len(X_train) // batch_size + 1

#Model used: Logistic regression
from sklearn.neural_network import MLPClassifier
import numpy as np

with mlflow.start_run():
    #Model used: MLPClassifier
    mlpclassifier = MLPClassifier(activation='relu',solver='adam',alpha=0.0001,max_iter=1000)



    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X_train))
        X_batch = X_train[start_idx:end_idx]
        y_batch = y_train[start_idx:end_idx]
        mlpclassifier.partial_fit(X_batch, y_batch, classes=np.unique(y_train))
        
    # print("Tuned logreg param: {}".format(mlpclass_cv.best_params_))
    # print("Tuned logreg Best Accuracy Score: {}".format(mlpclass_cv.best_score_))

    #Predicting X_test
    y_hat = mlpclassifier.predict(X_test)


    #ROC Curve
    y_pred_probs = mlpclassifier.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Dangerous prediction')
    plt.show()

    #accuracy score 
    print(f'\nAccuracy score: {accuracy_score(y_test,y_hat)}')
    print(f'\nClassification report: \n{classification_report(y_test,y_hat)}')
    print(f'\nConfusion matrix: \n{confusion_matrix(y_test,y_hat)}')
    print(f'\nROC accuracy score: {roc_auc_score(y_test, y_pred_probs)}')
    mlflow.sklearn.log_model(mlpclassifier, 'MLP Classifier Model')


