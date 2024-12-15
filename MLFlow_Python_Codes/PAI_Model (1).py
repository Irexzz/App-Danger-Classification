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
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

sensor_df = pd.read_csv("C:/Users/aiman/Downloads/sensor_df.csv")
driver_df = pd.read_csv("C:/Users/aiman/Downloads/driver_df.csv")
safety_df = pd.read_csv("C:/Users/aiman/Downloads/safety_status.csv")

df = pd.merge(safety_df, sensor_df, on='booking_key', how='inner')
df = pd.merge(df, driver_df, on='driver_key', how='inner')
df = df.drop_duplicates(subset=['booking_key'])

dangerous_percentage = df.groupby('driver_key')['dangerous'].sum() / df.groupby('driver_key')['dangerous'].count() * 100

# Create a new DataFrame to display the results
result_df = pd.DataFrame({
    'driver_key': dangerous_percentage.index,
    'dangerous_percentage': dangerous_percentage.values
})
#print(result_df)

result_df_filtered = result_df[result_df['dangerous_percentage'] > 20]
df = pd.merge(df, result_df, on='driver_key', how='inner')
df['dangerous'] = df['dangerous_percentage'].apply(lambda x: 1 if x > 20 else 0)
df = df.drop_duplicates(subset=['driver_key'])
columns_to_drop = ['booking_key', 'driver_key','sensor_key','acceleration_x','acceleration_y','acceleration_z','accuracy','bearing','gyro_x','gyro_y','gyro_z','driver_name','seconds','speed']
df = df.drop(columns=columns_to_drop)

label_encoder = LabelEncoder()
label_encoder2 = LabelEncoder()

df['date_of_birth'] = df['date_of_birth'].str[:4].astype(int)

df['gender'] = label_encoder.fit_transform(df['gender'])
df['car_brand'] = label_encoder2.fit_transform(df['car_brand'])

gender_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
car_brand_mapping = dict(zip(label_encoder2.classes_, label_encoder2.transform(label_encoder2.classes_)))

# Print the mappings
#print("Gender mapping:", gender_mapping)
#print("Car brand mapping:", car_brand_mapping)

X = df.drop(['dangerous','dangerous_percentage'], axis=1)
y=df['dangerous']
#X#function StandardScaler before and after
def aftstd(data):
    std_scaler = StandardScaler()
    df_std = std_scaler.fit(data)
    df_std = df_std.transform(data)
    plt.boxplot(df_std)
    plt.title('After Standardisation')
   # plt.show()
    return df_std

def befstd(data):
    plt.boxplot(data)
    plt.title('Before Standardisation')
   # plt.show()



to_standardize = X

print("Standardization of X")
#Before standardization
befstd(to_standardize)
# standardization
X_scaled = pd.DataFrame(aftstd(to_standardize))
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X_ros, y_ros = ros.fit_resample(X_scaled, y)
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_ros, y_ros, train_size=0.75, random_state=42)
# define model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
# make prediction
y_pred = lr_model.predict(X_test)


best_params_rf = {'bootstrap': False,
                  'class_weight': None,
                   'max_depth': 30,
                   'max_features': 'log2',
                   'min_samples_leaf': 1,
                   'min_samples_split': 6,
                   'n_estimators': 110,
                   'random_state':42}

best_params_gb =  {'learning_rate': 0.2,
                   'max_depth': 5,
                   'max_features': 'sqrt',
                   'min_samples_leaf': 1,
                   'min_samples_split': 5,
                   'n_estimators': 150,
                   'subsample': 1.0,
                   'random_state':42}  
    
best_params_mlp = {'activation': 'logistic',
                   'alpha': 0.001,
                   'early_stopping': True,
                   'hidden_layer_sizes': (100,),
                   'learning_rate': 'constant',
                   'max_iter': 200,
                   'solver': 'lbfgs',
                   'validation_fraction': 0.1,
                   'random_state':42}


import mlflow
with mlflow.start_run():

    # Log parameters for Random Forest Classifier
    for param_name, param_value in best_params_rf.items():
        mlflow.log_param(f"rf_{param_name}", param_value)

    rf_classifier = RandomForestClassifier(**best_params_rf)
    rf_classifier.fit(X_train, y_train)
    y_pred_rf = rf_classifier.predict(X_test)

    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    y_pred_proba_rf = rf_classifier.predict_proba(X_test)
    roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf[:, 1], multi_class='ovr')
    
    mlflow.log_metric("roc_auc_rf", roc_auc_rf)
    mlflow.log_metric("accuracy_rf", accuracy_rf)
    
    mlflow.sklearn.log_model(rf_classifier, "random_forest_model")

    # Log parameters for Logistic Regression
    for param_name, param_value in best_params_gb.items():
        mlflow.log_param(f"gb_{param_name}", param_value)

    gb_classifier = GradientBoostingClassifier(**best_params_gb)
    gb_classifier.fit(X_train, y_train)
    y_pred_gb = gb_classifier.predict(X_test)
    
    y_pred_proba_gb = gb_classifier.predict_proba(X_test)
    roc_auc_gb = roc_auc_score(y_test, y_pred_proba_gb[:, 1], multi_class='ovr')
    accuracy_gb = accuracy_score(y_test, y_pred_gb)
    
    mlflow.log_metric("roc_auc_gb", roc_auc_gb)
    mlflow.log_metric("accuracy_gb", accuracy_gb)

    mlflow.sklearn.log_model(gb_classifier, "gradient_boosting_model")
    
    for param_name, param_value in best_params_mlp.items():
        mlflow.log_param(f"mlp_{param_name}", param_value)

    mlp_classifier = MLPClassifier(**best_params_mlp)
    mlp_classifier.fit(X_train, y_train)
    y_pred_mlp = mlp_classifier.predict(X_test)
    
    accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
    y_pred_proba_mlp = mlp_classifier.predict_proba(X_test)
    roc_auc_mlp = roc_auc_score(y_test, y_pred_proba_mlp[:, 1], multi_class='ovr')
    
    mlflow.log_metric("roc_auc_mlp", roc_auc_mlp)
    mlflow.log_metric("accuracy_mlp", accuracy_mlp)

    mlflow.sklearn.log_model(mlp_classifier, "MLP_classifier_model")
