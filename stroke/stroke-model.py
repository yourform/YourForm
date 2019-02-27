import pandas as pd
import numpy as np
import joblib
from matplotlib import pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler,SMOTE, ADASYN
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,auc,roc_auc_score,precision_score,recall_score

train_data = pd.read_csv('stroke_trainset.csv')
test_data = pd.read_csv('stroke_testset.csv')
train_data.drop(axis=1,columns=['id', 'avg_glucose_level'],inplace=True)
test_data.drop(axis=1,columns=['id', 'avg_glucose_level'],inplace=True)

train_data.shape

# ### Data Cleaning
# ### Missing Values for Train and Test Data
train_data.isnull().sum()/len(train_data)*100
test_data.isnull().sum()/len(test_data)*100
joined_data = pd.concat([train_data,test_data])
print ('Joined Data Shape: {}'.format(joined_data.shape))

# ### Missing Data for Joined Data
joined_data.isnull().sum()/len(joined_data)*100

# ### Joined Data has bmi 3.33% data is missing and smoking_status is 30.7% missing
train_data["bmi"]=train_data["bmi"].fillna(train_data["bmi"].mean())

train_data.head()

# ### Handling Categorical Variables
label = LabelEncoder()
train_data['gender'] = label.fit_transform(train_data['gender'])
train_data['ever_married'] = label.fit_transform(train_data['ever_married'])
train_data['work_type']= label.fit_transform(train_data['work_type'])
train_data['Residence_type']= label.fit_transform(train_data['Residence_type'])

train_data_without_smoke = train_data[train_data['smoking_status'].isnull()]
train_data_with_smoke = train_data[train_data['smoking_status'].notnull()]

train_data_without_smoke.drop(columns='smoking_status',axis=1,inplace=True)

train_data_with_smoke['smoking_status']= label.fit_transform(train_data_with_smoke['smoking_status'])
ros = RandomOverSampler(random_state=0)
smote = SMOTE()
X_resampled, y_resampled = ros.fit_resample(train_data_with_smoke.loc[:,train_data_with_smoke.columns!='stroke'], 
                                            train_data_with_smoke['stroke'])
train_data_without_smoke.loc[:,train_data_without_smoke.columns!='stroke'].columns

print ('ROS Input Data Shape for Smoke Data: {}'.format(X_resampled.shape))
print ('ROS Output Data Shape for Smoke Data: {}'.format(y_resampled.shape))

X_resampled_1, y_resampled_1 = ros.fit_resample(train_data_without_smoke.loc[:,train_data_without_smoke.columns!='stroke'], 
                                            train_data_without_smoke['stroke'])

print ('ROS Input Data Shape for Non Smoke Data: {}'.format(X_resampled_1.shape))
print ('ROS Output Data Shape for Non Smoke Data: {}'.format(y_resampled_1.shape))

X_train,X_test,y_train,y_test = train_test_split(X_resampled,y_resampled,test_size=0.2)
print(X_train.shape)
print(X_test.shape)

X_train_1,X_test_1,y_train_1,y_test_1 = train_test_split(X_resampled_1,y_resampled_1,test_size=0.2)
print(X_train_1.shape)
print(X_test_1.shape)

ran = RandomForestClassifier(n_estimators=50,random_state=0)
ran.fit(X_train_1,y_train_1)

pred = ran.predict(X_test_1)
print(classification_report(y_test_1,pred))
print (accuracy_score(y_test_1,pred))
print (confusion_matrix(y_test_1,pred))

precision = precision_score(y_test_1,pred)
recall = recall_score(y_test_1,pred)
print( 'precision = ', precision, '\n', 'recall = ', recall)

y_pred_proba = ran.predict_proba(X_test_1)[::,1]
fpr, tpr, _ = roc_curve(y_test_1,  y_pred_proba)
auc = roc_auc_score(y_test_1, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


impFeatures = pd.DataFrame((ran.feature_importances_) ,index=train_data_without_smoke.loc[:,train_data_without_smoke.columns!='stroke'].columns,columns=['Importance']).sort_values(by='Importance',ascending=False)
print (impFeatures)

# joblib.dump(ran, 'stroke_model.joblib')
import pickle
with open('model.pkl', 'wb') as model_file:
      pickle.dump(ran, model_file)

train_data_without_smoke.to_csv('out')
