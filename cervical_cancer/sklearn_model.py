import joblib
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn import model_selection
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from imblearn.over_sampling import SMOTE, ADASYN

cancer_df = pd.read_csv('kag_risk_factors_cervical_cancer.csv')

numerical_df = ['Age', 'Number of sexual partners', 'First sexual intercourse','Num of pregnancies', 'Smokes (years)',
                'Smokes (packs/year)','Hormonal Contraceptives (years)','IUD (years)','STDs (number)']
categorical_df = ['Smokes','Hormonal Contraceptives','IUD','STDs','STDs:condylomatosis','STDs:cervical condylomatosis',
                  'STDs:vaginal condylomatosis','STDs:vulvo-perineal condylomatosis', 'STDs:syphilis',
                  'STDs:pelvic inflammatory disease', 'STDs:genital herpes','STDs:molluscum contagiosum', 'STDs:AIDS',
                  'STDs:HIV','STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis','Dx:Cancer', 'Dx:CIN',
                  'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller','Citology', 'Biopsy']

cancer_df = cancer_df.replace('?', np.NaN)

### Filling the missing values of numeric data columns with mean of the column data.
for feature in numerical_df:
    print(feature,'', pd.to_numeric(cancer_df[feature]).mean())
    feature_mean = round(pd.to_numeric(cancer_df[feature]).mean(),1)
    cancer_df[feature] = cancer_df[feature].fillna(feature_mean)

for feature in categorical_df:
    cancer_df[feature] = pd.to_numeric(cancer_df[feature]).fillna(1.0)

category_df = ['Hinselmann', 'Schiller','Citology', 'Biopsy']

cancer_df['Number of sexual partners'] = pd.to_numeric(cancer_df['Number of sexual partners']).round()
cancer_df['First sexual intercourse'] = pd.to_numeric(cancer_df['First sexual intercourse'])
cancer_df['Num of pregnancies']= pd.to_numeric(cancer_df['Num of pregnancies']).round()
cancer_df['Smokes'] = pd.to_numeric(cancer_df['Smokes'])
cancer_df['Smokes (years)'] = pd.to_numeric(cancer_df['Smokes (years)'])
cancer_df['Hormonal Contraceptives'] = pd.to_numeric(cancer_df['Hormonal Contraceptives'])
cancer_df['Hormonal Contraceptives (years)'] = pd.to_numeric(cancer_df['Hormonal Contraceptives (years)'])
cancer_df['IUD (years)'] = pd.to_numeric(cancer_df['IUD (years)'])

print('minimum:',min(cancer_df['Hormonal Contraceptives (years)']))
print('maximum:',max(cancer_df['Hormonal Contraceptives (years)']))

cancer_df['Smokes (packs/year)'] = pd.to_numeric(cancer_df['Smokes (packs/year)'])
print('Correlation between Smokes and Smokes (years) feature:',cancer_df['Smokes'].corr(cancer_df['Smokes (years)']))
print('Correlation between Smokes and Smokes (packs/year) feature:',cancer_df['Smokes'].corr(cancer_df['Smokes (packs/year)']))

cancer_df_label = pd.DataFrame(data=cancer_df['Hinselmann'])

cancer_df_label['Schiller'] = cancer_df['Schiller']
cancer_df_label['Citology'] = cancer_df['Citology']
cancer_df_label['Biopsy'] = cancer_df['Biopsy']

def cervical_cancer(cancer_label):
    hil, sch, cit, bio = cancer_label
    return hil+sch+cit+bio

cancer_df_label['cervical_cancer'] = cancer_df_label[['Hinselmann', 'Schiller', 'Citology','Biopsy']].apply(cervical_cancer,axis=1)
cancer_df_label.drop(['Hinselmann', 'Schiller', 'Citology','Biopsy'],axis=1,inplace=True)

print('Value counts of each target variable:',cancer_df_label['cervical_cancer'].value_counts())
cancer_df_label = cancer_df_label.astype(int)
cancer_df_label = cancer_df_label.values.ravel()

cancer_df_features = cancer_df[['Age', 'Number of sexual partners', 'First sexual intercourse',
    'Num of pregnancies', 'Smokes (years)', 'Smokes (packs/year)', 'Hormonal Contraceptives (years)',
    'IUD (years)', 'STDs', 'STDs:condylomatosis', 'STDs:cervical condylomatosis',
    'STDs:vaginal condylomatosis', 'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis',
    'STDs:pelvic inflammatory disease', 'STDs:genital herpes', 'STDs:molluscum contagiosum',
    'STDs:AIDS', 'STDs:HIV', 'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis']]

print('Final feature vector shape:',cancer_df_features.shape)
print('Final target vector shape',cancer_df_label.shape)


# evaluate each model in turn
results_all = []
names = []
dict_method_score = {}
scoring = 'recall_weighted'

### Building a model for future predictions:
random_forest_model = RandomForestClassifier(n_jobs=4, bootstrap=True, class_weight=None, criterion='gini',
        max_depth=None, max_features='auto', max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2,
        min_weight_fraction_leaf=0.0, n_estimators=10, oob_score=False, random_state=None, verbose=0,warm_start=False)

random_forest_model.fit(cancer_df_features,cancer_df_label)
joblib.dump(random_forest_model, 'model.joblib')

print('Testing the model on women with age less than 20:',
        random_forest_model.predict(np.array([[19,1,17,1,1,3.4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])))
print('Testing the model on women with age 35',
        random_forest_model.predict(np.array([[35,5,11,2,15,15,14,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])))
print('Testing the model on Raw Data:',
        random_forest_model.predict(np.array([[48,2,15,2,0,0,0.5,19,0,0,0,0,0,0,0,1,0,0,0,0,0,1]])))

