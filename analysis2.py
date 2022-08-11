import pandas as pd

df = pd.read_csv('owid_covid_19_dataset_bangladesh.csv')

# df = df.loc[df['Country_code'] == 'BD']
# df.to_csv('covid_19_dataset_bangladesh.csv',index=False)

# df_2020 = df[df['Date_reported'].between('2020-01-01', '2020-12-31', inclusive='both')]
# df = df.dropna(axis=1, how="any", thresh=None, subset=None, inplace=False)
df['year'] = pd.DatetimeIndex(df['date']).year
df.fillna(0,inplace=True)
# print(df.loc[df['year'] == 2022])

feature_names = ['new_cases', 'new_deaths','new_tests','aged_65_older','gdp_per_capita']
X = df[feature_names]
y = df['year']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5,train_size=0.75)
from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


#SVC classification
from sklearn.svm import SVC
svm = SVC(kernel="linear",gamma='auto',random_state=10)
svm.fit(X_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(svm.score(X_test, y_test)))
print('')

#AdaBoost classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
base = DecisionTreeClassifier()
adb = AdaBoostClassifier(n_estimators=500,base_estimator=base, learning_rate=0.1)
adb.fit(X_train, y_train)
print('Accuracy of AdaBoost classifier on training set: {:.2f}'
     .format(adb.score(X_train, y_train)))
print('Accuracy of AdaBoost classifier on test set: {:.2f}'
     .format(adb.score(X_test, y_test)))
print('')


#K-Means classification
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X_train, y_train)
print('Accuracy of KMeans classifier on training set: {:.2f}'
     .format(kmeans.score(X_train, y_train)))
print('Accuracy of KMeans classifier on test set: {:.2f}'
     .format(kmeans.score(X_test, y_test)))
print('')


#K-NN classification
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=36,)
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))
print('')


#Decision Tree classification
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy",max_depth=3)
dt.fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(dt.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(dt.score(X_test, y_test)))
print('')


# #Decision Tree classification
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1000, random_state=5,max_depth=2)
rf.fit(X_train, y_train)
print('Accuracy of Random Forest Tree classifier on training set: {:.2f}'
     .format(rf.score(X_train, y_train)))
print('Accuracy of Random Forest classifier on test set: {:.2f}'
     .format(rf.score(X_test, y_test)))
print('')


# #Decision Tree classification
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print('Accuracy of Naive Bayes classifier on training set: {:.2f}'
     .format(gnb.score(X_train, y_train)))
print('Accuracy of Naive Bayes classifier on test set: {:.2f}'
     .format(gnb.score(X_test, y_test)))
print('')


# Predict y data with classifier: 
from sklearn.metrics import classification_report,precision_recall_fscore_support,confusion_matrix,accuracy_score

y_pred_svm = svm.predict(X_test)
y_pred_knn = knn.predict(X_test)
y_pred_dt = dt.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_gnb= gnb.predict(X_test)
y_pred_adb= adb.predict(X_test)

# print(confusion_matrix(y_test, pred))
# print(classification_report(y_test, y_pred_knn))
print('\n\n')
clf_rpt_svm = ['SVM']+[round(item*100,2) for item in precision_recall_fscore_support(y_test, y_pred_svm,average='weighted') if item] + [round(accuracy_score(y_test,y_pred_svm)*100,2)]
clf_rpt_knn = ['KNN']+[round(item*100,2) for item in precision_recall_fscore_support(y_test, y_pred_knn,average='weighted') if item] + [round(accuracy_score(y_test,y_pred_knn)*100,2)]
clf_rpt_dt = ['DT']+[round(item*100,2) for item in precision_recall_fscore_support(y_test, y_pred_dt,average='weighted') if item] + [round(accuracy_score(y_test,y_pred_dt)*100,2)]
clf_rpt_rf = ['RF']+[round(item*100,2) for item in precision_recall_fscore_support(y_test, y_pred_rf,average='weighted') if item] + [round(accuracy_score(y_test,y_pred_rf)*100,2)]
clf_rpt_gnb = ['NB']+[round(item*100,2) for item in precision_recall_fscore_support(y_test, y_pred_gnb,average='weighted') if item] + [round(accuracy_score(y_test,y_pred_gnb)*100,2)]
clf_rpt_adb = ['ADB']+[round(item*100,2) for item in precision_recall_fscore_support(y_test, y_pred_adb,average='weighted') if item] + [round(accuracy_score(y_test,y_pred_adb)*100,2)]

from tabulate import tabulate
print(tabulate([clf_rpt_svm,clf_rpt_knn, clf_rpt_dt,clf_rpt_rf,clf_rpt_gnb,clf_rpt_adb], headers=['Technique','Precision', 'Recall', 'F1 Score', 'Accuracy']))

