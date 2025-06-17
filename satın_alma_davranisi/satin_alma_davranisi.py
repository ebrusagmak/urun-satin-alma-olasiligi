import pandas as pd 
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,recall_score
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


df=pd.read_csv("Social_Network_Ads.csv")

# print(df)
le=LabelEncoder()
df["Gender"]=le.fit_transform(df[["Gender"]])

# print(df["Gender"])

X=df[["Gender","Age","EstimatedSalary"]]
y=df["Purchased"]

# print(df.corr())

X_train,X_test,y_trian,y_test=train_test_split(X,y,test_size=0.33,random_state=42)

Standard_Scaler=StandardScaler()
X_train_scaler=Standard_Scaler.fit_transform(X_train)
X_test_scaler=Standard_Scaler.transform(X_test)

smote=SMOTE(random_state=42)
X_train_smote,y_train_smote=smote.fit_resample(X_train_scaler,y_trian)


# Logistic_Regression=LogisticRegression()
# Logistic_Regression.fit(X_train_scaler,y_trian)
# log_predict=Logistic_Regression.predict(X_test_scaler)
# # print("Logistik Regresyonun Tahmini: ",log_predict)

# accuracyscore=accuracy_score(y_test,log_predict)
# print("Logistik Regresyonun accuracy score'a göre doğruluk oranı: ",accuracyscore)

# confusionmatrix=confusion_matrix(y_test,log_predict)
# print("Logistik Regresyonun confusion matrix'e göre doğruluk oranı: ",confusionmatrix)

# f1score=f1_score(y_test,log_predict)
# print("Logistik Regresyonun f1 scoruna göre doğruluk oranı: ",f1score)
# recallscore=recall_score(y_test,log_predict)
# print("Logistik Regresyonun recall scoruna göre doğruluk oranı: ",recallscore)



Logistic_Regression_smote=LogisticRegression()
Logistic_Regression_smote.fit(X_train_smote,y_train_smote)
log_predict_smote=Logistic_Regression_smote.predict(X_test_scaler)

log_accuracyscore=accuracy_score(y_test,log_predict_smote)
print("Logistik Regresyonun accuracy score'a(smote) göre doğruluk oranı: ",log_accuracyscore)

log_confusionmatrix=confusion_matrix(y_test,log_predict_smote)
print("Logistik Regresyonun confusion matrix'e(smote) göre doğruluk oranı: ",log_confusionmatrix)

log_f1score=f1_score(y_test,log_predict_smote)
print("Logistik Regresyonun f1(smote) scoruna göre doğruluk oranı: ",log_f1score)
log_recallscore=recall_score(y_test,log_predict_smote)
print("Logistik Regresyonun recall scoruna(smote) göre doğruluk oranı: ",log_recallscore)

print("-----------------------------------------------------------------------------------------------------------------------")

DecisionTree_Classifier=DecisionTreeClassifier(random_state=42,max_depth=10,min_samples_leaf=10,min_samples_split=5,criterion="entropy",class_weight='balanced')
DecisionTree_Classifier.fit(X_train_smote,y_train_smote)
DecisionTree_Classifier_predict=DecisionTree_Classifier.predict(X_test_scaler)

tree_accuracyscore=accuracy_score(y_test,DecisionTree_Classifier_predict)
print("DecisionTreeClassifier'ın accuracy score'a göre doğruluk oranı: ",tree_accuracyscore)

tree_confusionmatrix=confusion_matrix(y_test,DecisionTree_Classifier_predict)
print("DecisionTreeClassifier'ın confusion matrix'e göre doğruluk oranı: ",tree_confusionmatrix)

tree_f1score=f1_score(y_test,DecisionTree_Classifier_predict)
print("DecisionTreeClassifier'ın f1 scoruna göre doğruluk oranı: ",tree_f1score)

tree_recallscore=recall_score(y_test,log_predict_smote)
print("DecisionTreeClassifier'ın recall scoruna göre doğruluk oranı: ",tree_recallscore)

print("----------------------------------------------------------------------------------------------------")

naive_bayes=GaussianNB()

naive_bayes.fit(X_train_smote,y_train_smote)
naive_bayes_predict=naive_bayes.predict(X_test_scaler)


naive_accuracyscore=accuracy_score(y_test,naive_bayes_predict)
print("Naive Bayes'in accuracy score'a göre doğruluk oranı: ",naive_accuracyscore)

naive_confusionmatrix=confusion_matrix(y_test,naive_bayes_predict)
print("Naive Bayes'in confusion matrix'e göre doğruluk oranı: ",naive_confusionmatrix)

naive_f1score=f1_score(y_test,naive_bayes_predict)
print("Naive Bayes'in f1 scoruna göre doğruluk oranı: ",naive_f1score)

naive_recall_score=recall_score(y_test,naive_bayes_predict)
print("Naive Bayes'in recall scoruna göre doğruluk oranı: ",naive_recall_score)

print("--------------------------------------------------------------------------------------")

RandomForest_Classifier=RandomForestClassifier(random_state=42,n_estimators=100,max_depth=2)
RandomForest_Classifier.fit(X_train_smote,y_train_smote)
RandomForest_Classifier_predict=RandomForest_Classifier.predict(X_test_scaler)

random_accuracyscore=accuracy_score(y_test,RandomForest_Classifier_predict)
print("Random Forest'in accuracy score'a göre doğruluk oranı: ",random_accuracyscore)

random_confusionmatrix=confusion_matrix(y_test,RandomForest_Classifier_predict)
print("Random Forest'in confusion matrix'e göre doğruluk oranı: ",random_confusionmatrix)

random_f1score=f1_score(y_test,RandomForest_Classifier_predict)
print("Random Forest'in f1 scoruna göre doğruluk oranı: ",random_f1score)

random_recall_score=recall_score(y_test,RandomForest_Classifier_predict)
print("Random Forest'in recall scoruna göre doğruluk oranı: ",random_recall_score)

print("--------------------------------------------------------------------------------------------------")


svm_model=SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
svm_model.fit(X_train_smote,y_train_smote)
svm_model_predict=svm_model.predict(X_test_scaler)

svm_accuracyscore=accuracy_score(y_test,svm_model_predict)
print("SVM model'in accuracy score'a göre doğruluk oranı: ",svm_accuracyscore)

svm_confusionmatrix=confusion_matrix(y_test,svm_model_predict)
print("SVM model'in confusion matrix'e göre doğruluk oranı: ",svm_confusionmatrix)

svm_f1score=f1_score(y_test,svm_model_predict)
print("SVM model'in f1 scoruna göre doğruluk oranı: ",svm_f1score)

svm_recall_score=recall_score(y_test,svm_model_predict)
print("SVM model'in recall scoruna göre doğruluk oranı: ",svm_recall_score)

print("------------------------------------------------------------------------------------------------------")


kneighborsgraph=KNeighborsClassifier(n_neighbors=5)
kneighborsgraph.fit(X_train_smote,y_train_smote)
kneighborsgraph_predict=kneighborsgraph.predict(X_test_scaler)

knn_accuracyscore=accuracy_score(y_test,kneighborsgraph_predict)
print("KNN'nin accuracy score'a göre doğruluk oranı: ",knn_accuracyscore)

knn_confusionmatrix=confusion_matrix(y_test,kneighborsgraph_predict)
print("KNN'nin confusion matrix'e göre doğruluk oranı: ",knn_confusionmatrix)

knn_f1score=f1_score(y_test,kneighborsgraph_predict)
print("KNN'nin f1 scoruna göre doğruluk oranı: ",knn_f1score)

knn_recall_score=recall_score(y_test,kneighborsgraph_predict)
print("KNN'nin recall scoruna göre doğruluk oranı: ",knn_recall_score)















