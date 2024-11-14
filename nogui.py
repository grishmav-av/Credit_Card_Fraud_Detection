#LOADING DATASET & PROCESSING
import pandas as pd
from collections import Counter
df = pd.read_csv('card_transdata.csv')

print(df.columns[df.isna().any()])

    #performing basic EDA
# print(df.shape)
# print(df.describe())
# print(df.info())

# print(df['fraud'].value_counts())

#SEGGREGATING & SPLITTING DATASET
df = df.sample(frac=1)    #shuffling the dataset

    # seggreagting dataset into input and output
x = df.iloc[:,:-1].values#selecting input columns
y = df.iloc[:,-1].values#selecting output column

    #since the dataset is imbalanced we will use a technique called SMOTE to balance the dataset

print(Counter(y))#checking output distribution before SMOTE
from imblearn.over_sampling import SMOTE#importing SMOTE
sm = SMOTE()#calling smote
X,Y = sm.fit_resample(x,y)#resampling using SMOTE
print(Counter(Y))#checking output distribution after SMOTE

    #splitting dataset into training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_s4plit(x,y,random_state=22,test_size=0.1)

    #scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#TRAINING THE ML MODELS
    #LINEAR REGRESSION
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
lr_predicted_x_test = lr.predict(x_test)

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
print("ACCURACY : ",accuracy_score(y_test,lr_predicted_x_test))
print("PRECISION : ",precision_score(y_test,lr_predicted_x_test))
print("RECALL : ",recall_score(y_test,lr_predicted_x_test))
print("F1 SCORE : ",f1_score(y_test,lr_predicted_x_test))
    #DECISION TREE
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
dt_predicted_x_test = dt.predict(x_test)

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
print("ACCURACY : ",accuracy_score(y_test,dt_predicted_x_test))
print("PRECISION : ",precision_score(y_test,dt_predicted_x_test))
print("RECALL : ",recall_score(y_test,dt_predicted_x_test))
print("F1 SCORE : ",f1_score(y_test,dt_predicted_x_test))





