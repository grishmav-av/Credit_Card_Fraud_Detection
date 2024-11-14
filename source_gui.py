#LOADING DATASET & PROCESSING
import pandas as pd
from collections import Counter
df = pd.read_csv('c:\\Users\\grish\\OneDrive\\Desktop\\Credit Card Fraud Detection\\card_transdata.csv')

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
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=22,test_size=0.1)

    #scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#CREATING A STREAMLIT WEBPAGE
import streamlit as st
from streamlit_option_menu import option_menu

st.title('CREDIT CARD FRAUD DETECTOR')
with st.sidebar:
    option = option_menu('SELECT A MODEL',options=['LOGISTIC REGRESSION','DECISION TREE','NAIVE BAYES','KNN','MODEL GRAPHS'])

yes_no = {"YES":1,"NO":0}

distance_from_home = st.text_input('ENTER THE DISTANCE OF TRANSACTION : ')
distance_from_last_transaction = st.text_input('ENTER THE DISTANCE BETWEEN LAST 2 TRANSACTIONS : ')
ratio_to_median_purchase_price= st.text_input('ENTER AMOUNT TO AVERAGE PURCHASE PRICE RATIO  : ')

repeat_retailer = st.selectbox("HAS THE CUSTOMER PREVIOUSLY MADE PAYMENT TO THIS RETAILER?",options={"YES","NO"})
repeat_retailer = yes_no[repeat_retailer]

used_chip = st.selectbox("WAS TRANSACTION MADE USING CARD TAP?",options={"YES","NO"})
used_chip = yes_no[used_chip]

used_pin_number = st.selectbox("WAS CREDIT CARD PIN USED?",options={"YES","NO"})
used_pin_number = yes_no[used_pin_number]

online_order = st.selectbox("IS THE TRANSACTION ONLINE?",options={"YES","NO"})
online_order = yes_no[online_order]

submit = st.button('SUBMIT')

new = [[distance_from_home,distance_from_last_transaction,
        ratio_to_median_purchase_price,repeat_retailer,used_chip,used_pin_number,
        online_order]]

#TRAINING THE ML MODELS & INTEGRATING WITH WEBPAGE
if submit:
    if option == 'LOGISTIC REGRESSION':
        #LOGISTIC REGRESSION
        from sklearn.linear_model import LogisticRegression#importing the package
        logistic_regression = LogisticRegression()#calling the function
        logistic_regression.fit(x_train,y_train)#training the model
        logistic_regression_test_prediction = logistic_regression.predict(x_test)#using the trained model to predict for x_test
        logistic_regression_new_prediction = logistic_regression.predict(sc.transform(new))#using the model to predict for new user input
            #DISPLAYING MODEL RESULTS
        st.header("RESULT OF LOGISTIC REGRESSION MODEL IS : ")
        if logistic_regression_new_prediction[0] == 1:
            st.error("THERE IS A POTENTIAL CREDIT CARD FRAUD")
        else:
            st.success("NO CREDIT CARD FRAUD DETECTED")
            #DISPLAYING MODEL PARAMETERS
        st.header("MODEL PARAMETERS : ")
        from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score#importing metrics
        st.info("accuracy of the model : {0}%".format(accuracy_score(y_test, logistic_regression_test_prediction) * 100))
        st.error("precision of the model : {0}%".format(precision_score(y_test, logistic_regression_test_prediction) * 100))
        st.warning("recall score of the model : {0}%".format(recall_score(y_test, logistic_regression_test_prediction) * 100))
        st.success("f1 score of the model : {0}%".format(f1_score(y_test, logistic_regression_test_prediction) * 100))
            #DISPLAYING CLASSIFICATION REPORT
        from sklearn.metrics import classification_report#importing
        st.header('CLASSIFICATION REPORT')
        report = classification_report(y_test, logistic_regression_test_prediction, output_dict=True)  # creating classification report
        report_df = pd.DataFrame(report).transpose()  # converting the report into a dataframe
        st.dataframe(report_df, height=212, width=1000)  # displaying the dataframe on the webpage

    if option == 'DECISION TREE':
        #DECISION TREE
        from sklearn.tree import DecisionTreeClassifier
        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(x_train,y_train)
        decision_tree_test_prediction = decision_tree.predict(x_test)
        decision_tree_new_prediction = decision_tree.predict(sc.transform(new))
            #DISPLAYING MODEL RESULTS
        st.header("RESULT OF DECISION TREE MODEL IS : ")
        if decision_tree_new_prediction[0] == 1:
            st.error("THERE IS A POTENTIAL CREDIT CARD FRAUD")
        else:
            st.success("NO CREDIT CARD FRAUD DETECTED")
            #DISPLAYING MODEL PARAMETERS
        st.header("MODEL PARAMETERS : ")
        from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score#importing metrics
        st.info("accuracy of the model : {0}%".format(accuracy_score(y_test, decision_tree_test_prediction) * 100))
        st.error("precision of the model : {0}%".format(precision_score(y_test, decision_tree_test_prediction) * 100))
        st.warning("recall score of the model : {0}%".format(recall_score(y_test, decision_tree_test_prediction) * 100))
        st.success("f1 score of the model : {0}%".format(f1_score(y_test, decision_tree_test_prediction) * 100))
            #DISPLAYING CLASSIFICATION REPORT
        from sklearn.metrics import classification_report#importing
        st.header('CLASSIFICATION REPORT')
        report = classification_report(y_test, decision_tree_test_prediction, output_dict=True)  # creating classification report
        report_df = pd.DataFrame(report).transpose()  # converting the report into a dataframe
        st.dataframe(report_df, height=212, width=1000)  # displaying the dataframe on the webpage

    if option == 'NAIVE BAYES':
        #RANDOM FOREST
        from sklearn.naive_bayes import GaussianNB
        naive_bayes = GaussianNB()
        naive_bayes.fit(x_train,y_train)
        naive_bayes_test_prediction = naive_bayes.predict(x_test)
        naive_bayes_new_prediction = naive_bayes.predict(sc.transform(new))
            #DISPLAYING MODEL RESULTS
        st.header("RESULT OF NAIVE BAYES MODEL IS : ")
        if naive_bayes_new_prediction[0] == 1:
            st.error("THERE IS A POTENTIAL CREDIT CARD FRAUD")
        else:
            st.success("NO CREDIT CARD FRAUD DETECTED")
            #DISPLAYING MODEL PARAMETERS
        st.header("MODEL PARAMETERS : ")
        from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score#importing metrics
        st.info("accuracy of the model : {0}%".format(accuracy_score(y_test, naive_bayes_test_prediction) * 100))
        st.error("precision of the model : {0}%".format(precision_score(y_test, naive_bayes_test_prediction) * 100))
        st.warning("recall score of the model : {0}%".format(recall_score(y_test, naive_bayes_test_prediction) * 100))
        st.success("f1 score of the model : {0}%".format(f1_score(y_test, naive_bayes_test_prediction) * 100))
            #DISPLAYING CLASSIFICATION REPORT
        from sklearn.metrics import classification_report#importing
        st.header('CLASSIFICATION REPORT')
        report = classification_report(y_test, naive_bayes_test_prediction, output_dict=True)  # creating classification report
        report_df = pd.DataFrame(report).transpose()  # converting the report into a dataframe
        st.dataframe(report_df, height=212, width=1000)  # displaying the dataframe on the webpage

    if option == 'KNN':
        #RANDOM FOREST
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier()
        knn.fit(x_train,y_train)
        knn_test_prediction = knn.predict(x_test)
        knn_new_prediction = knn.predict(sc.transform(new))
            #DISPLAYING MODEL RESULTS
        st.header("RESULT OF KNN MODEL IS : ")
        if knn_new_prediction[0] == 1:
            st.error("THERE IS A POTENTIAL CREDIT CARD FRAUD")
        else:
            st.success("NO CREDIT CARD FRAUD DETECTED")
            #DISPLAYING MODEL PARAMETERS
        st.header("MODEL PARAMETERS : ")
        from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score#importing metrics
        st.info("accuracy of the model : {0}%".format(accuracy_score(y_test, knn_test_prediction) * 100))
        st.error("precision of the model : {0}%".format(precision_score(y_test, knn_test_prediction) * 100))
        st.warning("recall score of the model : {0}%".format(recall_score(y_test, knn_test_prediction) * 100))
        st.success("f1 score of the model : {0}%".format(f1_score(y_test, knn_test_prediction) * 100))
            #DISPLAYING CLASSIFICATION REPORT
        from sklearn.metrics import classification_report#importing
        st.header('CLASSIFICATION REPORT')
        report = classification_report(y_test, knn_test_prediction, output_dict=True)  # creating classification report
        report_df = pd.DataFrame(report).transpose()  # converting the report into a dataframe
        st.dataframe(report_df, height=212, width=1000)  # displaying the dataframe on the webpage

    if option == 'MODEL GRAPHS':
        # DISPLAYING MODEL GRAPHS
        st.header("MODEL GRAPHS")
        import matplotlib.pyplot as plt

        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(10, 6))
        ax = (df['repeat_retailer'].value_counts() * 100.0 / len(df)) \
            .plot.pie(autopct='%.1f%%', labels=['Yes', 'No'], fontsize=12,explode=[0.1,0.05],startangle=225)
        st.pyplot(fig=fig)
        st.text('fig 1.1 PIE PLOT OF "REPEAT RETAILER"')

        plt.style.use('fivethirtyeight')
        fig = plt.figure(figsize=(10, 6))
        ax = (df['used_chip'].value_counts() * 100.0 / len(df)) \
            .plot.pie(autopct='%.1f%%', labels=['Yes', 'No'], fontsize=12,explode=[0.1,0.05],startangle=90)
        st.pyplot(fig=fig)
        st.text('fig 1.2 PIE PLOT OF "USED CHIP"')

        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(10, 6))
        ax = (df['used_pin_number'].value_counts() * 100.0 / len(df)) \
            .plot.pie(autopct='%.1f%%', labels=['Yes', 'No'], fontsize=12,explode=[0.1,0.05],startangle=135)
        st.pyplot(fig=fig)
        st.text('fig 1.3 PIE PLOT OF "USED PIN NUMBER"')

        plt.style.use('fivethirtyeight')
        fig = plt.figure(figsize=(10, 6))
        ax = (df['online_order'].value_counts() * 100.0 / len(df)) \
            .plot.pie(autopct='%.1f%%', labels=['Yes', 'No'], fontsize=12,explode=[0.1,0.05],startangle=270)
        st.pyplot(fig=fig)
        st.text('fig 1.4 PIE PLOT OF "ONLINE ORDER"')

else:
    st.error("CLICK SUBMIT")