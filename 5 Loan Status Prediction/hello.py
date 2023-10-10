import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

loan_dataset = pd.read_csv('loan_data.csv')

loan_dataset = loan_dataset.dropna()
loan_dataset.isnull().sum()

# lable endcoding
loan_dataset.replace({'Loan_Status':{'N':0,'Y':1}},inplace=True)

# replacing all the values 3+ to 4
update_dataset = loan_dataset.replace(to_replace='3+',value=4)

update_dataset['Dependents'].value_counts()

# convert cetagorical columls to numerical values
update_dataset.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)

# separating the data and label
x = update_dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
y = update_dataset['Loan_Status']

# train test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

print('x shape:...',x.shape)
print('x_train shalpe:...',x_train.shape)
print('x_test:...',x_test.shape)

# support vector machine model
classifier = svm.SVC(kernel='linear')

# training the support vector machine model
classifier.fit(x_train,y_train)

# accurac score on training data
x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction,y_train)
print(training_data_accuracy)

# accuracy score on tasting data
x_test_prediction = classifier.predict(x_test)
training_data_accuracy = accuracy_score(x_test_prediction,y_test)
print(training_data_accuracy)