import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


##Part 1 --- Data Pre-Processing

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values ##Row Number,Customer ID,Surname has no impact on the final output. So exclude them
                                ##Row Number's index = 0
                                ##3-12 are independent variables. But write 13 because it is [  )
y = dataset.iloc[:,13].values

##Data has categorical values so encoding has to be done
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
##3 Countries and 2 Genders
labelencoder_X_1 = LabelEncoder() ## Created object for Countries
X[:,1] = labelencoder_X_1.fit_transform(X[:,1]) ##Encoded

labelencoder_X_2 = LabelEncoder() ## Created object for Gender
X[:,2] = labelencoder_X_2.fit_transform(X[:,2]) ## Encoded

## It gives the value 0,1,2 to Germany Spain and France.
## But Germany is not less or greater than others so ---Convert into dummy variables  using OneHotEncoder
## or gender there is not need since it is 0 or 1 only
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

## Avoid Dummy Variable trap for the countries
X = X[:,1:] ## Just remove the first dummy variable

##Splitiing the data in test and training data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

##Feature Scaling -- VERY VERY COMPULOSORY FOR ANN's
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



## Part 2 ---Making the ANN
import keras
from keras.models import Sequential ##This module will be used to initialize the NN
from keras.layers import Dense ## This module will be used to form the layers

## There are two ways to initialize the NN -- 1) Defining the layers one by one. 
## Here we will do it by layers               2) By graph
##It's a classification problem
classifier = Sequential() ## classifier is the name of object which we create in Sequential class. We will be adding layers one by one starting from input layer. Classifier object is basically our ANN Model

##Adding Input and 1st Hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_shape = (11,))) ## This is the first hidden layer(no. of nodes in hidden layer = average of output layer & input layer) and input layer(no. of nodes =  11 as you can see the variables) is also added along with it relu  = rectifier function
## No. of nodes hidden layer is an art, based on many experiments, experience, model selection technique. But for safety, take them as the average.

## Adding the second hiden layer
classifier.add(Dense(6,kernel_initializer='uniform', activation = 'relu',)) 
## Second hidden layer, no need to add input because its already done

##Adding the output layer
classifier.add(Dense(1,kernel_initializer='uniform', activation = 'sigmoid')) ## This is to add the output layer. 1 because we only have to classifiy whether they will leave or stay. Sigmoid activation is used so as to determine the probability(for 2 categories only)
## If you had to classifiy 3 categories, then size had to be 3 so as to classifiy.But here since it 2, so size = 1
## Activation function will be softmax insted of SIGMOID !!!!!(If more than 2 categories)


## Part 3 -- Compiling the ANN


## Applying Stochastic Gradient Descent on the model
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
## Arguments being used :-
## Optimizer --  is the algo we are using
                 ##Since the wiights were just initialized, we need an algorithm to assign best possible weights.(Basically Stochastic GD)
                 ## There are several types of stochastic GD,, we will use --'adam.
## Loss -- Refer Notes
           ##Since we have binary output. Hence we use binary_crossentropy.
           ##IF WE HAD more than 2 categories to classify, use - categorial_crossentropy
## Metrics -- Accuracy
              ## Keeps calculating the accuracy at every stage

##Fitting the ANN to our training dataset
classifier.fit(X_train,y_train,batch_size = 10, nb_epoch = 100)
##Arguments being used :-
## batch_size -- update the weights after this no. of observations.(How to choose a batch size -  art/experience/more techniques/parameter tuning)
## epochs -- How many times will it pass the ANN


## Part 4 -- Predicting the result
y_pred = classifier.predict(X_test)
##This will give the probailites of the person leaving the bank
##Convert it into binary form. That is true or false
y_pred = (y_pred > 0.5)
## 0.5 was taken as the threshold. But it can vary. For example - For cancer to be malignant, it can be 0.7

##Making the confusion matrix to check accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
##Refrer notes
