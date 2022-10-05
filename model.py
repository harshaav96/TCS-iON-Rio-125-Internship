# Smartphone Price range  Prediction model
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
df = pd.read_csv(r"C:\Users\User\Desktop\MobileTrain.csv")
#print(df.head())
df2= df[['battery_power','int_memory', 'mobile_wt', 'n_cores', 'px_height',
       'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time','price_range']]

X=df2.drop(["price_range"] , axis=1)
y=df2["price_range"] 
print(df2.info())
print("Feature data dimension: ", X.shape) 
from sklearn.preprocessing import StandardScaler

scaler= StandardScaler()
X=scaler.fit_transform(X)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.linear_model import LogisticRegression
logit_model= LogisticRegression()
   
print(X_train.shape, y_train.shape)
#Fitting the model
m=logit_model.fit(X_train,y_train)
#Saving the model to disk
pickle.dump(logit_model,open('model.pkl','wb') )



