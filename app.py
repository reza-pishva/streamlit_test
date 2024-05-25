import joblib
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

model = joblib.load('rf_model.joblib')

st.title('Titanic Detection')
st.write("""### We need some information to predict""")

input1 = st.number_input('Pclass')
input2 = st.number_input('Age')
input3 = st.number_input('SibSp')
input4 = st.number_input('Parch')
input5 = st.number_input('Fare')

new_data = pd.DataFrame([{      
       'Pclass':input1, 'Age':input2,'SibSp':input3,'Parch':input4,'Fare':input5 
    }])


prediction = model.predict(new_data)
prediction_text = 'dead' if prediction[0] == 1 else 'alive'

if st.button('prediction'):
   st.write(prediction_text)