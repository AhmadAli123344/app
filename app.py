import pickle
import streamlit as st
#from sklearn.compose import ColumnTransformer
#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import StandardScaler, OneHotEncoder
#from sklearn.linear_model import LogisticRegression
import pandas as pd

# Load the trained model from the pickle file
with open('D:/model_heartdisease.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

sex=('M','F')
chestPainType=('ASY','NAP','ATA','TA')
restingECG=('Normal','LVH','ST')
st_Slope=('Flat','Up','Down')

def main():
    st.title('Heart Disease Prediction')
    
    Age=st.text_input('Enter Your Age')
    Sex=st.selectbox('Sex',sex)
    ChestPainType=st.selectbox('ChestPainType',chestPainType)
    RestingBP=st.text_input('Enter RestingBP')
    Cholesterol=st.text_input('Enter Cholesterol level')
    FastingBS=st.text_input('Enter FastingBS')
    RestingECG=st.selectbox('RestingECG',restingECG)
    MaxHR=st.text_input('Enter MaxHR')
    Oldpeak=st.text_input('Enter Oldpeak')
    ST_Slope=st.selectbox('ST_Slope',st_Slope)
    
    ok=st.button('Predict Disease')
    if ok:
        # Create a new input data array
        data = {
            'Age': [Age],
            'Sex': [Sex],
            'ChestPainType': [ChestPainType],
            'RestingBP': [RestingBP],
            'Cholesterol': [Cholesterol],
            'FastingBS': [FastingBS],
            'RestingECG': [RestingECG],
            'MaxHR': [MaxHR],
            'Oldpeak': [Oldpeak],
            'ST_Slope': [ST_Slope]
        }
        X=pd.DataFrame(data)
        
        # Perform the same preprocessing as during training
        preprocess = loaded_model.named_steps['preprocess']
        X_preprocessed = preprocess.transform(X)
        
        # Use the loaded model to make predictions
        y_pred = loaded_model.named_steps['predict'].predict(X_preprocessed)
        if y_pred[0]==0:
            st.subheader('Patient has No Heart Disease')
        else:
            st.subheader('Patient has Heart Disease')
        #st.subheader(f'The Patient has {y_pred[0]}')

if __name__ == "__main__":
    main()
