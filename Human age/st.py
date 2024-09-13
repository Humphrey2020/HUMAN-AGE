import numpy as np 
import pandas as pd 
import streamlit as st 
import joblib as jb


st.header('Prediction of age') 
model=jb.load('model.py')
scalar=jb.load('scalar.py')

blood_pressure=st.number_input('Blood Pressure (s/d)',max_value=1605,min_value=0,step=1,value=10)
bone_denisty=st.number_input('Bone Density (g/cm²)',max_value=3.5,min_value=0.0,step=0.1,value=1.2)
hear_ability=st.number_input('Hearing Ability (dB)',max_value=94,min_value=0,step=1,value=10) 

input_data=np.array([[blood_pressure,bone_denisty,hear_ability]]) 
scaled_data=scalar.transform(input_data) 
st.write(f"blood_pressure:{scaled_data[0][0]:.2f}")
st.write(f"bone_density:{scaled_data[0][1]:.2f}")
st.write(f"hear_ability: {scaled_data[0][0]:.2f}")

st.button('prediction of age') 
prediction=model.predict(scaled_data)
st.success(f"prediction of the age is {prediction}")