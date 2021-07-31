import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sklearn
from PIL import Image
import pickle

with open('xgb_model.pkl','rb') as f:
    xgb_model = pickle.load(f)
 
selectbox1 = st.sidebar.radio("Menu",("Introduction", "Description of Inputs", "Empirical Drift Capacity Equation", "XGBoost Model"))
st.sidebar.success("Read the Full Paper at _link_")
st.sidebar.info("Visit the Burton Research Group at https://www.henryburtonjr.com/")
 
if selectbox1 == "Introduction":
    st.title("Explainable Machine Learning Model For Predicting The Drift Capacity Of Reinforced Concrete Walls")
    st.write("This app estimates the drift capacity, which is the drift at which lateral strength degrades by 20% from the peak strength, of reinforced concrete walls with special boundary elements. It is based on the Extreme Gradient Boosting (XGBoost) machine learning algorithm and using 3 input parameters.")
    image1 = Image.open('homepage.png')
    st.image(image1)
 
if selectbox1 == "XGBoost Model":
    st.title('Machine Learning Model for Drift Capacity Prediction')
    st.warning('Please input your desired parameters.')
 
    st.markdown("1. Slenderness Parameter ($\lambda_{b}=\dfrac{l_{w}c}{b^2}$)")
    x_1 = st.number_input("l_w (in)", 0.000000001)
    x_2 = st.number_input("c (in)", 0.000000001)
    x_3 = st.number_input("b (in)", 0.000000001)
    x_4 = (x_1*x_2)/(x_3**2)
    df1 = pd.DataFrame(columns=['Slenderness Parameter'])
    df1['Slenderness Parameter'] = [x_4]
    st.write(df1)
    
    st.markdown("2. Shear Stress Demand ($\dfrac{v_{max}}{\sqrt{f'_{c}(psi)}}$)")
    x_5 = st.number_input("enter value", 0.000000001)
    
    st.markdown("3. Configuration of Boundary Transverse Reinforcement")
    
    selectbox = st.selectbox("select", ("Overlapping Hoops",
    "Combination of a Perimeter Hoop and Crossties with 90-135 Degrees Hooks",
    "Combination of a Perimeter Hoop and Crossties with 135-135 Degrees Hooks",
    "Combination of a Perimeter Hoop and Crossties with Headed Bars",
    "Single Hoop without Intermediate Legs of Crossties"))
    if selectbox == "Overlapping Hoops":
        x_6 = 2.6482
    if selectbox == "Combination of a Perimeter Hoop and Crossties with 90-135 Degrees Hooks":
        x_6 = 2.8069
    if selectbox == "Combination of a Perimeter Hoop and Crossties with 135-135 Degrees Hooks":
        x_6 = 2.7916
    if selectbox == "Combination of a Perimeter Hoop and Crossties with Headed Bars":
        x_6 = 3.7967
    if selectbox == "Single Hoop without Intermediate Legs of Crossties":
        x_6 = 2.8750
    
    input = np.array([x_4,x_5,x_6])
    input = np.array(input).reshape((1,-1))
    predicted = xgb_model.predict(input)
    predicted = predicted[0]
 
    st.warning('The drift capacity prediction can be found below.')
    df = pd.DataFrame(columns=['Drift Capacity (%)'])
    df['Drift Capacity (%)'] = [predicted]
    st.write(df)
 
if selectbox1 == "Empirical Drift Capacity Equation":
    st.title('Empirical Drift Capacity Equation (ACI 318-19)')
    st.markdown("The equation adopted in ACI 318-19 was developed by Abdullah and Wallace (2019) to predict the mean drift capacity ($\dfrac{\delta_{x}}{h_{w}}$) of walls with SBEs.")
    st.markdown("$\dfrac{\delta_{x}}{h_{w}}$" + "(_%_)" + "$=3.85-\dfrac{\lambda_{b}}{α} - \dfrac{v_{max}}{10\sqrt{f'_{c}(psi)}}$")
    st.markdown("where α=60 for overlapping hoops and α=45 for a combination of a single perimeter hoop with supplemental crossties.")
    st.warning('Please input your desired parameters.')
 
    st.markdown("1. Slenderness Parameter ($\lambda_{b}=\dfrac{l_{w}c}{b^2}$)")
    x_7 = st.number_input("l_w (in) ", 0.000000001)
    x_8 = st.number_input("c (in) ", 0.000000001)
    x_9 = st.number_input("b (in) ", 0.000000001)
    x_10 = (x_7*x_8)/(x_9**2)
    df3 = pd.DataFrame(columns=['Slenderness Parameter'])
    df3['Slenderness Parameter'] = [x_10]
    st.write(df3)
    
    st.markdown("2. Shear Stress Demand ($\dfrac{v_{max}}{\sqrt{f'_{c}(psi)}}$)")
    x_11 = st.number_input("enter value  ", 0.000000001)
    
    st.markdown("3. Configuration of Boundary Transverse Reinforcement")
    x_12 = 60
    selectbox6 = st.selectbox("select", ("Overlapping Hoop", "Single Perimeter Hoop with Crossties"))
    if selectbox6 == "Overlapping Hoop":
        x_12 = 60
    if selectbox6 == "Single Perimeter Hoop with Crossties":
        x_12 = 45
 
    model = 3.85 - x_10/x_12 - 0.1*x_11
 
    st.warning('The drift capacity prediction can be found below.')
    df4 = pd.DataFrame(columns=['Drift Capacity (%)'])
    df4['Drift Capacity (%)'] = [model]
    st.write(df4)
 
if selectbox1 == "Description of Inputs":
    st.title("Description of Inputs")
    st.write("The three most influential parameters to drift capacity were selected as inputs to develop the predictive machine learning model.")
    selectbox3 = st.radio("select",("Slenderness Parameter", "Shear Stress Demand", "Configuration of the Boundary Transverse Reinforcement"))
    
    if selectbox3 == "Slenderness Parameter":
        st.subheader("Slenderness Parameter ($\lambda_{b}$)")
        st.write("A parameter that accounts for the slenderness of the cross section and the compression zone.")
        st.markdown("$\lambda_{b}=\dfrac{l_{w}c}{b^2}$")
        st.write("where $l_{w}$ is the wall length, $c$ is the depth of neutral axis computed at a concrete compressive strain of 0.003, and $b$ is the width of flexural compression zone.")
        image2 = Image.open('wall.png')
        st.image(image2)
    
    if selectbox3 == "Shear Stress Demand":
        st.subheader("Shear Stress Demand ($\dfrac{v_{max}}{\sqrt{f'_{c}(psi)}}$)")
        st.write("The maximum experimental shear stress normalized by the square root of the concrete compressive strength.")
    
    if selectbox3 == "Configuration of the Boundary Transverse Reinforcement":
        st.subheader("Configuration of Boundary Transverse Reinforcement")
        
        st.write("1. Overlapping Hoops")
        image3 = Image.open('OH.png')
        st.image(image3)
        
        st.write("2. Combination of a Perimeter Hoop and Crossties with 90-135 Degrees Hooks")
        image4 = Image.open('PH-90-135.png')
        st.image(image4)
        
        st.write("3. Combination of a Perimeter Hoop and Crossties with 135-135 Degrees Hooks")
        image5 = Image.open('PH-135-135.png')
        st.image(image5)
        
        st.write("4. Combination of a Perimeter Hoop and Crossties with Headed Bars")
        image6 = Image.open('PH-HB.png')
        st.image(image6)
        
        st.write("5. Single Hoop without Intermediate Legs of Crossties")
        image7 = Image.open('SH.png')
        st.image(image7)

