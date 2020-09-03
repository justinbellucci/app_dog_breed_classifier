# PROGRAMMER: Justin Bellucci 
# DATE CREATED: 09_03_2020                                  
# REVISED DATE: 

# Import python modules
import streamlit as st
import numpy as np
import pandas as pd

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title('Dog Breed Classifier App')
img = st.file_uploader('Upload Image', type='jpg')
st.text('{}'.format(type(img)))

if not st.button('Reset'):
    st.image(img, output_format='JPEG', use_column_width=True)
else:
    st.markdown('## Want to try again?')