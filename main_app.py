# PROGRAMMER: Justin Bellucci 
# DATE CREATED: 09_03_2020                                  
# REVISED DATE: 

# Import python modules
import streamlit as st
import numpy as np
import pandas as pd
import time

def main():

    st.set_option('deprecation.showfileUploaderEncoding', False)

    reset = False
    st.title('Dog Breed Classifier App')
    uploaded_file = st.file_uploader('Upload Image', type='jpg')
    # st.text('{}'.format(type(uploaded_file)))
    
    if uploaded_file is not None:
        side_title = add_sidebar()
        st.text(type(side_title))
        st.sidebar.image(uploaded_file, output_format='JPEG', use_column_width=True)
    
    if st.button('Reset'):
        reset_page()

def add_sidebar():
    side_title = st.sidebar.title('Doggy')
    return side_title

def reset_page():
    placeholder = st.empty()
    return 
if __name__ == "__main__":
    main()