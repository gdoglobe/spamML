import streamlit as st
from streamlit_option_menu import option_menu

# Main menu Display
with st.expander("MENU"):
     menu = option_menu(None, ["Home Interface", "Unsupervised Interface", "Supervised Interface","Semi-Supervised Interface","Spam Or Not ?"], 
    icons=['house', "envelope", "envelope-fill","envelope-check","question-circle", 'graph-up'], 
    menu_icon="cast", default_index=0, orientation="vertical")

# All pages redirection
if menu == 'Home Interface':
    exec(open("home_page.py").read())    

elif menu == 'Unsupervised Interface':
    exec(open("unsupervised_interface.py").read())
    

elif menu == 'Supervised Interface':
    exec(open("supervised_interface.py").read())

elif menu == 'Semi-Supervised Interface':
    exec(open("semi_supervised_interface.py").read())

elif menu == 'Spam Or Not ?':
    exec(open("spam_or_not.py").read())

