import streamlit as st
import pandas as pd
import os

# Set page configuration
st.set_page_config(page_title="Pinoybaiting Dashboard", layout="wide")

import bertopic_module  
import lda_module
import hlta_module


# Sidebar navigation
st.sidebar.title("Pinoybaiting Dashboard")
selection = st.sidebar.radio("Go to", ["Summary", "LDA", "BERTopic", "HLTM"])

st.title("Pinoybaiting Dashboard")

if selection == "Summary":
    st.header("📌 Summary")
    st.write("This section will contain summary statistics, metrics, and data previews.")

# ---------------------------------
#  LDA
# ---------------------------------

elif selection == "LDA":
    lda_module.show_lda_section()

# ---------------------------------
#  BERTopic
# ---------------------------------

elif selection == "BERTopic":
    bertopic_module.show_bertopic_section()

# ---------------------------------
#  HLTM
# ---------------------------------

elif selection == "HLTM":
    st.header("HLTM (Hierarchical Latent Tree Model)")
    st.write("This section is for visualizing HLTM results.")
    hlta_module.show_hlta_section()