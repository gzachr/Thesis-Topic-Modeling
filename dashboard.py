import streamlit as st
import pandas as pd
import os

import bertopic_module  

# Set page configuration
st.set_page_config(page_title="Pinoybaiting Dashboard", layout="wide")

# Sidebar navigation
st.sidebar.title("Pinoybaiting Dashboard")
selection = st.sidebar.radio("Go to", ["Summary", "LDA", "BERTopic", "HLTM"])

st.title("🇵🇭 Pinoybaiting Dashboard")

if selection == "Summary":
    st.header("📌 Summary")
    st.write("This section will contain summary statistics, metrics, and data previews.")

# ---------------------------------
#  LDA
# ---------------------------------

elif selection == "LDA":
    st.header("LDA (Latent Dirichlet Allocation)")
    st.write("This section is for visualizing topics generated using LDA.")
    # Add 

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
    # Add

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit")
