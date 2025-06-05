# bertopic_module.py
import streamlit as st
import pandas as pd
import os
import streamlit.components.v1 as components

def show_bertopic_section():
    st.header("BERTopic")

    csv_path = "BERTopic-coherence_scores.csv"
    try:
        scores_df = pd.read_csv(csv_path)
        scores_df["Score"] = scores_df["Score"].round(3)

        data = {
            "model_metrics": {
                metric: float(score) for metric, score in zip(scores_df["Metric"], scores_df["Score"])
            }
        }

        if "bertopic_active_tab" not in st.session_state:
            st.session_state.bertopic_active_tab = "metrics"

        tab = st.radio(
            "Navigate BERTopic Section",
            ["Model Metrics", "Topics Summary", "Video Summary"],
            index=["metrics", "topics", "videos"].index(st.session_state.bertopic_active_tab)
        )
        st.markdown("---")

        if tab == "Model Metrics":
            st.session_state.bertopic_active_tab = "metrics"
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Coherence (c_v)", f"{data['model_metrics'].get('c_v', 'N/A'):.2f}")
                st.metric("Coherence (u_mass)", f"{data['model_metrics'].get('u_mass', 'N/A'):.2f}")
            with col2:
                st.metric("Coherence (c_npmi)", f"{data['model_metrics'].get('c_npmi', 'N/A'):.2f}")

            st.markdown("---")
            st.subheader("📊 Topic Frequency Bar Chart")

            # Path to your previously saved HTML barchart
            html_file_path = "topic_barchart.html"

            if os.path.exists(html_file_path):
                with open(html_file_path, "r", encoding="utf-8") as f:
                    html_data = f.read()
                # Embed the full html content with height
                components.html(html_data, height=700, scrolling=True)
            else:
                st.error(f"HTML file not found at: {os.path.abspath(html_file_path)}")

        elif tab == "Topics Summary":
            st.session_state.bertopic_active_tab = "topics"
            st.header("Topics Summary")
            st.write("Add topics summary visualizations here.")

        elif tab == "Video Summary":
            st.session_state.bertopic_active_tab = "videos"
            st.header("Video Summary")
            st.write("Add video summary visualizations here.")

    except FileNotFoundError:
        st.error(f"Coherence score file not found at: {os.path.abspath(csv_path)}")
