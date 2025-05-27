import streamlit as st
import json

# Load data
@st.cache_data
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

data = load_json("lda_results.json")

# Initialize session state
if "selected_video" not in st.session_state:
    st.session_state.selected_video = list(data["videos"].keys())[0]
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "metrics"

# Navigation radio buttons instead of tabs
tab = st.radio("🧭 Navigate", ["📈 Model Metrics", "🧠 Topics Explorer", "🎬 Video Explorer"], index=["metrics", "topics", "videos"].index(st.session_state.active_tab))

# ---------------------------------
# 📈 Model Metrics
# ---------------------------------
if tab == "📈 Model Metrics":
    st.session_state.active_tab = "metrics"
    st.header("Model Metrics")
    st.write(f"Coherence (c_v): {data['model_metrics']['coherence_c_v']}")
    st.write(f"Coherence (u_mass): {data['model_metrics']['coherence_u_mass']}")
    st.write(f"Coherence (npmi): {data['model_metrics']['coherence_npmi']}")
    st.write(f"Topic Diversity: {data['model_metrics']['topic_diversity']}")
    st.write(f"Jaccard Similarity: {data['model_metrics']['jaccard_similarity']}")

    st.subheader("📚 All Topics and Top Words")
    for topic_id, topic_info in sorted(data["topics"].items(), key=lambda x: int(x[0])):
        with st.expander(f"Topic {topic_id}"):
            st.markdown(", ".join(topic_info["top_words"]))

# ---------------------------------
# 🧠 Topics Explorer
# ---------------------------------
elif tab == "🧠 Topics Explorer":
    st.session_state.active_tab = "topics"
    st.header("Topics Explorer")

    selected_topic = st.selectbox("Select Topic", options=list(data["topics"].keys()))
    st.write(f"Top words: {', '.join(data['topics'][selected_topic]['top_words'])}")

    videos = data["topics"][selected_topic]["videos"]
    st.subheader(f"🎥 Videos in Topic {selected_topic} ({len(videos)} total)")

    # Pagination
    page_size = 10
    total_pages = (len(videos) - 1) // page_size + 1
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paged_videos = videos[start_idx:end_idx]

    for vid in paged_videos:
        title = data["videos"][vid]["title"]
        if st.button(f"📽️ {title}", key=f"vid_button_{vid}"):
            st.session_state.selected_video = vid
            st.session_state.active_tab = "videos"
            st.rerun()  # trigger tab switch

# ---------------------------------
# 🎬 Video Explorer
# ---------------------------------
elif tab == "🎬 Video Explorer":
    st.session_state.active_tab = "videos"
    st.header("Video Explorer")

    # Prepare mapping from video_id to title
    video_ids = list(data["videos"].keys())
    video_titles = [data["videos"][vid]["title"] for vid in video_ids]

    # Find current selected video index to set as default
    selected_index = video_ids.index(st.session_state.selected_video)

    # Selectbox shows titles, returns title selected
    selected_title = st.selectbox(
        "Select Video",
        options=video_titles,
        index=selected_index
    )

    # Map selected title back to video_id
    selected_video = video_ids[video_titles.index(selected_title)]
    st.session_state.selected_video = selected_video

    video_data = data["videos"][selected_video]

    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Video Details")
        st.write(f"**Title:** {video_data['title']}")
        st.write("**Assigned Topics:**", ", ".join(map(str, video_data["topics"])) if video_data["topics"] else "No topics assigned")
        
    with col2:
        st.subheader("Topic Distribution")
        # Sort topics by probability in descending order
        sorted_topics = sorted(
            video_data["topic_probabilities"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Create a bar chart of top topics
        top_n = 10  # Show top 10 topics
        top_topics = sorted_topics[:top_n]
        chart_data = {
            "Topic": [f"Topic {t[0]}" for t in top_topics],
            "Probability": [t[1] for t in top_topics]
        }
        st.bar_chart(chart_data, x="Topic", y="Probability", use_container_width=True)

    # Detailed topic probabilities section
    st.subheader("Detailed Topic Probabilities")
    
    # Create tabs for different views of the probabilities
    prob_tab1, prob_tab2 = st.tabs(["All Topics", "Assigned Topics"])
    
    with prob_tab1:
        # Show all topics in a table sorted by probability
        st.write("All topics sorted by probability:")
        all_topics = []
        for topic_id, prob in sorted_topics:
            topic_words = ", ".join(data["topics"].get(topic_id, {}).get("top_words", ["N/A"]))
            all_topics.append({
                "Topic": topic_id,
                "Probability": prob,
                "Top Words": topic_words,
                "Assigned": "✅" if int(topic_id) in video_data["topics"] else ""
            })
        
        st.dataframe(
            all_topics,
            column_config={
                "Probability": st.column_config.ProgressColumn(
                    "Probability",
                    format="%.4f",
                    min_value=0,
                    max_value=1.0
                )
            },
            hide_index=True,
            use_container_width=True
        )
    
    with prob_tab2:
        # Show only assigned topics with more details
        if not video_data["topics"]:
            st.write("No topics assigned to this video")
        else:
            assigned_topics = []
            for topic in video_data["topics"]:
                topic_str = str(topic)
                prob = video_data["topic_probabilities"].get(topic_str, "N/A")
                topic_words = ", ".join(data["topics"].get(topic_str, {}).get("top_words", ["N/A"]))
                matching_words = video_data["topic_words_mapping"].get(topic_str, {}).get("words", [])
                
                assigned_topics.append({
                    "Topic": topic,
                    "Probability": prob,
                    "Top Words": topic_words,
                    "Matching Words": ", ".join(matching_words) if matching_words else "None"
                })
            
            st.dataframe(
                assigned_topics,
                column_config={
                    "Probability": st.column_config.ProgressColumn(
                        "Probability",
                        format="%.4f",
                        min_value=0,
                        max_value=1.0
                    )
                },
                hide_index=True,
                use_container_width=True
            )

    # Preprocessed text and ngrams sections remain the same
    st.subheader("Preprocessed Text")
    st.text(video_data["preprocessed_text"])

    st.subheader("N-grams")
    st.text(video_data["ngrams"])
