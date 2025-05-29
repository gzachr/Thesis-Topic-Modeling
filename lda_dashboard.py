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
tab = st.radio("Navigate", 
              ["Model Metrics", "Topics Summary", "Subtopics Summary", "Video Summary"],
              index=["metrics", "topics", "subtopics", "videos"].index(st.session_state.active_tab))

# ---------------------------------
#  Model Metrics
# ---------------------------------
if tab == "Model Metrics":
    st.session_state.active_tab = "metrics"
    st.header("Model Metrics")

    # Display model coherence and diversity metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Coherence (c_v)", f"{data['model_metrics']['coherence_c_v']:.3f}")
        st.metric("Coherence (u_mass)", f"{data['model_metrics']['coherence_u_mass']:.3f}")
    with col2:
        st.metric("Coherence (npmi)", f"{data['model_metrics']['coherence_npmi']:.3f}")
        st.metric("Topic Diversity", f"{data['model_metrics']['topic_diversity']:.3f}")

    # -------------------------------
    #  All Topics and Top Words
    # -------------------------------
    st.subheader(" All Topics and Top Words")

    for topic_id, topic_info in sorted(data["topics"].items(), key=lambda x: int(x[0])):
        with st.expander(f"Topic {topic_id}"):
            st.markdown(", ".join(topic_info["top_words"]))

# ---------------------------------
#  Topics Summary
# ---------------------------------
elif tab == "Topics Summary":
    st.session_state.active_tab = "topics"
    st.header("Topics Explorer")

    selected_topic = st.selectbox("Select Topic", options=list(data["topics"].keys()))
    topic_info = data["topics"][selected_topic]

    st.subheader(f"Topic {selected_topic}")
    st.write(f"**Top words:** {', '.join(topic_info['top_words'])}")
    
    # Tab layout for videos and subtopics
    tab1, tab2 = st.tabs(["🎥 Videos", "🏷️ Subtopics"])
    
    with tab1:
        # Existing videos display with pagination
        videos = topic_info["videos"]
        st.subheader(f"Videos in Topic {selected_topic} ({len(videos)} total)")
        
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
                st.rerun()
    
    with tab2:
        if "topic_to_subtopics" in data and selected_topic in data["topic_to_subtopics"]:
            st.subheader("Subtopics for this Topic")
            
            for subtopic in data["topic_to_subtopics"][selected_topic]:
                category = data["subtopic_categories"].get(subtopic, "Unknown")
                
                with st.expander(f"{subtopic} ({category})"):
                    # Show subtopic details
                    if "subtopic_keywords" in data and subtopic in data["subtopic_keywords"]:
                        st.write("**Keywords:**")
                        for topic_id, keywords in data["subtopic_keywords"][subtopic].items():
                            if topic_id == selected_topic:
                                st.write(f"- {', '.join(keywords)}")
                    
                    # Show videos in this subtopic
                    if "subtopic_to_videos" in data and subtopic in data["subtopic_to_videos"]:
                        subtopic_videos = data["subtopic_to_videos"][subtopic]
                        st.write(f"**Videos in this subtopic:** {len(subtopic_videos)}")
                        
                        for vid in list(subtopic_videos)[:5]:  # Show first 5
                            title = data["videos"][vid]["title"]
                            if st.button(f"▶️ {title}", key=f"sub_vid_{subtopic}_{vid}"):
                                st.session_state.selected_video = vid
                                st.session_state.active_tab = "videos"
                                st.rerun()
                        if len(subtopic_videos) > 5:
                            st.write(f"... and {len(subtopic_videos)-5} more")
        else:
            st.write("No subtopics defined for this topic")

# ---------------------------------
#  Video Summary
# ---------------------------------
elif tab == "Video Summary":
    st.session_state.active_tab = "videos"
    st.header("Video Summary")

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

    st.subheader("Video Details")
    st.write("**Assigned Topics:**", ", ".join(map(str, video_data["topics"])) if video_data["topics"] else "No topics assigned")
    
    # Add subtopics information
    if "video_to_subtopics" in data and selected_video in data["video_to_subtopics"]:
        st.write("**Assigned Subtopics:**")
        for subtopic in data["video_to_subtopics"][selected_video]:
            category = data["subtopic_categories"].get(subtopic, "Unknown")
            st.write(f"- {subtopic} ({category})")

    sorted_topics = sorted(
            video_data["topic_probabilities"].items(),
            key=lambda x: x[1],
            reverse=True
        )    

    top_n = 10  # Show top 10 topics
    top_topics = sorted_topics[:top_n]

    prob_tab1, prob_tab2, prob_tab3 = st.tabs(["All Topics", "Assigned Topics", "Assigned Subtopics"])

    with prob_tab1:
        st.subheader("All Topics")
        # Show all topics in a table sorted by probability
        all_topics = []
        top_n = 10  # Show top 10 topics
        top_topics = sorted_topics[:top_n]
        for topic_id, prob in sorted_topics:
            topic_words = ", ".join(data["topics"].get(topic_id, {}).get("top_words", ["N/A"]))
            all_topics.append({
                "Topic": topic_id,
                "Probability": prob,
                "Top Words": topic_words,
                "Assigned": "✅" if topic_id in video_data["topics"] else ""
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
        st.subheader("Assigned Topics")
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
    with prob_tab3:
        st.subheader("Assigned Subtopics")
        # Show assigned subtopics with their categories and matched keywords
        if "video_to_subtopics" in data and selected_video in data["video_to_subtopics"]:
            assigned_subtopics = []
            for subtopic in data["video_to_subtopics"][selected_video]:
                category = data["subtopic_categories"].get(subtopic, "Unknown")
                matched_keywords = set()

                # Get topics for this subtopic
                topic_ids = data["subtopic_to_topics"].get(subtopic, [])

                for topic_id in topic_ids:
                    topic_str = str(topic_id)

                    # Get subtopic keywords for this topic
                    subtopic_kw_set = set(data["subtopic_keywords"].get(subtopic, {}).get(topic_str, []))

                    # Get matching words from the video's topic-word mapping
                    topic_match_info = data["videos"][selected_video].get("topic_words_mapping", {}).get(topic_str, {})
                    matched_words = set(topic_match_info.get("words", []))

                    # Find intersection
                    matched_keywords.update(subtopic_kw_set & matched_words)

                assigned_subtopics.append({
                    "Subtopic": subtopic,
                    "Category": category,
                    "Matched Keywords": ", ".join(sorted(matched_keywords)) if matched_keywords else "None"
                })

            st.dataframe(
                assigned_subtopics,
                hide_index=True,
                use_container_width=True
            )
        else:
            st.write("No subtopics assigned to this video")


    st.subheader("Original Text")
    st.text_area(
        label="",
        value=video_data["original_text"],
        height=300,
        key="original_text_display"
    )

    st.subheader("Preprocessed Text")
    st.text_area(
        label="",
        value=video_data["preprocessed_text"],
        height=300,
        key="preprocessed_text_display"
    )


    st.subheader("N-grams")
    st.text(video_data["ngrams"])

# ---------------------------------
#  Subtopics Summary
# ---------------------------------
elif tab == "Subtopics Summary":
    st.session_state.active_tab = "subtopics"
    st.header("Subtopics Summary")
    
    st.subheader("Subtopic and Category Metrics")

    category_stats = data["subtopic_metrics"]["category_stats"]
    subtopic_stats = data["subtopic_metrics"]["subtopic_stats"]

    # Display per category
    for category, stats in category_stats.items():
        with st.expander(f" {category} — {stats['subtopic_count']} subtopics"):
            st.markdown(f"- **Unique Videos :** {stats['unique_video_count']}")
            st.markdown(f"- **Total Video Matches:** {stats['video_count']}")
            st.markdown(f"- **Avg. Videos per Subtopic:** {stats['avg_videos_per_subtopic']:.2f}")

            # Subtopics under this category
            st.markdown("#### Subtopics:")
            for subtopic, sub_stats in subtopic_stats.items():
                if sub_stats["category"] == category:
                    st.markdown(
                        f"- **{subtopic}** — {sub_stats['video_count']} videos, "
                        f"{sub_stats['matched_keywords']} matched keywords"
                    )

    if "category_to_subtopics" not in data:
        st.warning("Subtopics data not available")
        st.stop()
    
    # Category selection
    selected_category = st.selectbox("Select Category", options=list(data["category_to_subtopics"].keys()))
    
    # Display subtopics in this category
    subtopics = data["category_to_subtopics"][selected_category]
    selected_subtopic = st.selectbox("Select Subtopic", options=subtopics)
    
    # Subtopic details
    st.subheader(f"Subtopic: {selected_subtopic}")
    
    # Keywords for this subtopic
    if "subtopic_keywords" in data and selected_subtopic in data["subtopic_keywords"]:
        st.write("**Keywords by Topic:**")
        for topic_id, keywords in data["subtopic_keywords"][selected_subtopic].items():
            st.write(f"- Topic {topic_id}: {', '.join(keywords)}")
    
    if "video_to_subtopics" in data:
        # Get all videos assigned to this subtopic
        videos_in_subtopic = [
            vid for vid, subtopics in data["video_to_subtopics"].items() 
            if selected_subtopic in subtopics
        ]
        
        st.subheader(f"Videos in this Subtopic ({len(videos_in_subtopic)})")
        
        if videos_in_subtopic:
            # Pagination
            page_size = 10
            total_pages = (len(videos_in_subtopic) - 1) // page_size + 1
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1, key="subtopic_page")
            
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paged_videos = videos_in_subtopic[start_idx:end_idx]
            
            for vid in paged_videos:
                title = data["videos"][vid]["title"]
                if st.button(f"📽️ {title}", key=f"sub_vid_button_{vid}"):
                    st.session_state.selected_video = vid
                    st.session_state.active_tab = "videos"
                    st.rerun()
        else:
            st.write("No videos assigned to this subtopic")
    else:
        st.write("No video-to-subtopic mapping available")