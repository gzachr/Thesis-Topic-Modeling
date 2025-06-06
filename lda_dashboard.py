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

# Navigation radio buttons
tab = st.radio("Navigate", 
              ["Model Metrics", "Topics Summary", "Topic Group Summary", "Channel Summary", "Video Summary"],
              index=["metrics", "topics", "topic_groups", "channel", "videos"].index(st.session_state.active_tab))

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
    # Extract YouTube video ID (remove .txt)
    video_id = selected_video.replace(".txt", "")
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    st.markdown(f"[Watch on YouTube]({youtube_url})", unsafe_allow_html=True)


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
        
        # Check if subtopic assignments exist
        if "subtopic_assignments" in data:
            assigned_subtopics = []
            
            # Iterate through all topic groups
            for group_str, group_data in data["subtopic_assignments"].items():
                # Skip unassigned videos section
                if group_str == "unassigned_videos_by_group":
                    continue
                    
                # Check if this group has categories
                if not isinstance(group_data, dict):
                    continue
                    
                for category, subtopics in group_data.items():
                    for subtopic, subtopic_data in subtopics.items():
                        # Check if current video is in this subtopic
                        video_ids = [v["video_id"] for v in subtopic_data["videos"]]
                        if selected_video in video_ids:
                            # Find the video details to get matched words
                            video_details = next(
                                (v for v in subtopic_data["videos"] if v["video_id"] == selected_video), 
                                None
                            )
                            
                            if video_details:
                                # Collect all matched words across topics
                                matched_words = set()
                                for topic_match in video_details.get("topics", []):
                                    matched_words.update(topic_match.get("matched_words", []))
                                
                                assigned_subtopics.append({
                                    "Group": group_str,
                                    "Category": category,
                                    "Subtopic": subtopic,
                                    "Matched Keywords": ", ".join(sorted(matched_words)) if matched_words else "None"
                                })
            
            if assigned_subtopics:
                st.dataframe(
                    assigned_subtopics,
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.write("No subtopics assigned to this video")
        else:
            st.write("No subtopic assignments available")


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
#  Topic Group Summary
# ---------------------------------
elif tab == "Topic Group Summary":
    st.session_state.active_tab = "topic_groups"
    st.header("Topic Group Summary")
    
    # Get available topic groups
    if "merged_topic_groups" not in data:
        st.warning("Topic group data not available")
        st.stop()
    
    topic_groups = list(data["merged_topic_groups"].keys())
    selected_group = st.selectbox("Select Topic Group", options=sorted(topic_groups, key=int))
    group_data = data["merged_topic_groups"][selected_group]
    
   # Display group overview
    st.subheader(f"Group {selected_group} Overview")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Topics in Group", ", ".join(map(str, group_data["topics"])))
        st.metric("Total Videos", len(group_data["videos"]))
    with col2:
        st.empty()  # Leave this empty for visual balance

    # Top keywords section in a container below
    with st.container():
        st.markdown("### Top Keywords")
        top_keywords = group_data["keywords"][:50] 

        keyword_text = ", ".join(kw["word"] for kw in top_keywords)
        st.markdown(f"<div style='flex-wrap:wrap; line-height:1.6em'>{keyword_text}</div>", unsafe_allow_html=True)

    
    # Display subtopic assignments for this group
    st.subheader("Subtopic Assignments")

    # Check if subtopic assignments exist for this group
    if "subtopic_assignments" not in data or selected_group not in data["subtopic_assignments"]:
        st.warning("No subtopic assignments for this group")
    else:
        group_assignments = data["subtopic_assignments"][selected_group]

        # Create tabs for each category
        category_tabs = st.tabs(list(group_assignments.keys()))

        for idx, category in enumerate(group_assignments.keys()):
            with category_tabs[idx]:
                st.subheader(f"{category} Subtopics")

                for subtopic, subtopic_data in group_assignments[category].items():
                    with st.expander(f"{subtopic} ({subtopic_data['video_count']} videos)"):
                        # Display subtopic details
                        st.write(f"**Keywords:** {', '.join(subtopic_data['keywords'])}")
                        st.write(f"**Video Coverage:** {subtopic_data['video_ratio']:.1%} of group videos")

                        st.write("**Videos in this subtopic:**")

                        videos = subtopic_data["videos"]

                        # Sort videos by total matched keywords (across all topics)
                        videos_sorted = sorted(
                            videos,
                            key=lambda v: sum(len(topic.get("matched_words", [])) for topic in v.get("topics", [])),
                            reverse=True
                        )


                        videos_per_page = 5
                        total_pages = (len(videos_sorted) - 1) // videos_per_page + 1

                        page = st.number_input(
                            label="Page",
                            min_value=1,
                            max_value=total_pages,
                            value=1,
                            step=1,
                            key=f"page_selector_{subtopic}"
                        )

                        start_idx = (page - 1) * videos_per_page
                        end_idx = start_idx + videos_per_page
                        paged_videos = videos_sorted[start_idx:end_idx]

                        for video in paged_videos:
                            if st.button(f"📽️ {video['title']}", key=f"subtopic_{subtopic}_{video['video_id']}"):
                                st.session_state.selected_video = video["video_id"]
                                st.session_state.active_tab = "videos"
                                st.rerun()


    # Show unassigned videos if available
    if "unassigned_videos_by_group" in data and selected_group in data["unassigned_videos_by_group"]:
        unassigned = data["unassigned_videos_by_group"][selected_group]
        with st.expander(f"Unassigned Videos ({unassigned['count']})"):
            st.write(f"**Ratio:** {unassigned['ratio']:.1%} of group videos")
            
            # Show sample of unassigned videos
            for video in unassigned["videos"]:
                if st.button(f"📽️ {video['title']}", key=f"unassigned_{video['video_id']}"):
                    st.session_state.selected_video = video["video_id"]
                    st.session_state.active_tab = "videos"
                    st.rerun()

elif tab == "Channel Summary":
    st.session_state.active_tab = "channel"
    st.header("Channel Summary Across Topic Groups")

    if "channel_summary" not in data or not data["channel_summary"]:
        st.warning("No channel summary data found.")
        st.stop()

    channel_summary = data["channel_summary"]
    all_video_ids = list(data["videos"].keys())

    # Create a dict for quick title lookup
    video_titles = {vid: data["videos"][vid]["title"] for vid in all_video_ids}

    sorted_channels = sorted(
        channel_summary.items(),
        key=lambda x: len(x[1]["video_ids"]),
        reverse=True
    )

    ## Pagination for channels ##
    channels_per_page = 10
    total_channel_pages = (len(sorted_channels) - 1) // channels_per_page + 1
    current_channel_page = st.session_state.get("channel_page", 0)

    st.markdown("### 📺 Channels")
    col1, col2 = st.columns([1, 6])
    with col1:
        if st.button("⬅️ Prev", disabled=current_channel_page == 0):
            st.session_state.channel_page = max(0, current_channel_page - 1)
            st.rerun()
    with col2:
        if st.button("➡️ Next", disabled=current_channel_page >= total_channel_pages - 1):
            st.session_state.channel_page = min(total_channel_pages - 1, current_channel_page + 1)
            st.rerun()

    start_idx = current_channel_page * channels_per_page
    end_idx = start_idx + channels_per_page
    paginated_channels = sorted_channels[start_idx:end_idx]

    for channel_id, info in paginated_channels:
        channel_title = info["channel_title"]
        video_ids = info["video_ids"]
        group_map = info["group_category_map"]

        with st.expander(f"📺 {channel_title} ({len(video_ids)} videos)"):
            channel_url = f"https://www.youtube.com/channel/{channel_id}"
            st.markdown(f"[🌐 Visit Channel]({channel_url})", unsafe_allow_html=True)
            group_tabs = st.tabs([f"🔹 Topic Group {group_index}" for group_index in sorted(group_map, key=int)])

            for tab, group_index in zip(group_tabs, sorted(group_map, key=int)):
                with tab:
                    for category in sorted(group_map[group_index]):
                        st.markdown(f"**📂 {category}**")
                        for subtopic in sorted(group_map[group_index][category]):
                            vids = group_map[group_index][category][subtopic]
                            st.markdown(f"- **{subtopic}** ({len(vids)} video{'s' if len(vids) > 1 else ''})")

                            # Pagination state
                            video_page_key = f"video_page_{channel_id}_{group_index}_{category}_{subtopic}"
                            if video_page_key not in st.session_state:
                                st.session_state[video_page_key] = 0

                            videos_per_page = 10
                            total_video_pages = (len(vids) - 1) // videos_per_page + 1
                            current_video_page = st.session_state[video_page_key]
                            video_start = current_video_page * videos_per_page
                            video_end = video_start + videos_per_page

                            paginated_vids = vids[video_start:video_end]

                            for i, vid in enumerate(paginated_vids):
                                title = data["videos"].get(vid, {}).get("title", "Unknown Title")
                                unique_key = f"channel_summary_vid_{channel_id}_{group_index}_{category}_{subtopic}_{i}_{vid}"
                                if st.button(f"▶ {title}", key=unique_key):
                                    st.session_state.selected_video = vid
                                    st.session_state.active_tab = "videos"
                                    st.rerun()

                            # Pagination controls
                            vcol1, vcol2 = st.columns([1, 1])
                            with vcol1:
                                if st.button("⬅️ Prev Videos", key=f"{video_page_key}_prev", disabled=current_video_page == 0):
                                    st.session_state[video_page_key] = max(0, current_video_page - 1)
                                    st.rerun()
                            with vcol2:
                                if st.button("➡️ Next Videos", key=f"{video_page_key}_next", disabled=current_video_page >= total_video_pages - 1):
                                    st.session_state[video_page_key] = min(total_video_pages - 1, current_video_page + 1)
                                    st.rerun()
