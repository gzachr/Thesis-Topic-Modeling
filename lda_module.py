import streamlit as st
import json

# Load data
@st.cache_data
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

data = load_json("lda_results.json")

def show_lda_section():
    # Initialize session state
    if "selected_video" not in st.session_state:
        st.session_state.selected_video = list(data["videos"].keys())[0]
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Summary"

    # Navigation radio buttons
    tab = st.radio("Navigate", 
                ["Model Summary", "Topics Summary", "Channel Summary", "Video Summary"],
                index=["Summary", "topics", "channel", "videos"].index(st.session_state.active_tab))

    # ---------------------------------
    #  Model Summary
    # ---------------------------------
    if tab == "Model Summary":
        st.session_state.active_tab = "Summary"
        st.header("Model Summary")

        # Display model coherence and diversity metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Coherence (c_v)", f"{data['model_metrics']['coherence_c_v']:.3f}")
            st.metric("Coherence (u_mass)", f"{data['model_metrics']['coherence_u_mass']:.3f}")
            st.metric("Jaccard Similarity", f"{data['model_metrics']['jaccard_similarity']:.3f}")
        with col2:
            st.metric("Coherence (npmi)", f"{data['model_metrics']['coherence_npmi']:.3f}")
            st.metric("Topic Diversity", f"{data['model_metrics']['topic_diversity']:.3f}")

        # -------------------------------
        #  All Topics and Top Words (Paginated)
        # -------------------------------
        st.subheader("All Topics and Top Words")

        all_topics = sorted(data["topics"].items(), key=lambda x: int(x[0]))
        topics_per_page = 10
        total_topic_pages = (len(all_topics) - 1) // topics_per_page + 1

        topic_page = st.number_input(
            label="Topic Page",
            min_value=1,
            max_value=total_topic_pages,
            value=1,
            step=1,
            key="topic_page_selector"
        )

        start = (topic_page - 1) * topics_per_page
        end = start + topics_per_page
        paged_topics = all_topics[start:end]

        from collections import defaultdict
        for topic_id, topic_info in paged_topics:
            video_count = len(topic_info.get("videos", []))  # Get number of videos for this topic
            with st.expander(f"Topic {topic_id} ({video_count} videos)"):
                st.markdown("**Top Words:**")
                st.markdown(", ".join(topic_info["top_words"]))

                topic_subtopics = data.get("topic_to_subtopics", {}).get(topic_id)

                if topic_subtopics:
                    st.markdown("### **Subtopic Summary:**")
                    category_to_subtopics = defaultdict(list)

                    for subtopic in topic_subtopics:
                        category = data.get("subtopic_categories", {}).get(subtopic, "Unknown")
                        category_to_subtopics[category].append(subtopic)

                    for category in sorted(category_to_subtopics):
                        st.markdown(f"### Category: {category}")
                        for subtopic in sorted(category_to_subtopics[category]):
                            # Keywords
                            keywords = data.get("subtopic_keywords", {}).get(subtopic, {}).get(topic_id, [])

                            # Video count
                            video_count = len(data.get("subtopic_to_videos", {}).get(subtopic, []))

                            st.markdown(f"- #### **{subtopic}** ({video_count} videos)")
                            if keywords:
                                st.markdown(f"  - **Keywords:** {', '.join(keywords)}")
                            else:
                                st.markdown("  - *(No keywords available)*")
                else:
                    st.info("No category/subtopic assignments for this topic.")

        # -------------------------------
        #  Videos Without Assigned Topics
        # -------------------------------
        st.subheader("Videos Without Assigned Topics")

        if "videos_without_topics" in data and data["videos_without_topics"]:
            unassigned_videos = data["videos_without_topics"]
            st.write(f"{len(unassigned_videos)} video(s) have no assigned topics.")

            videos_per_page = 10
            total_pages = (len(unassigned_videos) - 1) // videos_per_page + 1

            page = st.number_input(
                label="Unassigned Videos",
                min_value=1,
                max_value=total_pages,
                value=1,
                step=1,
                key="unassigned_video_page"
            )

            start_idx = (page - 1) * videos_per_page
            end_idx = start_idx + videos_per_page
            paged_video_ids = unassigned_videos[start_idx:end_idx]

        
            for video_id in paged_video_ids:
                title = data["videos"].get(video_id, {}).get("title", "Unknown Title")
                if st.button(f"📽️ {title}", key=f"no_topic_{video_id}"):
                    st.session_state.selected_video = video_id
                    st.session_state.active_tab = "videos"
                    st.rerun()
        else:
            st.success("All videos have at least one topic assignment.")


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
        
        from collections import defaultdict

        with tab2:
            if "topic_to_subtopics" in data and selected_topic in data["topic_to_subtopics"]:
                st.subheader("Subtopics for this Topic")
                
                # Group subtopics by category
                category_to_subtopics = defaultdict(list)
                for subtopic in data["topic_to_subtopics"][selected_topic]:
                    category = data["subtopic_categories"].get(subtopic, "Unknown")
                    category_to_subtopics[category].append(subtopic)

                for category in sorted(category_to_subtopics):
                    with st.expander(f"{category}"):
                        for subtopic in sorted(category_to_subtopics[category]):
                            st.markdown(f"### {subtopic}")
                            
                            # Show subtopic keywords
                            if "subtopic_keywords" in data and subtopic in data["subtopic_keywords"]:
                                for topic_id, keywords in data["subtopic_keywords"][subtopic].items():
                                    if topic_id == selected_topic:
                                        st.write("**Keywords:**")
                                        st.write(f"- {', '.join(keywords)}")

                            # Show videos in this subtopic
                            if "subtopic_to_videos" in data and subtopic in data["subtopic_to_videos"]:
                                subtopic_videos = data["subtopic_to_videos"][subtopic]
                                st.write(f"**Videos in this subtopic:** {len(subtopic_videos)}")

                                # Pagination
                                page_size = 5
                                total_pages = (len(subtopic_videos) - 1) // page_size + 1
                                page_key = f"page_{selected_topic}_{subtopic}"  # unique key per subtopic
                                current_page = st.number_input(
                                    f"Page for {subtopic}",
                                    min_value=1,
                                    max_value=total_pages,
                                    value=1,
                                    step=1,
                                    key=page_key
                                )

                                start_idx = (current_page - 1) * page_size
                                end_idx = start_idx + page_size
                                paged_videos = list(subtopic_videos)[start_idx:end_idx]

                                for vid in paged_videos:
                                    title = data["videos"][vid]["title"]
                                    if st.button(f"▶️ {title}", key=f"sub_vid_{subtopic}_{vid}"):
                                        st.session_state.selected_video = vid
                                        st.session_state.active_tab = "videos"
                                        st.rerun()
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
        video_id = selected_video.replace(".txt", "")
        youtube_url = f"https://www.youtube.com/watch?v={video_id}"

        # Reverse lookup: find channel containing this video
        channel_id = None
        channel_title = "Unknown Channel"

        for ch_id, ch_info in data.get("channel_summary", {}).items():
            if selected_video in ch_info.get("video_ids", []):
                channel_id = ch_id
                channel_title = ch_info.get("channel_title", "Unknown Channel")
                break

        channel_url = f"https://www.youtube.com/channel/{channel_id}" if channel_id else None

        # Display video and channel info
        st.markdown(f"[Watch on YouTube]({youtube_url})", unsafe_allow_html=True)
        if channel_url:
            st.markdown(f"**Channel:** [{channel_title}]({channel_url})", unsafe_allow_html=True)
        else:
            st.markdown(f"**Channel:** {channel_title}")


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

            subtopic_data = video_data.get("subtopic_assignments", {})
            
            if not subtopic_data:
                st.write("No subtopics assigned to this video")
            else:
                assigned_subtopics = []

                for topic_id, category_map in subtopic_data.items():
                    for category, subtopics in category_map.items():
                        for subtopic, matched_keywords in subtopics.items():
                            assigned_subtopics.append({
                                "Topic": topic_id,
                                "Category": category,
                                "Subtopic": subtopic,
                                "Matched Keywords": ", ".join(sorted(matched_keywords)) if matched_keywords else "None"
                            })

                if assigned_subtopics:
                    st.dataframe(
                        assigned_subtopics,
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.write("No subtopics assigned to this video")

        st.subheader("Original Text")
        st.text_area(
            label="Original Text",
            value=video_data["original_text"],
            height=300,
            key="original_text_display"
        )

        st.subheader("Preprocessed Text")
        st.text_area(
            label="Preprocessed Text",
            value=video_data["preprocessed_text"],
            height=300,
            key="preprocessed_text_display"
        )

        st.subheader("N-grams")
        st.text(video_data["ngrams"])

    
    elif tab == "Channel Summary":
        st.session_state.active_tab = "channel"
        st.header("Channel Summary")

        if "channel_summary" not in data or not data["channel_summary"]:
            st.warning("No channel summary data found.")
            st.stop()

        channel_summary = data["channel_summary"]
        all_channels = [(cid, info["channel_title"]) for cid, info in channel_summary.items()]
        channel_titles = [f"{title} ({cid})" for cid, title in all_channels]

        prev_cid = st.session_state.get("selected_channel_id")
        default_index = next((i for i, (cid, _) in enumerate(all_channels) if cid == prev_cid), 0)

        selected_title = st.selectbox("Select a Channel", options=channel_titles, index=default_index)
        selected_channel_id = all_channels[channel_titles.index(selected_title)][0]
        st.session_state.selected_channel_id = selected_channel_id

        info = channel_summary[selected_channel_id]
        channel_title = info["channel_title"]
        channel_video_ids = set(info["video_ids"])
        topic_map = info.get("topic_category_map", {})

        # Videos without assigned topic
        unassigned_topic_vids = [vid for vid in data.get("videos_without_topics", []) if vid in channel_video_ids]

        # Videos with topic but without subtopic
        unassigned_videos_dict = data.get("subtopic_assignments", {}).get("unassigned_videos_by_group", {})
        vids_with_subtopic = {
            vid
            for topic in topic_map
            for category in topic_map[topic]
            for subtopic in topic_map[topic][category]
            for vid in topic_map[topic][category][subtopic]
            if vid in channel_video_ids
        }

        unassigned_subtopic_vids = []
        for group_str, group_data in unassigned_videos_dict.items():
            group_index = int(group_str)
            for vid_info in group_data.get("videos", []):
                vid_id = vid_info.get("video_id")
                if vid_id in channel_video_ids and vid_id not in vids_with_subtopic:
                    title = vid_info.get("title", "Unknown Title")
                    unassigned_subtopic_vids.append((vid_id, title, group_index))

        st.markdown(f"## {channel_title} ({len(channel_video_ids)} videos)")
        st.markdown(f"[Visit Channel](https://www.youtube.com/channel/{selected_channel_id})", unsafe_allow_html=True)

        main_tabs = st.tabs(["Videos With Topics", "Videos Without Topic"])

        # ----------------------
        # Videos With Topics
        # ----------------------
        with main_tabs[0]:
            sub_tabs = st.tabs([" Assigned Subtopics", " Without Subtopics"])

            # Assigned Subtopics
            with sub_tabs[0]:
                # Build a list of (topic_id, unique_channel_vid_count)
                topic_counts = []
                for topic in topic_map:
                    unique_vids = set()
                    for category in topic_map[topic]:
                        for subtopic in topic_map[topic][category]:
                            for vid in topic_map[topic][category][subtopic]:
                                if vid in channel_video_ids:
                                    unique_vids.add(vid)
                    topic_counts.append((topic, len(unique_vids)))

                # Sort topics by descending unique video count
                sorted_topics = sorted(topic_counts, key=lambda x: x[1], reverse=True)

                for topic, topic_video_count in sorted_topics:
                    with st.expander(f"Topic {topic} ({topic_video_count} videos)"):
                        for category in sorted(topic_map[topic]):
                            st.markdown(f"**{category}**")
                            for subtopic in sorted(topic_map[topic][category]):
                                vids = topic_map[topic][category][subtopic]
                                vids = [vid for vid in vids if vid in channel_video_ids]

                                subtopic_key = f"show_subtopic_{selected_channel_id}_{topic}_{category}_{subtopic}"
                                if st.button(f"{subtopic} ({len(vids)} videos)", key=subtopic_key + "_btn"):
                                    st.session_state[subtopic_key] = not st.session_state.get(subtopic_key, False)

                                if st.session_state.get(subtopic_key, False):
                                    video_page_key = f"video_page_{selected_channel_id}_{topic}_{category}_{subtopic}"
                                    st.session_state.setdefault(video_page_key, 0)

                                    videos_per_page = 10
                                    current_video_page = st.session_state[video_page_key]
                                    paginated_vids = vids[current_video_page * videos_per_page : (current_video_page + 1) * videos_per_page]

                                    for i, vid in enumerate(paginated_vids):
                                        title = data["videos"].get(vid, {}).get("title", "Unknown Title")
                                        if st.button(title, key=f"{video_page_key}_{i}_{vid}"):
                                            st.session_state.selected_video = vid
                                            st.session_state.active_tab = "videos"
                                            st.rerun()

                                    vcol1, vcol2 = st.columns([1, 1])
                                    with vcol1:
                                        if st.button("⬅️ Prev", key=f"{video_page_key}_prev", disabled=current_video_page == 0):
                                            st.session_state[video_page_key] = max(0, current_video_page - 1)
                                            st.rerun()
                                    with vcol2:
                                        if st.button("➡️ Next", key=f"{video_page_key}_next", disabled=(current_video_page + 1) * videos_per_page >= len(vids)):
                                            st.session_state[video_page_key] += 1
                                            st.rerun()


            # Without Subtopics
            with sub_tabs[1]:
                if not unassigned_subtopic_vids:
                    st.info("No videos without subtopic assignment.")
                else:
                    st.write(f"{len(unassigned_subtopic_vids)} videos without subtopic")

                    video_page_key = f"video_page_{selected_channel_id}_unassigned_subtopic"
                    st.session_state.setdefault(video_page_key, 0)

                    videos_per_page = 10
                    current_video_page = st.session_state[video_page_key]
                    start_idx = current_video_page * videos_per_page
                    end_idx = min(start_idx + videos_per_page, len(unassigned_subtopic_vids))
                    paginated_vids = unassigned_subtopic_vids[start_idx:end_idx]

                    for i, (vid, title, group_index) in enumerate(paginated_vids):
                        if st.button(f"▶️ {title} (Group {group_index})", key=f"unassigned_subtopic_{selected_channel_id}_{i}_{vid}"):
                            st.session_state.selected_video = vid
                            st.session_state.active_tab = "videos"
                            st.rerun()

                    vcol1, vcol2 = st.columns([1, 1])
                    with vcol1:
                        if st.button("⬅️ Prev", key=f"{video_page_key}_prev", disabled=current_video_page == 0):
                            st.session_state[video_page_key] = max(0, current_video_page - 1)
                            st.rerun()
                    with vcol2:
                        if st.button("➡️ Next", key=f"{video_page_key}_next", disabled=end_idx >= len(unassigned_subtopic_vids)):
                            st.session_state[video_page_key] += 1
                            st.rerun()

        # -------------------------
        # Videos Without Topics
        # -------------------------
        with main_tabs[1]:
            if not unassigned_topic_vids:
                st.info("No videos without assigned topic.")
            else:
                video_page_key = f"video_page_{selected_channel_id}_unassigned_topic"
                st.session_state.setdefault(video_page_key, 0)

                videos_per_page = 10
                current_video_page = st.session_state[video_page_key]
                start_idx = current_video_page * videos_per_page
                end_idx = min(start_idx + videos_per_page, len(unassigned_topic_vids))
                paginated_vids = unassigned_topic_vids[start_idx:end_idx]

                for i, vid in enumerate(paginated_vids):
                    title = data["videos"].get(vid, {}).get("title", "Unknown Title")
                    if st.button(title, key=f"unassigned_topic_vid_{selected_channel_id}_{i}_{vid}"):
                        st.session_state.selected_video = vid
                        st.session_state.active_tab = "videos"
                        st.rerun()

                vcol1, vcol2 = st.columns([1, 1])
                with vcol1:
                    if st.button("⬅️ Prev", key=f"{video_page_key}_prev", disabled=current_video_page == 0):
                        st.session_state[video_page_key] = max(0, current_video_page - 1)
                        st.rerun()
                with vcol2:
                    if st.button("➡️ Next", key=f"{video_page_key}_next", disabled=end_idx >= len(unassigned_topic_vids)):
                        st.session_state[video_page_key] += 1
                        st.rerun()
