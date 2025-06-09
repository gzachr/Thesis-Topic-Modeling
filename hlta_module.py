import streamlit as st
import pandas as pd
import os
import re

def show_hlta_section():
    st.header("HLTA Analysis")

    @st.cache_data
    def load_data():
        path = './HLTM/!POSTRUN/'
        topics_df = pd.read_csv(path + 'T3_categories.csv', encoding='utf-8')
        videos_df = pd.read_csv(path + 'T3-topics-per-vid-with-channels.csv', encoding='latin1')
        return topics_df, videos_df

    @st.cache_data
    def load_coherence_scores():
        csv_path = "hlta-coherence.csv"
        if os.path.exists(csv_path):
            scores_df = pd.read_csv(csv_path)
            scores_df["score"] = scores_df["score"].round(3)
            return scores_df
        return None

    @st.cache_data
    def process_data(topics_df, videos_df):
        topics_df["Id"] = topics_df["Id"].astype(str).str.strip()
        id_to_category = topics_df.set_index("Id")[["General Category", "Specific Category"]].to_dict("index")

        def extract_topics(row):
            levels = []
            current_level = []
            for line in row['HLTA Topics'].split('\n'):
                if line.startswith('Level'):
                    if current_level:
                        levels.append(current_level)
                        current_level = []
                elif line.strip() and not line.startswith('Video Title'):
                    match = re.match(r'^(.*?):\s*\(([^)]+)\)\s*\(([^)]+)\)', line)
                    if match:
                        topic = match.group(1).strip()
                        topic_id = str(match.group(2).strip())
                        score = float(match.group(3).strip())
                        current_level.append((topic, topic_id, score))
            if current_level:
                levels.append(current_level)
            return levels

        videos_df['Topics'] = videos_df.apply(extract_topics, axis=1)

        topic_to_videos = {}
        video_to_topics = {}

        for _, row in videos_df.iterrows():
            video_title = row['Video Title']
            link = row['Link']
            video_topics = []

            for level_idx, level in enumerate(row['Topics'], 1):
                for topic, topic_id, score in level:
                    cat_info = id_to_category.get(topic_id, {"General Category": "Unknown", "Specific Category": "Unknown"})

                    if topic not in topic_to_videos:
                        topic_to_videos[topic] = {
                            'videos': [],
                            'general_category': cat_info["General Category"],
                            'specific_category': cat_info["Specific Category"]
                        }

                    topic_to_videos[topic]['videos'].append((video_title, link, score, topic_id))

                    video_topics.append({
                        'topic': topic,
                        'topic_id': topic_id,
                        'level': level_idx,
                        'probability': score,
                        'general_category': cat_info["General Category"],
                        'specific_category': cat_info["Specific Category"]
                    })

            video_to_topics[video_title] = {'link': link, 'topics': video_topics}

        return topic_to_videos, video_to_topics

    try:
        topics_df, videos_df = load_data()
        topic_to_videos, video_to_topics = process_data(topics_df, videos_df)

        if "hlta_active_tab" not in st.session_state:
            st.session_state.hlta_active_tab = "metrics"
        if "hlta_selected_video" not in st.session_state:
            st.session_state.hlta_selected_video = next(iter(video_to_topics.keys()))
        if "hlta_selected_topic" not in st.session_state:
            st.session_state.hlta_selected_topic = next(iter(topic_to_videos.keys()))

        tab = st.radio(
            "Navigate HLTA Section",
            ["Model Summary", "Topics Summary", "Video Summary", "Channel Summary"],
            index=["metrics", "topics", "videos", "channels"].index(st.session_state.hlta_active_tab)
        )
        st.markdown("---")

        if tab == "Model Summary":
            st.session_state.hlta_active_tab = "metrics"
            scores_df = load_coherence_scores()
            st.header("Quantitative Metrics")
            if scores_df is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Coherence (c_v)", f"{scores_df.loc[scores_df['metric'] == 'c_v', 'score'].values[0]:.3f}")
                    st.metric("Coherence (u_mass)", f"{scores_df.loc[scores_df['metric'] == 'u_mass', 'score'].values[0]:.3f}")
                with col2:
                    st.metric("Coherence (c_npmi)", f"{scores_df.loc[scores_df['metric'] == 'c_npmi', 'score'].values[0]:.3f}")
            else:
                st.warning("Coherence scores file not found.")

            st.header("Video Statistics")
            st.markdown("This section provides a summary of the topics sorted by the highest frequencey in videos.")

            # Build the statistics from topic_to_videos
            try:
                topic_stats = []
                for topic, data in topic_to_videos.items():
                    topic_stats.append({
                        "Topic": topic,
                        "General Category": data["general_category"],
                        "Specific Category": data["specific_category"],
                        "Video Count": len(data["videos"])
                    })

                stats_df = pd.DataFrame(topic_stats)

                # Sort by Video Count
                stats_df = stats_df.sort_values("Video Count", ascending=False)

                # Format the count nicely
                stats_df["Video Count"] = stats_df["Video Count"].apply(lambda x: f"{x:,}")

                st.dataframe(stats_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Could not compute video statistics: {e}")

        elif tab == "Topics Summary":
            st.session_state.hlta_active_tab = "topics"
            st.header("Topics Summary")

            # Add general category selector
            general_categories = ["All", "Food", "Culture", "Lifestyle", "Travel", "Politics", "Entertainment", "Others"]
            selected_category = st.selectbox("Filter by General Category", options=general_categories, index=0)

            # Filter topics based on selected general category
            if selected_category == "All":
                filtered_topics = sorted(topic_to_videos.keys())
            else:
                filtered_topics = sorted([
                    topic for topic, data in topic_to_videos.items()
                    if data['general_category'] == selected_category
                ])

            if not filtered_topics:
                st.warning("No topics found under the selected category.")
            else:
                selected_topic = st.selectbox("Select Topic", options=filtered_topics, index=None, key="hlta_selected_topic")

                if selected_topic:
                    st.markdown(f"### {selected_topic}")

                    topic_info = topic_to_videos[selected_topic]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**General Category:** {topic_info['general_category']}")
                    with col2:
                        st.markdown(f"**Specific Category:** {topic_info['specific_category']}")

                    videos = topic_info['videos']
                    st.markdown(f"**Videos containing this topic ({len(videos)} total):**")

                    videos_per_page = 10
                    total_pages = (len(videos) - 1) // videos_per_page + 1
                    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)

                    start_idx = (page - 1) * videos_per_page
                    end_idx = start_idx + videos_per_page
                    paged_videos = videos[start_idx:end_idx]

                    for idx, (video_title, link, score, _) in enumerate(paged_videos):
                        with st.container():
                            cols = st.columns([4, 1, 1])
                            with cols[0]:
                                st.markdown(f"[{video_title}]({link})")
                            with cols[1]:
                                st.progress(score, text=f"{score:.2f}")
                            with cols[2]:
                                if st.button("View", key=f"view_{idx}_{video_title}"):
                                    st.session_state.hlta_selected_video = video_title
                                    st.session_state.hlta_active_tab = "videos"
                                    st.rerun()

        elif tab == "Video Summary":
            st.session_state.hlta_active_tab = "videos"
            st.header("Video Summary")

            video_titles = sorted(video_to_topics.keys())
            selected_video = st.selectbox("Select Video", options=video_titles, index=None, key="hlta_selected_video")

            st.markdown(f"### {selected_video}")
            video_info = video_to_topics[selected_video]
            st.markdown(f"[Watch on YouTube]({video_info['link']})", unsafe_allow_html=True)

            levels = {}
            for topic in video_info['topics']:
                levels.setdefault(topic['level'], []).append(topic)

            for level in sorted(levels.keys()):
                st.markdown(f"#### Level {level} Topics")
                level_data = sorted(levels[level], key=lambda x: x['probability'], reverse=True)
                df = pd.DataFrame(level_data).drop(columns=["level"])

                st.dataframe(
                    df,
                    column_config={
                        "Score": st.column_config.ProgressColumn("Score", format="%.2f", min_value=0, max_value=1.0)
                    },
                    hide_index=True,
                    use_container_width=True
                )
        
        elif tab == "Channel Summary":
            st.session_state.hlta_active_tab = "channels"
            st.header("Channel Summary")

            # Filter out missing channels
            if "Channel Title" not in videos_df.columns:
                st.warning("Channel data is not available in the current dataset.")
            else:
                total_channels = videos_df["Channel Title"].nunique()
                channel_video_counts = (
                    videos_df.groupby("Channel Title")
                    .agg(Video_Count=("Video Title", "count"))
                    .reset_index()
                    .sort_values(by="Video_Count", ascending=False)
                )

                st.markdown(f"### 📊 Total Channels: {total_channels}")
                st.markdown("### Top Channels by Video Count")
                st.dataframe(channel_video_counts.head(10).reset_index(drop=True), use_container_width=True)

                channel_options = sorted(channel_video_counts["Channel Title"].tolist())
                selected_channel = st.selectbox("Select a channel", options=channel_options, key="hlta_selected_channel")

                if selected_channel:
                    st.markdown("---")
                    st.subheader(f"📺 {selected_channel}")

                    channel_videos = videos_df[videos_df["Channel Title"] == selected_channel]
                    st.markdown(f"**🎬 Total Videos:** {len(channel_videos)}")

                    # Group by general category and topics
                    general_group = {}
                    for _, row in channel_videos.iterrows():
                        video_title = row["Video Title"]
                        link = row["Channel Link"]
                        if video_title not in video_to_topics:
                            continue
                        for topic in video_to_topics[video_title]["topics"]:
                            cat = topic["general_category"]
                            topic_label = f"{topic['topic']} (Level {topic['level']}, {topic['probability']:.2f})"

                            if cat not in general_group:
                                general_group[cat] = {}
                            if topic_label not in general_group[cat]:
                                general_group[cat][topic_label] = {}
                            if video_title not in general_group[cat][topic_label]:
                                general_group[cat][topic_label][video_title] = {
                                    "link": link,
                                    "count": 0
                                }
                            general_group[cat][topic_label][video_title]["count"] += 1

                    for cat, topics in general_group.items():
                        st.markdown("---")
                        with st.expander(f"📂 Category: {cat} ({len(topics)} topic{'s' if len(topics) != 1 else ''})"):
                            for topic_label, videos in topics.items():
                                st.markdown(f"#### 🔹 {topic_label}")
                                st.markdown(f"**Videos in this topic:** {len(videos)}")
                                for title, data in videos.items():
                                    st.markdown(f"- [{title}]({data['link']}) — {data['count']} mention(s)")



    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
