# hlta_module.py
import streamlit as st
import pandas as pd
import os
import re
import streamlit.components.v1 as components

def show_hlta_section():
    st.header("HLTA Analysis")

    @st.cache_data
    def load_data():
        path = './HLTM/!POSTRUN/'
        topics_df = pd.read_csv(path + 'T3_categories.csv')
        videos_df = pd.read_csv(path + 'T3-topics-per-vid.csv')
        small_videos_df = pd.read_csv(path + 'T3_videos_per_topic.csv')
        return topics_df, videos_df, small_videos_df

    try:
        topics_df, videos_df, small_videos_df = load_data()

        # Process the data (same as in your notebook)
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
                        topic_id = match.group(2).strip()
                        score = float(match.group(3).strip())
                        current_level.append((topic, topic_id, score))
            if current_level:
                levels.append(current_level)
            return levels

        videos_df['Topics'] = videos_df.apply(extract_topics, axis=1)

        # Create mappings
        topic_to_videos = {}
        for _, row in videos_df.iterrows():
            video_title = row['Video Title']
            link = row['Link']
            for level in row['Topics']:
                for topic, topic_id, score in level:
                    if topic not in topic_to_videos:
                        topic_to_videos[topic] = []
                    topic_to_videos[topic].append((video_title, link, score, topic_id))

        video_to_topics = {}
        for _, row in videos_df.iterrows():
            video_title = row['Video Title']
            link = row['Link']
            topics = []
            
            for level_idx, level in enumerate(row['Topics'], 1):
                for topic, topic_id, score in level:
                    topic_details = topics_df[topics_df['Id'] == topic_id]
                    if not topic_details.empty:
                        general = topic_details.iloc[0]['General Category']
                        specific = topic_details.iloc[0]['Specific Category']
                    else:
                        general = "Unknown"
                        specific = "Unknown"
                    
                    topics.append({
                        'topic': topic,
                        'topic_id': topic_id,
                        'level': level_idx,
                        'score': score,
                        'general_category': general,
                        'specific_category': specific
                    })
            
            video_to_topics[video_title] = {
                'link': link,
                'topics': topics
            }

        # Initialize session state
        if "hlta_active_tab" not in st.session_state:
            st.session_state.hlta_active_tab = "metrics"
        if "hlta_selected_video" not in st.session_state:
            st.session_state.hlta_selected_video = next(iter(video_to_topics.keys()))
        if "hlta_selected_topic" not in st.session_state:
            st.session_state.hlta_selected_topic = next(iter(topic_to_videos.keys()))

        # Navigation tabs
        tab = st.radio(
            "Navigate HLTA Section",
            ["Model Metrics", "Topics Explorer", "Video Explorer"],
            index=["metrics", "topics", "videos"].index(st.session_state.hlta_active_tab)
        )
        st.markdown("---")

        if tab == "Model Metrics":
            st.session_state.hlta_active_tab = "metrics"
            
            # Load coherence scores if available
            csv_path = "HLTA-coherence_scores.csv"
            if os.path.exists(csv_path):
                scores_df = pd.read_csv(csv_path)
                scores_df["Score"] = scores_df["Score"].round(3)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Coherence (c_v)", f"{scores_df.loc[scores_df['Metric'] == 'c_v', 'Score'].values[0]:.3f}")
                    st.metric("Coherence (u_mass)", f"{scores_df.loc[scores_df['Metric'] == 'u_mass', 'Score'].values[0]:.3f}")
                with col2:
                    st.metric("Coherence (c_npmi)", f"{scores_df.loc[scores_df['Metric'] == 'c_npmi', 'Score'].values[0]:.3f}")
            else:
                st.warning("Coherence scores file not found")


        elif tab == "Topics Explorer":
            st.session_state.hlta_active_tab = "topics"
            st.header("Topics Explorer")
            
            # Topic selector
            topics = sorted(topic_to_videos.keys())
            selected_topic = st.selectbox(
                "Select Topic",
                options=topics,
                index=topics.index(st.session_state.hlta_selected_topic)
            )
            st.session_state.hlta_selected_topic = selected_topic
            
            # Display topic info
            st.markdown(f"### {selected_topic}")
            
            # Get topic details
            topic_match = selected_topic.split(',')[0].strip()
            topic_details = topics_df[topics_df['Texts'].str.contains(topic_match)]
            if not topic_details.empty:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**General Category:** {topic_details.iloc[0]['General Category']}")
                with col2:
                    st.markdown(f"**Specific Category:** {topic_details.iloc[0]['Specific Category']}")
            
            # Display videos with this topic
            videos = topic_to_videos[selected_topic]
            st.markdown(f"**Videos containing this topic ({len(videos)} total):**")
            
            # Pagination
            videos_per_page = 10
            total_pages = (len(videos) - 1) // videos_per_page + 1
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
            
            start_idx = (page - 1) * videos_per_page
            end_idx = start_idx + videos_per_page
            paged_videos = videos[start_idx:end_idx]
            
            # Display videos in a table
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

        elif tab == "Video Explorer":
            st.session_state.hlta_active_tab = "videos"
            st.header("Video Explorer")
            
            # Video selector
            video_titles = sorted(video_to_topics.keys())
            selected_video = st.selectbox(
                "Select Video",
                options=video_titles,
                index=video_titles.index(st.session_state.hlta_selected_video)
            )
            st.session_state.hlta_selected_video = selected_video
            
            # Display video info
            video_info = video_to_topics[selected_video]
            st.markdown(f"### {selected_video}")
            st.markdown(f"[Watch on YouTube]({video_info['link']})", unsafe_allow_html=True)
            
            # Group topics by level
            levels = {}
            for topic in video_info['topics']:
                levels.setdefault(topic['level'], []).append(topic)
            
            for level in sorted(levels.keys()):
                st.markdown(f"#### Level {level} Topics")
                
                # Create a DataFrame for the table
                level_data = []
                for topic in sorted(levels[level], key=lambda x: x['score'], reverse=True):
                    level_data.append({
                        "Topic": topic['topic'],
                        "Score": topic['score'],
                        "General Category": topic['general_category'],
                        "Specific Category": topic['specific_category']
                    })
                
                st.dataframe(
                    level_data,
                    column_config={
                        "Score": st.column_config.ProgressColumn(
                            "Score",
                            format="%.2f",
                            min_value=0,
                            max_value=1.0
                        )
                    },
                    hide_index=True,
                    use_container_width=True
                )

    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")