# bertopic_module.py
import streamlit as st
import pandas as pd
import os
import streamlit.components.v1 as components
import html
from urllib.parse import quote
import re

csv_path="T3 BERTopic files/BERTopic-coherence_scores.csv"
topics_path="T3 BERTopic files/[T3 UPDATED] BERTopic-topics w labels.csv"

@st.cache_data
def load_topic_segment_df():
    return pd.read_csv("T3 BERTopic files/[T3 UPDATED] BERTopic-topics per segment w channels.csv")

@st.cache_data
def load_topic_category_df():
    return pd.read_csv("T3 BERTopic files/BERTopic w categories.csv")

@st.cache_data
def load_topics_per_video_df():
    return pd.read_csv("T3 BERTopic files/topics_per_video_with_channels.csv")


@st.cache_data
def load_coherence_scores(csv_path="T3 BERTopic files/BERTopic-coherence_scores.csv"):
    return pd.read_csv(csv_path)

@st.cache_data
def load_topics_with_labels(topics_path="T3 BERTopic files/[T3 UPDATED] BERTopic-topics w labels.csv"):
    return pd.read_csv(topics_path)

@st.cache_data
def load_html_barchart(html_file_path="T3 BERTopic files/topic_barchart.html"):
    try:
        with open(html_file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None
    
@st.cache_data
def load_video_stats_df():
    return pd.read_csv("T3 BERTopic files/[T3 UPDATED] BERTopic - videos per topic statistics.csv")


VIDEOS_PER_PAGE = 10
VIDEOS_PER_PAGE2 = 10  

def show_bertopic_section():
    st.header("BERTopic")
    try:
        
        scores_df = load_coherence_scores()
        scores_df["Score"] = scores_df["Score"].round(3)
        data = {
            "model_metrics": {
                metric: float(score) for metric, score in zip(scores_df["Metric"], scores_df["Score"])
            }
        }
        topics_df = load_topics_with_labels()
        topics_df = topics_df.fillna("")

        if "bertopic_active_tab" not in st.session_state:
            st.session_state.bertopic_active_tab = "metrics"

        tab = st.radio(
            "Navigate BERTopic Section",
            ["Model Summary", "Topics Summary", "Video Summary", "Channel Summary"],
            index=["metrics", "topics", "videos", "channels"].index(st.session_state.bertopic_active_tab)
        )
        st.markdown("---")

        ### Model Summary section

        if tab == "Model Summary":
            st.session_state.bertopic_active_tab = "metrics"

            st.subheader("Topics Overview")

            topic_options = [
                f"Topic {int(row['Topic'])}: {row['Custom Label']}" for _, row in topics_df.iterrows()
            ]
            selected_topic_option = st.selectbox("Select a Topic", topic_options)

            selected_topic_num = int(selected_topic_option.split(":")[0].replace("Topic", "").strip())
            selected_row = topics_df[topics_df["Topic"] == selected_topic_num].iloc[0]

            default_name = selected_row["Name"]
            custom_label = selected_row["Custom Label"]
            representation = eval(selected_row["Representation"]) if isinstance(selected_row["Representation"], str) else []

            st.markdown(f"### Topic {selected_topic_num}: {custom_label}")
            st.markdown(f"**Default Name:** {default_name}")
            st.markdown(f"**Custom Label:** {custom_label}")
            st.markdown("**Top Keywords:**")
            st.write(", ".join(representation[:10]))


            st.markdown("---")
            st.subheader("📊 Model Coherence Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Coherence (c_v)", f"{data['model_metrics'].get('c_v', 'N/A'):.2f}")
                st.metric("Coherence (u_mass)", f"{data['model_metrics'].get('u_mass', 'N/A'):.2f}")
            with col2:
                st.metric("Coherence (c_npmi)", f"{data['model_metrics'].get('c_npmi', 'N/A'):.2f}")

            st.markdown("---")
            st.subheader("📊 Topic Frequency Bar Chart")

            html_data = load_html_barchart()

            if html_data:
                components.html(html_data, height=700, scrolling=True)
            else:
                st.error(f"HTML file not found")

        ### Topic Summary section

        elif tab == "Topics Summary":
            st.session_state.bertopic_active_tab = "topics"

            st.markdown("## 📊 Video Distribution per Topic")

            topic_segment_df = load_topic_segment_df()
            topic_category_df = load_topic_category_df()

            try:
                video_stats_df = load_video_stats_df()

                video_stats_df = video_stats_df.sort_values("Video Count", ascending=False)
                video_stats_df = video_stats_df.drop(columns=["Label"])

                video_stats_df["Video Count"] = video_stats_df["Video Count"].apply(lambda x: f"{x:,}")

                st.dataframe(video_stats_df, use_container_width=True, hide_index=True)

            except FileNotFoundError:
                st.error(f"Could not find video statistics file at: {load_video_stats_df()}")

            st.header("Topics Summary")
            st.subheader("Categories")

            category_to_topics = topic_category_df.groupby("Category")["Topic #"].unique().to_dict()
            all_categories = sorted(category_to_topics.keys())

            categories = ["Food", "Culture", "Lifestyle", "Travel", "Politics", "Entertainment", "Others"]

            tabs = st.tabs(all_categories)
            for tab, category in zip(tabs, all_categories):
                with tab:
                    topics_in_cat = category_to_topics[category]
                    topic_options = {
                        f"Topic {tid}: {topic_category_df[topic_category_df['Topic #'] == tid].iloc[0]['Subtopics']} ({topic_category_df[topic_category_df['Topic #'] == tid].iloc[0]['Label']})": tid
                        for tid in sorted(topics_in_cat)
                    }

                    selected_topic_label = st.selectbox(f"Select a topic in {category}", options=list(topic_options.keys()), key=f"dropdown_{category}")
                    selected_topic_id = topic_options[selected_topic_label]

                    topic_info = topic_category_df[topic_category_df["Topic #"] == selected_topic_id].iloc[0]
                    label = topic_info["Label"]
                    subtopics = topic_info["Subtopics"]

                    topic_data = topic_segment_df[topic_segment_df["Topic"] == selected_topic_id]
                    grouped_videos = topic_data.groupby([
                        "Video Id", "Video Title", "Channel Title", "Channel Id"
                    ])['Segment'].apply(list).reset_index()
                    grouped_videos["Num Segments"] = grouped_videos["Segment"].apply(len)
                    grouped_videos = grouped_videos.sort_values("Num Segments", ascending=False).reset_index(drop=True)

                    total_videos = len(grouped_videos)
                    total_segments = len(topic_data)

                    st.markdown(f"### Topic {selected_topic_id}: {subtopics} ({label})")
                    st.markdown(f"**Total Videos:** {total_videos} | **Total Segments:** {total_segments}")

                    channel_summary = topic_data.groupby(["Channel Title", "Channel Id"]).agg(
                        Video_Count=("Video Id", pd.Series.nunique),
                        Segment_Count=("Segment", "count")
                    ).reset_index().sort_values("Segment_Count", ascending=False)

                    st.markdown("#### Top Channels Contributing to This Topic:")

                    channel_summary["Channel"] = channel_summary.apply(
                        lambda row: f"[{html.escape(row['Channel Title'])}](https://www.youtube.com/channel/{row['Channel Id']})", axis=1
                    )

                    channel_summary_display = channel_summary[["Channel", "Video_Count", "Segment_Count"]]
                    channel_summary_display.columns = ["Channel", "Video Count", "Segment Count"]

                    top_n = 10
                    top_channels = channel_summary_display.sort_values("Video Count", ascending=False).head(top_n)
                    st.markdown("#### Top 10 Channels for this Topic")
                    st.markdown(top_channels.to_markdown(index=False), unsafe_allow_html=True)

                    with st.expander("Show Full Channel Summary Table"):
                        full_sorted = channel_summary_display.sort_values("Video Count", ascending=False)
                        st.markdown(full_sorted.to_markdown(index=False), unsafe_allow_html=True)


                    total_pages = (len(grouped_videos) - 1) // VIDEOS_PER_PAGE + 1
                    page_number = st.number_input(
                        f"Page for Topic {selected_topic_id}",
                        min_value=1,
                        max_value=total_pages,
                        value=1,
                        key=f"page_topic_{selected_topic_id}"
                    )

                    start_idx = (page_number - 1) * VIDEOS_PER_PAGE
                    end_idx = start_idx + VIDEOS_PER_PAGE

                    for _, row in grouped_videos.iloc[start_idx:end_idx].iterrows():
                        video_id = row['Video Id']
                        title = html.escape(row['Video Title'])
                        channel = html.escape(row['Channel Title'])
                        channel_id = row['Channel Id']
                        segments = row['Segment']
                        num_segments = len(segments)

                        display_title = f"{title} ({video_id}) - {num_segments} segment{'s' if num_segments != 1 else ''}"
                        with st.expander(display_title):
                            st.markdown(f"[▶️ Watch on YouTube](https://www.youtube.com/watch?v={video_id})")
                            st.markdown(f"[📺 {channel}](https://www.youtube.com/channel/{channel_id})")
                            st.markdown("---")
                            for segment in segments:
                                st.markdown(f"- {segment}")

        ### Video Summary section

        elif tab == "Video Summary":
            st.session_state.bertopic_active_tab = "videos"
            st.header("🎬 Video Summary")

            try:
                topics_per_video_df = load_topics_per_video_df()
                segments_df = load_topic_segment_df()
                topic_category_df = load_topic_category_df()

                topics_per_video_df["Display Title"] = topics_per_video_df["Video Title"] + " (" + topics_per_video_df["Video Id"] + ")"

                selected_video = st.selectbox(
                    "Select a video to view summary:",
                    options=topics_per_video_df["Display Title"].tolist(),
                    index=0
                )

                selected_video_id = selected_video.split("(")[-1].strip(")")
                video_row = topics_per_video_df[topics_per_video_df["Video Id"] == selected_video_id].iloc[0]

                video_id = video_row["Video Id"]
                video_title = html.escape(video_row["Video Title"])
                channel_title = html.escape(video_row["Channel Title"])
                channel_id = video_row["Channel Id"]
                link = video_row["Link"]
                topic_str = video_row["BERTopic Topics"]

                # Parse the topics and percentage 
                topic_lines = topic_str.strip().split("\n")
                topic_data = []
                for line in topic_lines:
                    match = re.match(r"Topic (\d+): (.+?)\((.+?)\) \(([\d.]+)%\)", line)
                    if match:
                        topic_id = int(match.group(1))
                        subtopics = match.group(2).strip().rstrip(',')
                        label = match.group(3).strip()
                        pct = float(match.group(4))
                        topic_data.append({
                            "Topic ID": topic_id,
                            "Subtopics": subtopics,
                            "Label": label,
                            "Percentage": pct
                        })

                topic_df = pd.DataFrame(topic_data)
                topic_df = topic_df.sort_values("Percentage", ascending=False)

                # Segments grouped per topic
                video_segments = segments_df[segments_df["Video Id"] == video_id]
                segments_grouped = video_segments.groupby("Topic")["Segment"].apply(list).reset_index()
                segments_grouped = segments_grouped.merge(
                    topic_category_df[["Topic #", "Subtopics", "Label"]],
                    left_on="Topic", right_on="Topic #", how="left"
                )

                st.markdown(f"### 🎥 {video_title} ({video_id})")
                st.markdown(f"[▶️ Watch on YouTube](https://www.youtube.com/watch?v={video_id})")
                st.markdown(f"[📺 {channel_title}](https://www.youtube.com/channel/{channel_id})")

                st.markdown(f"**Total Topics:** {len(topic_df)}")
                st.markdown("**Category & Topics (with percentage):**")

                for _, row in topic_df.iterrows():
                    st.markdown(f"- **Topic {row['Topic ID']}**: {row['Subtopics']} (**{row['Label']}**) — {row['Percentage']}%")

                st.markdown("**📑 Segments Grouped by Topic:**")
                for _, row in segments_grouped.iterrows():
                    topic_id = row["Topic"]
                    subtopics = row["Subtopics"]
                    label = row["Label"]
                    seg_list = row["Segment"]

                    with st.expander(f"Topic {topic_id}: {subtopics} ({label}) - {len(seg_list)} segment{'s' if len(seg_list)!=1 else ''}"):
                        for seg in seg_list:
                            st.markdown(f"- {seg}")

                st.markdown("---")

            except FileNotFoundError as e:
                st.error(f"Missing file: {e.filename}")

        ### Channel Summary section

        elif tab == "Channel Summary":
            st.session_state.bertopic_active_tab = "channels"
            st.header("Channel Summary")

            topics_per_video_df = load_topics_per_video_df()
            segments_df = load_topic_segment_df()
            topic_category_df = load_topic_category_df()

            total_channels = topics_per_video_df['Channel Title'].nunique()
            channel_video_counts = topics_per_video_df.groupby(['Channel Title', 'Channel Id']).size().reset_index(name='Video Count')
            top_channels = channel_video_counts.sort_values(by='Video Count', ascending=False).head(10)

            st.markdown(f"### 📊 Total Channels: {total_channels}")
            st.markdown("### Top Channels by Video Count")
            st.dataframe(top_channels[['Channel Title', 'Video Count']].reset_index(drop=True), use_container_width=True)

            channel_options = channel_video_counts.sort_values(by="Channel Title")["Channel Title"].tolist()
            selected_channel = st.selectbox("Select a channel", options=channel_options)

            if selected_channel:
                channel_id = channel_video_counts[channel_video_counts['Channel Title'] == selected_channel]['Channel Id'].values[0]
                channel_link = f"https://www.youtube.com/channel/{channel_id}"

                channel_videos_df = topics_per_video_df[topics_per_video_df['Channel Title'] == selected_channel]
                channel_segments_df = segments_df[segments_df['Channel Title'] == selected_channel]

                st.markdown(f"### 📺 {selected_channel}")
                st.markdown(f"[🔗 Visit Channel]({channel_link})")
                st.markdown(f"**🎬 Total Videos:** {len(channel_videos_df)}")
                st.markdown(f"**Topics:** {channel_segments_df['Topic'].nunique()} | **Total Segments:** {len(channel_segments_df)}")
                
                topic_category_map = topic_category_df.drop_duplicates(subset="Topic #").set_index("Topic #").to_dict("index")

                
                category_group = {}
                for _, row in channel_segments_df.iterrows():
                    topic_id = row['Topic']
                    video_title = row['Video Title']
                    video_id = row['Video Id']
                    if topic_id not in topic_category_map:
                        continue
                    cat = topic_category_map[topic_id]['Category']
                    subtopics = topic_category_map[topic_id]['Subtopics']
                    label = topic_category_map[topic_id]['Label']
                    topic_key = f"Topic {topic_id}: {subtopics} ({label})"

                    if cat not in category_group:
                        category_group[cat] = {}
                    if topic_key not in category_group[cat]:
                        category_group[cat][topic_key] = {}
                    if video_title not in category_group[cat][topic_key]:
                        category_group[cat][topic_key][video_title] = {
                            "video_id": video_id,
                            "segments": []
                        }
                    category_group[cat][topic_key][video_title]["segments"].append(row['Segment'])

                category_video_counts = {}
                for cat, topics in category_group.items():
                    unique_videos_in_cat = set()
                    for topic_key, videos in topics.items():
                        unique_videos_in_cat.update(videos.keys())
                    category_video_counts[cat] = len(unique_videos_in_cat)

                # Sort categories by total videos (descending)
                sorted_categories = sorted(category_group.items(), key=lambda x: category_video_counts[x[0]], reverse=True)

                for cat, topics in sorted_categories:
                    st.markdown("---")
                    with st.expander(f"📂 Category: {cat} ({len(topics)} topic{'s' if len(topics) != 1 else ''})"):
                        # Sort topics by number of unique videos (descending)
                        sorted_topics = sorted(topics.items(), key=lambda x: len(x[1]), reverse=True)
                        for topic_key, videos in sorted_topics:
                            st.markdown(f"#### 🔹 {topic_key}")
                            st.markdown(f"**Videos in this topic:** {len(videos)}")

                            # Sort videos by number of segments (descending)
                            sorted_videos = sorted(videos.items(), key=lambda x: len(x[1]['segments']), reverse=True)
                            for title, data in sorted_videos:
                                link = f"https://www.youtube.com/watch?v={data['video_id']}"
                                st.markdown(f"- [{title}]({link}) — {len(data['segments'])} segment(s)")


    except FileNotFoundError:
        st.error(f"Required file not found. Check paths:\n• {csv_path}\n• {topics_path}")