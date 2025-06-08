# bertopic_module.py
import streamlit as st
import pandas as pd
import os
import streamlit.components.v1 as components
import html
from urllib.parse import quote
import re

VIDEOS_PER_PAGE = 10
VIDEOS_PER_PAGE2 = 10  

# Load data
topic_segment_df = pd.read_csv("[T3 UPDATED] BERTopic-topics per segment w channels.csv")
topic_category_df = pd.read_csv("BERTopic w categories.csv")

def show_bertopic_section():
    st.header("BERTopic")

    # File paths
    csv_path = "BERTopic-coherence_scores.csv"
    topics_path = "[T3 UPDATED] BERTopic-topics w labels.csv"
    html_file_path = "topic_barchart.html"

    try:
        # Load coherence scores
        scores_df = pd.read_csv(csv_path)
        scores_df["Score"] = scores_df["Score"].round(3)
        data = {
            "model_metrics": {
                metric: float(score) for metric, score in zip(scores_df["Metric"], scores_df["Score"])
            }
        }

        # Load topics
        topics_df = pd.read_csv(topics_path)
        # Fill missing values with empty string
        topics_df = topics_df.fillna("")

        if "bertopic_active_tab" not in st.session_state:
            st.session_state.bertopic_active_tab = "metrics"

        tab = st.radio(
            "Navigate BERTopic Section",
            ["Model Summary", "Topics Summary", "Video Summary", "Channel Summary"],
            index=["metrics", "topics", "videos", "channels"].index(st.session_state.bertopic_active_tab)
        )
        st.markdown("---")

        if tab == "Model Summary":
            st.session_state.bertopic_active_tab = "metrics"

            # Display all topics with default & custom label + expandable representation
            st.subheader("Topics Overview")
            for _, row in topics_df.iterrows():
                topic_num = int(row["Topic"])
                default_name = row["Name"]
                custom_label = row["Custom Label"]
                representation = eval(row["Representation"]) if isinstance(row["Representation"], str) else []

                with st.expander(f"**Topic {topic_num}** — {custom_label}"):
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
            if os.path.exists(html_file_path):
                with open(html_file_path, "r", encoding="utf-8") as f:
                    html_data = f.read()
                components.html(html_data, height=700, scrolling=True)
            else:
                st.error(f"HTML file not found at: {os.path.abspath(html_file_path)}")

        elif tab == "Topics Summary":
            st.session_state.bertopic_active_tab = "topics"

            st.markdown("## 📊 Video Distribution per Topic")

            topic_segment_df = pd.read_csv("[T3 UPDATED] BERTopic-topics per segment w channels.csv")
            topic_category_df = pd.read_csv("BERTopic w categories.csv")

            video_stats_path = "[T3 UPDATED] BERTopic - videos per topic statistics.csv"

            try:
                video_stats_df = pd.read_csv(video_stats_path)
                video_stats_df = video_stats_df.sort_values("Video Count", ascending=False)

                # Drop the "Label" column
                video_stats_df = video_stats_df.drop(columns=["Label"])

                # Optional: Format large numbers (e.g., 2,594)
                video_stats_df["Video Count"] = video_stats_df["Video Count"].apply(lambda x: f"{x:,}")

                st.dataframe(video_stats_df, use_container_width=True)

            except FileNotFoundError:
                st.error(f"Could not find video statistics file at: {video_stats_path}")


            


            st.header("Topics Summary")
            st.subheader("Categories")

            # Prepare category to topic mapping
            category_to_topics = topic_category_df.groupby("Category")["Topic #"].unique().to_dict()
            all_categories = sorted(category_to_topics.keys())

            categories = ["Food", "Culture", "Lifestyle", "Travel", "Politics", "Entertainment", "Others"]

            # Category Tabs
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

                    # Summarize channels for the selected topic
                    channel_summary = topic_data.groupby(["Channel Title", "Channel Id"]).agg(
                        Video_Count=("Video Id", pd.Series.nunique),
                        Segment_Count=("Segment", "count")
                    ).reset_index().sort_values("Segment_Count", ascending=False)

                    st.markdown("#### Top Channels Contributing to This Topic:")

                    # Format channel name as markdown link
                    channel_summary["Channel"] = channel_summary.apply(
                        lambda row: f"[{html.escape(row['Channel Title'])}](https://www.youtube.com/channel/{row['Channel Id']})", axis=1
                    )

                    # Select columns for display
                    channel_summary_display = channel_summary[["Channel", "Video_Count", "Segment_Count"]]
                    channel_summary_display.columns = ["Channel", "Video Count", "Segment Count"]

                    # Sort by Video Count and show top 10
                    top_n = 10
                    top_channels = channel_summary_display.sort_values("Video Count", ascending=False).head(top_n)

                    # Display top 10 summary
                    st.markdown("#### Top 10 Channels for this Topic")
                    st.markdown(top_channels.to_markdown(index=False), unsafe_allow_html=True)

                    # Full table inside an expander
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





        elif tab == "Video Summary":
            st.session_state.bertopic_active_tab = "videos"
            st.header("🎬 Video Summary")

            try:
                topics_per_video_df = pd.read_csv("topics_per_video_with_channels.csv")
                segments_df = pd.read_csv("[T3 UPDATED] BERTopic-topics per segment w channels.csv")
                topic_category_df = pd.read_csv("BERTopic w categories.csv")

                # Create display names for dropdown
                topics_per_video_df["Display Title"] = topics_per_video_df["Video Title"] + " (" + topics_per_video_df["Video Id"] + ")"

                selected_video = st.selectbox(
                    "Select a video to view summary:",
                    options=topics_per_video_df["Display Title"].tolist(),
                    index=0
                )

                # Get the selected video row
                selected_video_id = selected_video.split("(")[-1].strip(")")
                video_row = topics_per_video_df[topics_per_video_df["Video Id"] == selected_video_id].iloc[0]

                video_id = video_row["Video Id"]
                video_title = html.escape(video_row["Video Title"])
                channel_title = html.escape(video_row["Channel Title"])
                channel_id = video_row["Channel Id"]
                link = video_row["Link"]
                topic_str = video_row["BERTopic Topics"]

                # Parse topics
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

                # Segments grouped by topic
                video_segments = segments_df[segments_df["Video Id"] == video_id]
                segments_grouped = video_segments.groupby("Topic")["Segment"].apply(list).reset_index()
                segments_grouped = segments_grouped.merge(
                    topic_category_df[["Topic #", "Subtopics", "Label"]],
                    left_on="Topic", right_on="Topic #", how="left"
                )

                # Display section
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



        elif tab == "Channel Summary":
            st.session_state.bertopic_active_tab = "channels"
            st.header("Channel Summary")

            # Load data
            topics_per_video_df = pd.read_csv("topics_per_video_with_channels.csv")
            segments_df = pd.read_csv("[T3 UPDATED] BERTopic-topics per segment w channels.csv")
            topic_category_df = pd.read_csv("BERTopic w categories.csv")

            # Calculate overall channel stats
            total_channels = topics_per_video_df['Channel Title'].nunique()
            channel_video_counts = topics_per_video_df.groupby(['Channel Title', 'Channel Id']).size().reset_index(name='Video Count')
            top_channels = channel_video_counts.sort_values(by='Video Count', ascending=False).head(10)

            st.markdown(f"### 📊 Total Channels: {total_channels}")
            st.markdown("### Top Channels by Video Count")
            st.dataframe(top_channels[['Channel Title', 'Video Count']].reset_index(drop=True), use_container_width=True)

            # Dropdown to select a channel
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

                # Map topic numbers to labels and categories
                topic_category_map = topic_category_df.drop_duplicates(subset="Topic #").set_index("Topic #").to_dict("index")

                # Group segments by category -> topic -> video
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

                # Display in collapsible sections per category
                for cat, topics in category_group.items():
                    st.markdown("---")
                    with st.expander(f"📂 Category: {cat} ({len(topics)} topic{'s' if len(topics) != 1 else ''})"):
                        for topic_key, videos in topics.items():
                            st.markdown(f"#### 🔹 {topic_key}")
                            st.markdown(f"**Videos in this topic:** {len(videos)}")
                            for title, data in videos.items():
                                link = f"https://www.youtube.com/watch?v={data['video_id']}"
                                st.markdown(f"- [{title}]({link}) — {len(data['segments'])} segment(s)")





    except FileNotFoundError:
        st.error(f"Required file not found. Check paths:\n• {csv_path}\n• {topics_path}")
