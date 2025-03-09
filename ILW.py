#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import string
import unicodedata
import nltk
import spacy
import gensim
import requests
import hdbscan
import numpy as np
import re
import hdbscan
import matplotlib.pyplot as plt
import streamlit as st

from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures, TrigramCollocationFinder, TrigramAssocMeasures
from nltk import pos_tag
from nltk.util import ngrams
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


# Load NLP Models
spacy_nlp = spacy.load("en_core_web_sm")
#embedding_model = SentenceTransformer("all-MiniLM-L6-v2") faster
embedding_model = SentenceTransformer("all-mpnet-base-v2") 
stop_words = set(stopwords.words("english"))

# Fetch stopwords
def fetch_stopwords_from_github(url):
    response = requests.get(url)
    return set(response.text.splitlines())

github_stopwords_url = 'https://raw.githubusercontent.com/stopwords-iso/stopwords-en/master/stopwords-en.txt'
github_stopwords = fetch_stopwords_from_github(github_stopwords_url)

stop_words = set(stopwords.words('english'))
custom_stop_words = ['like', 'yeah', 'know', 'um', 'uh', 'really', 'one', 'go', 'right', 'okay', 'well', 'said', 
                    'going', 'got', 'na', 'always', 'every', 'each', 'say', 'el', 'little', 'still', 
                    'best', 'dutch', 'nice', 'great', 'awesome', 'good', 'cool', 'love', 'amazing', 'wow',
                    'breaking news', 'report', 'coverage', 'investigation', 'interview', 'documentary', 'news', 'netherlands', 'psy', 'subtitle', 'description', 'link', 
                    'journalist', 'headline', 'reporter', 'current events', 'special report', 
                    'analysis', 'documented', 'broadcast', 'reporting', 'v', 'food', 'travel', 'react', 
                    'reacts', 'reaction', 'foreigner', 'thing', 'visit', 'dc', 'japan', 'first', 'fast', 
                    'asia', 'ang', 'indian', 'thai', 'vietnamese', 'russia', 'gon', 'canada', 'canadian', 'russian', 
                    'russia', 'guy', 'lot', 'bit', 'diba', 'ola', 'cuz', 'thai', 'thailand', 'person', 'citizen', 'foreigner', 'foreign', 'foreigners',
                    'facebook', 'filipinos', 'filipinas', 'vlog', 'vlogs', 'vlogging', 'hashtag', 'india', 'bro', 'dito', 'people', 'time', 'music', 'gonna', 'life', 
                    'lol', 'guys', 'tho', 'cute', 'hmm', 'huh', 'channel', 'subscribe', 'day6', 'mandarin', 'chinese', 'beautiful',
                    'chuckles', 'fbe', 'hit', 'laughs', 'yo', 'ka', 'word', 'living', 'boi', 'minimum', 'ya', 'successful', 'perfectly', 'yeap', 
                    'wondering', 'fantastic', 'hurry', 'german', 'age', 'country', 'subscribing', 'bluesy', 'jump', 'pretty', 'understanding', 'personalized',
                    'and', 'the', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'up', 'about', 'over', 'into', 'through', 'between', 'under', 'against', 'all',
                    'you', 'haha', 'hahaha', 'ha', 'hey', 'bye', 'hello', 'hi', 'oh', 'blah', 'easy', 'alright', 'ta', 'day', 'ooh', 'en', 'do', 'lot', 'comment', 'notification', 
                    'korean', 'jjajangmyeon', 'jajangmyeon', 'damn', 'yall', 'month', 'week', 'year', 'ohhh', 'pvf', 'dude', 'mmm', 'kagilagilalas', 'ofcourse', 'australia', 'uxo', 
                    'atleast', 'yusuf', 'bangkok', 'ot', 'anytime', 'allover', 'kala', 'nope', 'wan', 'brazil', 'smooth', 'ot', 'timeshere', 'batchof', 'yep', 'opo', 'del',
                    'gosh', 'po', 'ourself', 'wo', 'wait', 'ugh', 'nyc', 'whoa', 'nicaragua', 'yup', 'em', 'bout', 'le', 'omg', 'overwhelm', 'maam', 'nicer', 'haha', 'hahaha', 'ha', 
                    'nbcs', 'lana', 'rc', 'whatsoever', 'oxy', 'decade', 'whyd', 'unknown', 'ahhhhh', 'ohoh', 'ohto', 'ohhhh', 'bruh', 'ooe', 'ahmedabad', 'mexico', 
                    'understand', 'excuse', 'kinda', 'applause', 'oooh', 'thiswhat', 'nevermind', 'ahh', 'againthank', 'toto', 'aww', 'nah', 'bbmas', 'ay', 'op', 'huh', 'huhu',
                    'tada', 'beacuse', 'voila', 'upstairs', 'thatswhy', 'yea', 'that', 'armenia', 'or', 'not', 'funwhat', 'aka', 'armeniathat', 'woosexy', 'worth', 'laugh', 'box', 
                    'xd', 'vb', 'eff', 'ananya', 'welsh', 'latron', 'shout', 'whatwhat', 'what', 'pause', 'why', 'thats', 'byebye', 'iv', 'bye', 'ado', 'ownup', 'dom', 'jomm', 'sir', 
                    'budgie', 'nomac', 'lavocha', 'germany', 'why', 'walang', 'superduper', 'philip', 'mom', 'jre', 'giddy', 'intro', 'dupe', 'europe', 'dream', 'team', 'dislike', 'content', 
                    'yoongi', 'royale', 'ilu', 'jhope', 'day', 'jin', 'ecc', 'nyhs', 'nego', 'chavez', 'pb', 'everyones', 'epic', 'matter', 'oneonone', 'region', 'change', 'ho', 'seetoh', 
                    'atin', 'vpn', 'facetune', 'busu', 'mackie', 'clyd', 'china', 'rest', 'friend', 'woah', 'dindins', 'poster', 'vibe', 'woman', 'boss', 'woah', 'type', 'mahana', 'joke', 
                    'taller', 'insane', 'whang', 'psa', 'manatee', 'recommend', 'caesar', 'mmmhmm', 'mosul', 'dun', 'clue', 'naysayer', 'hindi', 'ko', 'pero', 'bulgaria', 'question', 'video', 
                    'yobi', 'hindu', 'expat', 'option', 'gap', 'eu', 'simo', 'kouignamann', 'bct', 'month', 'cfo', 'philippines', 'philippine', 'british', 'filipino', 'video', 
                    'http', 'korea', 'korean', 'youtube', 'google', 'united', 'america', 'american', 'kpop', '필리핀', 'bts', 'blackpink', 'twice', 'exo', 'k-pop', 
                    'seventeen', 'stray kids', 'nct', 'kdrama', 'aespa', 'taehyung', 'jimin', 'jungkook']
stop_words.update(custom_stop_words, github_stopwords)

lemmatizer = WordNetLemmatizer()

# Folder paths
transcripts_folder_path = 'standard_dataset/'
tags_folder_path = 'tags/'

# Function to load video tags only for fetched video IDs
def load_video_tags(folder_path, video_ids):
    video_tags = {}
    for video_id in video_ids:
        tag_file = os.path.join(folder_path, f"{video_id}.txt")
        if os.path.exists(tag_file):
            with open(tag_file, "r", encoding="utf-8") as file:
                tags_content = file.read().lower()
                video_tags[video_id] = tags_content.split()  # Store as list of words
        else:
            video_tags[video_id] = []  # Default to empty list if no tags
    return video_tags

video_ids = []
transcript_files = []
for file_name in os.listdir(transcripts_folder_path):
    if file_name.endswith('.txt'):
        video_id = file_name.split('_captions')[0]
        video_ids.append(video_id)
        transcript_files.append((video_id, file_name)) 

video_tags = load_video_tags(tags_folder_path, video_ids)


# In[3]:


def is_latin_script(word):
    return all('LATIN' in unicodedata.name(char, '') or char.isdigit() for char in word)

# Function to detect both bigram and trigram collocations
def detect_collocations(tokens, min_freq=3):
    bigram_measures = BigramAssocMeasures()
    trigram_measures = TrigramAssocMeasures()

    # Find bigrams
    bigram_finder = BigramCollocationFinder.from_words(tokens)
    bigram_finder.apply_freq_filter(min_freq)
    bigrams = set(['_'.join(bigram) for bigram in bigram_finder.nbest(bigram_measures.pmi, 10)])

    # Find trigrams
    trigram_finder = TrigramCollocationFinder.from_words(tokens)
    trigram_finder.apply_freq_filter(min_freq)
    trigrams = set(['_'.join(trigram) for trigram in trigram_finder.nbest(trigram_measures.pmi, 10)])

    return bigrams, trigrams

def preprocess_text(doc, video_id, tag_weight=2, ngram_weight_factor=2):
    # Segment the text into meaningful chunks
    doc = re.sub(r'([a-zA-Z]+)[,;:!?.]', r'\1', doc)
    
    bigram_trigram_words = []
    
    doc = doc.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(doc)

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words and word.isalpha() and is_latin_script(word)]

    # Detect meaningful bigram and trigram collocations
    bigrams, trigrams = detect_collocations(tokens)

    # Generate n-grams (both bigrams and trigrams)
    bigram_tokens = ['_'.join(gram) for gram in ngrams(tokens, 2)]
    trigram_tokens = ['_'.join(gram) for gram in ngrams(tokens, 3)]

    # Count n-gram frequency
    bigram_frequencies = Counter(bigram_tokens)
    trigram_frequencies = Counter(trigram_tokens)

    # Merge n-grams into single tokens
    merged_tokens = []
    i = 0
    while i < len(tokens) - 2:  # Check for trigrams first
        trigram = f"{tokens[i]}_{tokens[i+1]}_{tokens[i+2]}"
        bigram = f"{tokens[i]}_{tokens[i+1]}"

        if trigram in trigrams:
            merged_tokens.append(trigram)
            bigram_trigram_words.append(trigram)
            i += 3  # Skip next two words since it's part of the trigram
        elif bigram in bigrams:
            merged_tokens.append(bigram)
            bigram_trigram_words.append(bigram)
            i += 2  # Skip next word since it's part of the bigram
        else:
            merged_tokens.append(tokens[i])
            i += 1

    # Append any remaining words
    while i < len(tokens):
        merged_tokens.append(tokens[i])
        i += 1

    # POS tagging
    tokens_with_pos = pos_tag(merged_tokens)

    # Apply lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in tokens_with_pos]

    # Assign weight based on n-gram occurrence
    weighted_tokens = []
    for token in lemmatized_tokens:
        if token in trigram_frequencies:
            token_weight = 1 + trigram_frequencies[token] * ngram_weight_factor  
        elif token in bigram_frequencies:
            token_weight = 1 + bigram_frequencies[token] * (ngram_weight_factor - 1)  
        else:
            token_weight = 1
        weighted_tokens.extend([token] * int(token_weight))

    # Include video tags
    if video_id in video_tags:
        tags = video_tags[video_id]
        for tag in tags:
            if tag.isalpha():
                weighted_tokens.extend([tag] * int(tag_weight))

    return ' '.join(weighted_tokens), bigram_trigram_words

def get_wordnet_pos(treebank_tag):
    """Convert POS tag to WordNet format for lemmatization."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun
    
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def plot_wordclouds(lda_model, num_words=30):
    topics = [lda_model.show_topic(i, num_words) for i in range(lda_model.num_topics)]
    non_empty_topics = [t for t in topics if t]  # Filter out empty topics
    num_topics = len(non_empty_topics)

    if num_topics == 0:
        print("No valid topics to display.")
        return

    # Determine number of rows dynamically
    cols = 3  # Fixed number of columns
    rows = (num_topics // cols) + (1 if num_topics % cols else 0)  # Adjust rows based on topics

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 5 * rows))

    # Flatten axes for easier iteration
    axes = axes.flatten() if num_topics > 1 else [axes]

    for i, ax in enumerate(axes):
        if i < num_topics:
            words = dict(non_empty_topics[i])  # Get words and their weights
            wordcloud = WordCloud(width=400, height=300, background_color='white').generate_from_frequencies(words)

            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title(f"Topic {i}", fontsize=14)
            ax.axis("off")
        else:
            ax.axis("off")  # Hide unused subplots

    plt.subplots_adjust(hspace=0.3)  # Reduce spacing between rows
    plt.tight_layout()
    plt.show()

# Plot a bar chart for the number of videos per topic
def plot_topic_distribution(topic_counts):
    topics, counts = zip(*sorted(topic_counts.items()))  # Get topic numbers and counts
    plt.figure(figsize=(12, 6))
    plt.bar(topics, counts, color='skyblue', edgecolor='black')
    plt.xlabel("Topic ID")
    plt.ylabel("Number of Videos")
    plt.title("Number of Videos Per Topic")
    plt.xticks(topics)  # Set topic labels on x-axis
    plt.show()


# In[4]:


all_documents = []
preprocessed_text = []
bigram_trigram_text = {}

for video_id, file_name in transcript_files:
    with open(os.path.join(transcripts_folder_path, file_name), 'r', encoding='utf-8') as file:
        content = file.read().lower()
        if len(content.split()) >= 100: 
            processed_text, bigram_trigram = preprocess_text(content, video_id)  # Get both processed text and segments
            preprocessed_text.append((video_id, processed_text))
            all_documents.append(processed_text)
            bigram_trigram_text[video_id] = bigram_trigram

# Create Dictionary and Corpus
dictionary = corpora.Dictionary([doc.split() for doc in all_documents])
corpus = [dictionary.doc2bow(doc.split()) for doc in all_documents]


# In[5]:


# Train LDA Model
lda_model_12 = LdaModel(corpus, num_topics=12, id2word=dictionary, alpha='auto', eta='symmetric', passes=100)

# Compute Coherence Score
coherence_model = CoherenceModel(model=lda_model_12, corpus=corpus, dictionary=dictionary, coherence='u_mass')
coherence_score = coherence_model.get_coherence()
print(f"Coherence Score: {coherence_score}")

# Print Topics
topics = lda_model_12.print_topics(num_words=20)
for topic_id, topic_words in topics:
    print(f"Topic {topic_id}: {topic_words}")

# Dictionary to store the count of videos per topic
topic_video_count = defaultdict(int)

# Mapping of video ID to its dominant topic
video_topic_mapping_12 = {}

for idx, doc_bow in enumerate(corpus):
    video_id = video_ids[idx]  # Get video ID
    topic_distribution = lda_model_12.get_document_topics(doc_bow, minimum_probability=0)

    # Get the most dominant topic (highest probability)
    dominant_topic = sorted(topic_distribution, key=lambda x: x[1], reverse=True)[0][0]
    
    # Store the mapping
    video_topic_mapping_12[video_id] = dominant_topic
    
    # Increase count for that topic
    topic_video_count[dominant_topic] += 1

# Print number of videos assigned to each topic
print("\nNumber of Videos per Topic:")
for topic, count in sorted(topic_video_count.items()):
    print(f"Topic {topic}: {count} videos")



# In[6]:


topic_to_videos = defaultdict(list)

video_topic_mapping_12 = {}

# probability threshold for assigning multiple topics
prob_threshold = 0.2

# Dictionary to store topic words for each video
video_topic_words_LDA1_12 = {}

for idx, doc_bow in enumerate(corpus):
    video_id = video_ids[idx]  # Get video ID
    topic_distribution = lda_model_12.get_document_topics(doc_bow, minimum_probability=0)

    # Get topics where probability is above threshold
    assigned_topics = [topic for topic, prob in topic_distribution if prob >= prob_threshold]
    video_topic_mapping_12[video_id] = assigned_topics  # Store assigned topics per video

    for topic in assigned_topics:
        topic_to_videos[topic].append(video_id)

    # Get the representative words for each assigned topic
    topic_words = []
    for topic in assigned_topics:
        words = [word for word, _ in lda_model_12.show_topic(topic, topn=30)]  # Get top 10 words
        topic_words.append(", ".join(words))  # Convert list to string

    # Store the topic words as a string
    video_topic_words_LDA1_12[video_id] = "; ".join(topic_words)  # Separate topics with `;`

# Count occurrences of each topic
topic_counts = Counter()

for topics in video_topic_mapping_12.values():
    for topic in topics:
        topic_counts[topic] += 1

# Print the number of videos per topic
print("\nNumber of Videos Per Topic:")
for topic, count in sorted(topic_counts.items()):
    print(f"Topic {topic}: {count} videos")

# Print topics assigned per video
print("\nTopics Assigned Per Video:")
for video_id, topics in video_topic_mapping_12.items():
    topic_list = ', '.join(map(str, topics)) if topics else "No dominant topic"
    print(f"Video ID: {video_id} → Topics: {topic_list}")

# Print videos per topic
print("\nTop Words Per Topic:")
num_words = 30  

for topic_id in sorted(topic_to_videos.keys()): 
    top_words = lda_model_12.show_topic(topic_id, num_words)
    words_str = ', '.join([word for word, prob in top_words])
    print(f"Topic {topic_id}: {words_str}")

topic_word_contributions = {}
for idx, doc_bow in enumerate(corpus):
    video_id = video_ids[idx]
    topic_distribution = lda_model_12.get_document_topics(doc_bow, minimum_probability=0)
    assigned_topics = [topic for topic, prob in topic_distribution if prob >= prob_threshold]
    video_topic_mapping_12[video_id] = assigned_topics
    for topic in assigned_topics:
        topic_to_videos[topic].append(video_id)
    topic_word_contributions[video_id] = {topic: lda_model_12.show_topic(topic, topn=30) for topic in assigned_topics}


# In[9]:

@st.cache_resource
def load_lda_model():
    return LdaModel(corpus, num_topics=12, id2word=dictionary, alpha='auto', eta='symmetric', passes=100)

lda_model_12 = load_lda_model()


topics = lda_model_12.print_topics(num_words=30)
# Streamlit UI
st.title("Topic Modeling Dashboard")
st.write(f"Coherence Score: {coherence_score}")

st.header("Topics Identified")
for topic_id, topic_words in topics:
    st.write(f"**Topic {topic_id}:** {topic_words}")

st.header("Video Topic Assignments")
selected_video = st.selectbox("Select a Video ID", video_ids)
if selected_video in video_topic_mapping_12:
    assigned_topics = video_topic_mapping_12[selected_video]
    st.write(f"Assigned Topics: {', '.join(map(str, assigned_topics))}")
    st.write("### Topic Word Contributions:")
    for topic in assigned_topics:
        words = ", ".join([word for word, _ in topic_word_contributions[selected_video][topic]])
        st.write(f"Topic {topic}: {words}")

@st.cache_data
def get_preprocessed_text():
    return dict(preprocessed_text)  # Avoid recomputation

preprocessed_text_dict = get_preprocessed_text()

@st.cache_data
def load_transcript(video_id):
    file_path = os.path.join(transcripts_folder_path, f"{video_id}_captions.txt")
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

selected_doc = st.selectbox("Select a Document", video_ids)

if selected_doc:
    st.write("### Original Transcript:")
    
    # Load transcript with caching
    original_text = load_transcript(selected_doc)
    
    with st.expander("View Original Transcript"):
        st.text_area("", original_text, height=200)
    
    with st.expander("View Preprocessed Text"):
        st.text_area("", preprocessed_text_dict[selected_doc], height=200)
    
    with st.expander("View Bigram/Trigram Words"):
        st.text_area("", ", ".join(bigram_trigram_text[selected_doc]), height=100)



# In[ ]:


#get_ipython().system('jupyter nbconvert --to script ILW.ipynb')

