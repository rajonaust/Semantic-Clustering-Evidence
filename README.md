# Semantic-Clustering-Evidence
# SCOPE Extension: Semantic Chat Segmentation, Topic Discovery, and Topic Probability Estimation

## Overview

This project extends a conversation analysis pipeline by automatically segmenting chat messages into semantically coherent discussion blocks, extracting representative keywords for each segment, discovering higher-level topics across segments, and estimating topic membership probabilities.

The notebook implements a lightweight semantic workflow using sentence embeddings, keyword extraction, clustering, and adaptive threshold selection. It is designed for exploratory analysis of chat or conversation datasets stored in CSV format.

The overall goal is to transform a raw sequence of chat messages into structured semantic units that can later be used for topic tracking, conversation understanding, summarization, and downstream analytics.

---

## Key Features

- Automatic semantic segmentation of conversations into coherent chat segments
- Embedding-based similarity analysis using `SentenceTransformer`
- Keyword extraction from each segment using `KeyBERT`
- Topic discovery across segments using `HDBSCAN`
- Topic probability estimation for each segment
- Adaptive threshold generation for topic assignment using a manual knee-point detection method
- CSV-based input/output for easy experimentation

---

## Project Workflow

The notebook follows these main stages:

### 1. Install required packages
The notebook installs the required Python packages:
- `sentence_transformers`
- `pyspellchecker`
- `keybert`

### 2. Import libraries
The implementation uses:
- `pandas` for CSV loading and preprocessing
- `numpy` for numerical computation
- `math` for distance calculations
- `sentence_transformers` for semantic embeddings
- `sklearn.metrics.pairwise.cosine_similarity` for similarity computation
- `KeyBERT` for keyword extraction
- `HDBSCAN` for topic clustering

### 3. Load conversation data
The conversation dataset is loaded from a CSV file. The notebook expects a column named:

- `Text`

This column contains the chat messages in sequential order.

### 4. Generate semantic segments
The conversation is split into semantically coherent segments using an embedding-based change-point style logic. The notebook tracks how the average semantic distance evolves as messages are added to the current segment.

When the semantic change rate increases significantly, the algorithm identifies a possible segment boundary. A reverse-check and local refinement procedure are then used to find a better split point.

The resulting segment index for each message is saved into a CSV file.

### 5. Merge messages within each segment
After segmentation, all messages belonging to the same segment are concatenated into one text block. Each text block represents one semantic segment.

### 6. Extract keywords for each segment
For each segment, `KeyBERT` extracts representative keywords using:
- Maximal Marginal Relevance (`use_mmr=True`)
- Diversity control (`diversity=0.25`)
- English stop-word filtering

The extracted keywords are combined into a compact topic phrase for that segment.

### 7. Encode segment topic phrases
Each keyword-based segment phrase is embedded using the sentence transformer model. These embeddings are then used for topic discovery.

### 8. Discover topics with HDBSCAN
The segment embeddings are clustered using HDBSCAN with:
- `min_cluster_size=3`
- `min_samples=1`
- `metric="cosine"`

Each cluster is interpreted as a higher-level topic shared across multiple segments.

For each discovered topic, the notebook computes:
- Cluster ID
- Mean embedding of the cluster
- Segment content belonging to the cluster

### 9. Estimate topic probabilities
For every segment, the notebook computes cosine similarity against each discovered topic embedding. These similarity scores are then converted into probability-like values using a softmax-style formulation.

This gives a topic distribution for every segment.

### 10. Compute adaptive topic thresholds
For each topic, the notebook computes:
- Mean of topic scores
- Standard deviation of topic scores
- Threshold candidates of the form:

\[
\text{threshold} = \mu + k \sigma
\]

A manual knee-detection method is used to select an appropriate `k`. The final threshold can later be used to determine whether a segment strongly belongs to a topic.

---

## Methodology

### Semantic Segmentation
The segmentation stage uses sentence embeddings to measure semantic drift across adjacent messages. Instead of fixed-length segmentation, the notebook tries to detect natural semantic boundaries based on changes in embedding distance.

This makes the segmentation more context-aware than simple message-count or time-window approaches.

### Keyword-Based Segment Representation
Once segments are formed, the notebook compresses each segment into a compact phrase using extracted keywords. This helps reduce noise while preserving semantic meaning.

### Topic Discovery
The keyword-based segment embeddings are clustered using HDBSCAN, a density-based clustering method that can identify variable-sized topic groups and naturally handle noise.

### Topic Probability Modeling
Each segment is compared against the discovered topics using cosine similarity, and the similarity vector is normalized into a probability-like distribution.

### Adaptive Thresholding
Rather than using a fixed threshold for all topics, the notebook computes topic-specific thresholds based on the score distribution of each topic and a manually detected knee point.

---

## Input Format

The notebook expects a CSV file named something like:

`conversation.csv`

with at least the following column:

| Column | Description |
|--------|-------------|
| Text | The chat message text |

Example:

```csv
Text
Hello, how are you?
I am doing well, thanks.
Can we discuss the project timeline?
Sure, let us start with the first milestone.
