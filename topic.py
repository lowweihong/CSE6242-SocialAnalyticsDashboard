import pandas as pd
import os
import transformers
import torch
import re
from torch import cuda
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, TextGeneration
from bertopic import BERTopic
from bertopic.cluster import BaseCluster
from sentence_transformers import SentenceTransformer

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Use only GPU 1

path = './res'

df_list = []

for file in os.listdir(path):
    df = pd.read_parquet(os.path.join(path, file))
    df_list.append(df)
    del df
    print(f'imported file: {file}')

# Combine all DataFrames into one
combined_df = pd.concat(df_list, ignore_index=True)
print(len(combined_df))
combined_df.drop_duplicates(subset=['clean_text'], inplace=True)
combined_df.reset_index(inplace=True, drop=True)
print(len(combined_df))
combined_df['retweets'] = combined_df['retweets'].astype(float).astype(int)

del df_list

combined_df = combined_df[combined_df['retweets']>1]
combined_df.reset_index(inplace=True, drop=True)
print(len(combined_df))

combined_df['remove_hashtag_text'] = combined_df['clean_text'].map(lambda x: re.sub(r'#\w+', '', x).strip())
combined_df = combined_df[combined_df['remove_hashtag_text']!='']
combined_df.reset_index(inplace=True, drop=True)
print(len(combined_df))


model_id = 'meta-llama/Llama-3.1-8B-Instruct'
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# Llama 2 Tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, token = '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')

# Llama 2 Model
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    device_map='auto',
    token = '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'
)
model.eval()

# Our text generator
generator = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    task='text-generation',
    temperature=0.1,
    max_new_tokens=500,
    repetition_penalty=1.1,
)

# Pre-calculate embeddings
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device='cuda')
embeddings = embedding_model.encode(combined_df['clean_text'], show_progress_bar=True)

# Train model and reduce dimensionality of embeddings
umap_model = UMAP(n_components=5, n_neighbors=15, random_state=42, metric="cosine", verbose=True)
reduced_embeddings = umap_model.fit_transform(embeddings)

hdbscan_model = HDBSCAN(min_samples=30, gen_min_span_tree=True, prediction_data=False, min_cluster_size=30, verbose=True)
clusters = hdbscan_model.fit(reduced_embeddings).labels_

system_prompt = """
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant for labeling topics.
<</SYS>>
"""

# Example prompt demonstrating the output we are looking for
example_prompt = """
I have a topic that contains the following documents:
- Bitcoin started as a decentralized digital currency but has grown into a speculative asset with significant energy demands.
- The process of mining Bitcoin consumes more electricity than some entire countries.
- Owning Bitcoin doesn’t inherently make you rich, nor does avoiding it make you poor.

The topic is described by the following keywords: 'bitcoin, crypto, mining, energy, blockchain, currency, transaction, wallet, decentralized, investment'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.

[/INST] Environmental impacts of Bitcoin mining
"""

# Our main prompt with documents ([DOCUMENTS]) and keywords ([KEYWORDS]) tags
main_prompt = """
[INST]
I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.
[/INST]
"""

prompt = system_prompt + example_prompt + main_prompt

# KeyBERT
keybert = KeyBERTInspired()

# MMR
mmr = MaximalMarginalRelevance(diversity=0.3)

# Text generation with Llama 2
llama = TextGeneration(generator, prompt=prompt, nr_docs=10,  # Increase to 10 documents (or more, depending on context length)
    diversity=0.05)
# All representation models
representation_model = {
    "KeyBERT": keybert,
    "llama": llama,
    "mmr": mmr
}

class Dimensionality:
  """ Use this for pre-calculated reduced embeddings """
  def __init__(self, reduced_embeddings):
    self.reduced_embeddings = reduced_embeddings

  def fit(self, X):
    return self

  def transform(self, X):
    return self.reduced_embeddings

      
umap_model = Dimensionality(reduced_embeddings)
hdbscan_model = BaseCluster()

# Fit BERTopic without actually performing any clustering
topic_model= BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        representation_model=representation_model,
        verbose=True
)
topics, probs = topic_model.fit_transform(combined_df['clean_text'], embeddings=embeddings, y=clusters)

topic_df = topic_model.get_topic_info()
topic_df.to_csv('topic_detail.csv', encoding='utf-8-sig', index=False)

combined_df['topic'] = topics
combined_df['topic_score']= probs
combined_df.to_csv('combined_df.csv', encoding='utf-8-sig', index=False)

topic_model.save("./model_dir", serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)
