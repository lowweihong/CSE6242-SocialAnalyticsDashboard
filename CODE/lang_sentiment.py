import pandas as pd
import torch
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import numpy as np
from tqdm import tqdm
import argparse

model_ckpt = "papluca/xlm-roberta-base-language-detection"
lang_tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
lang_model = AutoModelForSequenceClassification.from_pretrained(model_ckpt)

sent_tokenizer = BertTokenizer.from_pretrained("kk08/CryptoBERT")
sent_model = BertForSequenceClassification.from_pretrained("kk08/CryptoBERT")
sent_classifier = pipeline("sentiment-analysis", model=sent_model, tokenizer=sent_tokenizer, batch_size=16)
id2lang = lang_model.config.id2label

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove RT prefix
    pattern = r'^RT\s*@[\w]+:'
    # Remove the pattern and strip any extra spaces
    text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def pred_lang(text_ls):
    inputs = lang_tokenizer(text_ls, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        logits = lang_model(**inputs).logits

    preds = torch.softmax(logits, dim=-1)
    vals, idxs = torch.max(preds, dim=1)
    return [{id2lang[k.item()]: v.item()} for k, v in zip(idxs, vals)]

def process_chunk(chunk):
    chunk['clean_text'] = chunk['text'].map(clean_text)
    chunk = chunk[chunk['clean_text'].str.split(' ').str.len() > 1]
    chunk.reset_index(inplace=True, drop=True)

    # Map raw predictions to languages
    lang_pred = pred_lang(chunk['clean_text'].to_list())
    lang_pred_df = pd.DataFrame([(lang, val) for d in lang_pred for lang, val in d.items()],
                                columns=['lang', 'lang_score'])

    chunk = pd.concat([chunk, lang_pred_df], axis=1)
    chunk = chunk[chunk['lang'] == 'en']
    chunk.reset_index(inplace=True, drop=True)

    hashtag_pattern = r'#([A-Za-z0-9_]+)'
    chunk['topic'] = chunk['clean_text'].str.extractall(hashtag_pattern)[0].groupby(level=0).apply(list)
    chunk['topic'] = chunk['topic'].map(lambda x: '' if type(x) == float else ', '.join(x))

    sent_pred = sent_classifier([x if type(x) != float else '' for x in chunk['clean_text'].to_list()])
    sent_pred_df = pd.DataFrame(sent_pred)
    sent_pred_df.columns = ['sentiment_label', 'sentiment_score']
    chunk = pd.concat([chunk, sent_pred_df], axis=1)
    chunk['sentiment_label'] = chunk['sentiment_label'].map(lambda x: 'negative' if x == 'LABEL_0' else 'positive')

    chunk.reset_index(inplace=True, drop=True)

    return chunk


def main(fp='./tweets.csv',
         output_path=f'/content/drive/My Drive/DVA-spr2025/combined_df.csv'):
    # Define the chunk size and total rows to process
    chunk_size = 10000
    total_rows = len(pd.read_csv(fp, sep=';', usecols=[0]))
    # Step 1: Read the header from the first row of the CSV
    header = pd.read_csv(fp,
                         sep=';',
                         nrows=0)  # Only read the header
    column_names = header.columns.tolist()

    # List to store processed chunks
    processed_chunks = []

    # Calculate number of iterations for tqdm
    n_iterations = (total_rows - 1000) // chunk_size

    # Loop through the ranges with tqdm
    for start in tqdm(range(1000, total_rows, chunk_size),
                    total=n_iterations,
                    desc="Processing chunks"):

        # Read the specific range of rows
        chunk = pd.read_csv(fp,
                            sep=';',
                            skiprows=start,  # Start at this row
                            nrows=chunk_size,  # Read this many rows
                            names=column_names)

        # Process the chunk and append to list
        processed_chunk = process_chunk(chunk)
        processed_chunks.append(processed_chunk)

        # Concatenate all processed chunks into a single DataFrame
        data = pd.concat(processed_chunks, ignore_index=True)

    data.to_csv(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process files for topic modeling')
    parser.add_argument('--input_path', type=str, default='./tweets.csv',
                        help='Path to the csv files (default: ./tweets.csv)')
    parser.add_argument('--output_path', type=str, default='./combined_df.csv',
                        help='Filename of the processed csv files (default: ./combined_df.csv)')
    args = parser.parse_args()

    main(args.input_path, args.output_path)
