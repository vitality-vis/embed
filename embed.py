import pandas as pd
import umap
import spacy
import numpy as np
from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.matutils import sparse2full
from sklearn.decomposition import TruncatedSVD
from typing import Dict, List
import json
import requests
import os
import re
import ast
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
import config
from langchain_openai import OpenAIEmbeddings


# Read Data
df_master = pd.read_csv(config.input_data_path, sep='\t')

# Drop NAs for the given columns
target_column_subset = ['title_processed', 'author', 'source', 'year', 'abstract_processed', 'keywords_processed']
df_subset = df_master[target_column_subset]
df_subset = df_subset.rename(columns={'title_processed': 'Title', 'author': 'Authors', 'source': 'Source', 'year': 'Year', 'abstract_processed': 'Abstract', 'keywords_processed': 'Keywords'})

df_subset["Authors"] = df_subset["Authors"].apply(lambda x: ast.literal_eval(x) if type(x) == str else x)
df_subset["Keywords"] = df_subset["Keywords"].apply(lambda x: ast.literal_eval(x) if type(x) == str else x)
df_subset['AbstractLength'] = df_subset['Abstract'].astype(str).map(len)

# Run Baseline Embeddings (SIF + TF-IDF Weightings of GloVe)
## SIF = https://openreview.net/pdf?id=SyK00v5xx, ICLR 2017
df_subset = df_subset[df_subset["AbstractLength"]>50]
docs = [_ for _ in df_subset['Abstract']] # can modify for different minimum (or max) character length
nlp = spacy.load("en_core_web_sm", disable=("parser", "ner")) # "parser", "tagger", "ner"

print("Number of docs in dataset: " + str(df_subset.shape[0]))
print("Number of docs processed: " + str(len(docs))) # difference is dropped due to needing at least X characters; see docs above

## see https://github.com/PrincetonML/SIF
def compute_pc(X,npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_

def remove_pc(X, npc=1):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    pc = compute_pc(X, npc)
    if npc==1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX

# keep only alpha -- drop spaces, punctuation, stop words, and numbers
def keep_token(t):
    return (t.is_alpha and 
            not (t.is_space or t.is_punct or 
                 t.is_stop or t.like_num))

def lemmatize_doc(doc):
    return [t.lemma_ for t in doc if keep_token(t)]

# this is from http://dsgeek.com/2018/02/19/tfidf_vectors.html
def get_vectors(docs, svd_reduction = True):

    docs2 = [lemmatize_doc(nlp(doc)) for doc in docs] # use spaCy's lemma

    docs_dict = Dictionary(docs2)
    docs_dict.filter_extremes(no_below=5, no_above=0.2) # remove rare or common terms
    docs_dict.compactify()

    docs_corpus = [docs_dict.doc2bow(doc) for doc in docs2]
    model_tfidf = TfidfModel(docs_corpus, id2word=docs_dict)
    docs_tfidf  = model_tfidf[docs_corpus]
    docs_vecs   = np.vstack([sparse2full(c, len(docs_dict)) for c in docs_tfidf])

    to_vstack = [nlp(docs_dict[i]).vector for i in range(len(docs_dict))]
    if len(to_vstack) == 0:
        print("\n\nNot enough data for GloVe. Please add more training data.\n\n")
        return None

    tfidf_emb_vecs = np.vstack(to_vstack)
    docs_emb = np.dot(docs_vecs, tfidf_emb_vecs) 
    
    if svd_reduction:
        docs_emb = remove_pc(np.transpose(docs_emb),npc=1)
        docs_emb = np.transpose(docs_emb)

    return docs_emb

# nD embeddings
baseline_emb_300d = get_vectors(docs)
if baseline_emb_300d is None:
    baseline_emb_2d = None
else:
    # https://umap-learn.readthedocs.io/en/latest/parameters.html#n-neighbors
    # 2D embeddings
    baseline_emb_2d = umap.UMAP(
        # random_state=50, #42
        n_neighbors=4, #10
        min_dist=0.1, #0.1
        n_components=2, #2
        metric='cosine').fit(baseline_emb_300d) #cosine

# Set GloVe embeddings in the dataframe
df_subset["glove_embedding"] = [[float(i) for i in embed] for embed in baseline_emb_300d] if baseline_emb_300d is not None else None
df_subset["glove_umap"] = [[float(i) for i in embed] for embed in baseline_emb_2d.embedding_] if baseline_emb_2d is not None else None

print("GloVe done")

# Specter
tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
model = AutoModel.from_pretrained('allenai/specter')

def get_specter_embeddings(batch):
    inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
    result = model(**inputs)
    embeddings = result.last_hidden_state[:, 0, :]
    return list([list(i.detach().numpy()) for i in embeddings])

title_abs = []
for index, d in df_subset.iterrows():
    if d["Title"] is None or str(d["Title"]) == "nan" or d["Abstract"] is None or str(d["Abstract"]) == "nan":
        title_abs.append("")
    else:
        title_abs.append(d["Title"] + tokenizer.sep_token + d["Abstract"])

# nD embeddings
specter_embeddings = []
for i in range(0, len(title_abs), 128):
    batch = title_abs[i:i+128]
    specter_embeddings += get_specter_embeddings(batch)

# 2D embeddings
specter_emb_2d = umap.UMAP(
    # random_state=42, 
    n_neighbors=10, 
    min_dist=0.1, 
    n_components=2, 
    metric='cosine').fit(specter_embeddings)
specterUMap = [[float(i) for i in embed] for embed in specter_emb_2d.embedding_]

# Set Specter embeddings in the dataframe
df_subset["specter_embedding"] = specter_embeddings
df_subset["specter_umap"] = specterUMap
print("Specter done")

# ADA
ada_embedding_func = OpenAIEmbeddings(
    model='text-embedding-ada-002',
    openai_api_key=config.OPEN_AI_KEY
)

# nD embeddings
ada_embeddings = []
for title_ab in title_abs:
    ada_embeddings += [ada_embedding_func.embed_query(title_ab)]

# 2D embeddings
ada_emb_2d = umap.UMAP(
    # random_state=42, 
    n_neighbors=10, 
    min_dist=0.1, 
    n_components=2, 
    metric='cosine').fit(ada_embeddings)
adaUmap = [[float(i) for i in embed] for embed in ada_emb_2d.embedding_]


# Set ADA embeddings in the dataframe
df_subset["ada_embedding"] = ada_embeddings
df_subset["ada_umap"] = adaUmap
print("ADA done")


# Save/Export the dataframe
df_subset.to_csv(config.output_data_path, sep='\t')