from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import TokenTextSplitter
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
from tqdm import tqdm

from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os

# models
EMBEDDING_MODEL = "RPRTHPB-text-embedding-3-small"
GENERATION_MODEL = "RPRTHPB-gpt-5-mini"

# embedding parameters
TEXT_EMBEDDING_DIM = 1536
VECTOR_DB_INDEX_NAME = "ted-talks-embeddings"
VECTOR_DB_SIMILARITY_METRIC = "cosine"
VECTOR_DB_CLOUD = "aws"
VECTOR_DB_REGION = "us-east-1"
EMBEDDING_BATCH_SIZE = 100

# RAG parameters
OVERLAP_RATIO = 0.1 # range: 0-0.3
TOP_K = 10 # range: 1-30
CHUNK_SIZE = 2048 # limit 2048

# DataSet parameters
META_COLS = [
    "talk_id",
    "title",
    "speaker_1",
    "all_speakers",
    "occupations",
    "about_speakers",
    "views",
    "recorded_date",
    "published_date",
    "event",
    "native_lang",
    "available_lang",
    "comments",
    "duration",
    "topics",
    "related_talks",
    "url",
    "description",
]

# Generation model parameters
SYSTEM_PROMPT = "You are a TED Talk assistant that answers questions strictly and \
                 only based on the TED dataset context provided to you metadata \
                 and transcript passages. You must not use any external \
                 knowledge, the open internet, or information that is not explicitly \
                 contained in the retrieved context. If the answer cannot be \
                 determined from the provided context, respond: “I don’t know \
                 based on the provided TED data.” Always explain your answer \
                 using the given context, quoting or paraphrasing the relevant \
                 transcript or metadata when helpful."
                 
file_path = "ted_talks_en.csv"
ted_talks = pd.read_csv(file_path)

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# create index if it doesn't exist yet
if VECTOR_DB_INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=VECTOR_DB_INDEX_NAME,
        dimension=TEXT_EMBEDDING_DIM,
        metric=VECTOR_DB_SIMILARITY_METRIC,
        spec=ServerlessSpec(
            cloud=VECTOR_DB_CLOUD,
            region=VECTOR_DB_REGION,
        ),
    )

index = pc.Index(VECTOR_DB_INDEX_NAME)

def df_to_pinecone(df, index, emb_model, batch_size=100):
    """
    df: pandas DataFrame with at least 'talk_id' and 'chunk_text'
    index: pinecone.Index
    emb_model: OpenAIEmbeddings instance
    """
    meta_cols = [c for c in df.columns if c not in ("chunk_text",)]

    for start in tqdm(range(0, len(df), batch_size)):
        end = start + batch_size
        batch = df.iloc[start:end]

        # text used for embedding
        texts = batch["chunk_text"].tolist()

        # make sure IDs are unique per chunk
        # if you have a 'chunk_id' column, use that:
        # ids = batch["chunk_id"].astype(str).tolist()
        # otherwise, you can combine talk_id with the batch index:
        ids = (batch["talk_id"].astype(str) + "-" + batch.index.astype(str)).tolist()

        # other metadata (everything except chunk_text)
        metadatas = batch[meta_cols].to_dict(orient="records")

        # get embeddings via LangChain
        embeddings = emb_model.embed_documents(texts)   # list[list[float]]

        # build vectors and *also* add chunk_text into metadata
        vectors = [
            {
                "id": id_,
                "values": emb,
                "metadata": {
                    **meta,
                    "chunk_text": text,  # keep original chunk text in Pinecone
                },
            }
            for id_, emb, meta, text in zip(ids, embeddings, metadatas, texts)
        ]

        # upsert into Pinecone
        index.upsert(vectors=vectors)

chunk_overlap = int(CHUNK_SIZE * OVERLAP_RATIO)

text_splitter = TokenTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=chunk_overlap,
)

def make_chunks_df(df: pd.DataFrame, text_col="transcript") -> pd.DataFrame:
    """
    Creates a chunk-level dataframe from a TED Talks DF.
    Keeps all requested metadata fields.
    Replaces null / NaN metadata values with empty strings.
    """
    rows = []

    for _, row in df.iterrows():
        full_text = str(row.get(text_col, ""))
        chunks = text_splitter.split_text(full_text)

        for j, chunk in enumerate(chunks):
            # Normalize metadata: None / NaN -> ""
            chunk_metadata = {
                col: "" if pd.isna(row.get(col)) else str(row.get(col))
                for col in META_COLS
            }

            rows.append({
                "chunk_id": f"{row['talk_id']}-{j}",
                "row_id": row["talk_id"],
                "chunk_number": j,
                "chunk_text": chunk,
                **chunk_metadata,
            })

    return pd.DataFrame(rows)

# Create chunked DF:
chunks_df = make_chunks_df(ted_talks)

emb_model = OpenAIEmbeddings(model=EMBEDDING_MODEL,
                             api_key=os.environ["OPENAI_API_KEY"],
                             base_url=os.environ["LLMOD_BASE_URL"])

df_to_pinecone(chunks_df, index, emb_model, batch_size=EMBEDDING_BATCH_SIZE)