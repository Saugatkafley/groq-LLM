# RAG
import os
from typing import Union
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import Qdrant

embeddings = HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

def qdrant_retreiver(file_names: Union[str, list]):
    """similarity search using qdrant"""
    all_docs = []
    for f in file_names:
        pages = PyPDFLoader(f).load_and_split()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        ).split_documents(pages)
        all_docs.extend(chunks)
    doc_store = Qdrant.from_documents(
        chunks,
        embeddings,
        collection_name="RAG_PDFs",
        url=os.environ["QDRANT_URL"],
        prefer_grpc=False,
    )
    retriver = doc_store.as_retriever()
    return retriver
