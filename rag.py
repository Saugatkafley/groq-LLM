# RAG
import os
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import Qdrant

embeddings = HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)


def load_pages(file_name):
    loader = PyPDFLoader(f"files/{file_name}")
    pages = loader.load_and_split()
    return pages


def split_pages(pages):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(pages)
    return chunks


def qdrant_retreiver(file_name: str):
    """similarity search using qdrant"""

    pages = load_pages(file_name)
    chunks = split_pages(pages)
    doc_store = Qdrant.from_documents(
        chunks,
        embeddings,
        collection_name=file_name,
        url=os.environ["QDRANT_URL"],
        prefer_grpc=False,
    )
    retriver = doc_store.as_retriever()
    return retriver
