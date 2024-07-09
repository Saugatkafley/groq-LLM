from langchain_core.pydantic_v1 import BaseModel, Field


class WikiInputs(BaseModel):
    """Inputs to the wikipedia tool."""

    query: str = Field(
        description="query to look up in Wikipedia, should be 3 or less words"
    )


class RAGInputs(BaseModel):
    """Inputs to the RAG tool."""

    query: str = Field(
        description="Use the context from the documents to answer the question. Use the best context.Try to be as specific as possible. Do not make up answers."
    )
