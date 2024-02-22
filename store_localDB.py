# creating local storage for embedding vector database from documents
# By: Reuben

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.vectorstores import Chroma


OPENAI_API_KEY = "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"  # to access your openai account


# input folder address for where you place all your documents
directory = "pdfs"


# load PDFs from folder directory
def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents


documents = load_docs(directory)


# tokenisation - splitting documents by token count
# default token model can only max split at size 384 tokens, consider changing from default model
# over between chunks to compensate if sentences or ideas within text get cut off due to chunking process
def split_docs(documents, chunk_size=384, chunk_overlap=50):
    text_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=chunk_overlap, tokens_per_chunk=chunk_size
    )
    docs = text_splitter.split_documents(documents)
    return docs


docs = split_docs(documents)


# converting split text chunks to embedding vectors and storing locally in chroma db vector store


def get_vectorstore(docs):
    # calling openai ada-002 emvbedding model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=OPENAI_API_KEY,
    )
    persist_directory = "chroma_db"  # folder name where vectordb will be stored
    vectordb = Chroma.from_documents(
        documents=docs, embedding=embeddings, persist_directory=persist_directory
    )

    # ensuring the database folder persists in working directory after the function is done
    vectordb.persist()
    return vectordb


vectorstore = get_vectorstore(docs)
