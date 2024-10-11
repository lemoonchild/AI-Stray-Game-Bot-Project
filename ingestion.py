from dotenv import load_dotenv
import os
import sys
import io

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from consts import INDEX_NAME

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def ingest_docs():
    loader = ReadTheDocsLoader("stray.fandom.com", encoding='utf-8', patterns="*.html", custom_html_tag=('html', {}))
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} raw documents")
    print(raw_documents)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    print(f"Loaded {len(documents)} documents")
    for doc in documents:
        original_url = doc.metadata["source"]
        # Ensure the URL is corrected properly
        if "stray.fandom.com" not in original_url:
            new_url = "https://stray.fandom.com" + original_url.replace("\\", "/")
        else:
            new_url = original_url.replace("\\", "/").replace("stray.fandom.com", "https://stray.fandom.com")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to Pinecone")
    PineconeVectorStore.from_documents(
        documents, embedding=embeddings, index_name=INDEX_NAME
    )


if __name__ == "__main__":
    ingest_docs()
