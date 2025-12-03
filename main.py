

import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA


def load_documents(docs_path: str = "docs"):
    """Load all .txt files from the docs directory."""
    folder = Path(docs_path)
    if not folder.exists():
        raise FileNotFoundError(
            f"Docs folder '{docs_path}' not found. Create it and add some .txt files."
        )

    docs = []
    for file in folder.glob("*.txt"):
        loader = TextLoader(str(file), encoding="utf-8")
        docs.extend(loader.load())
    if not docs:
        raise ValueError(f"No .txt files found in '{docs_path}'.")
    return docs


def build_vectorstore(documents):
    """Split documents into chunks and build a FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(chunks, embeddings)
    return vectordb


def build_qa_chain(vectordb):
    """Create a RetrievalQA chain on top of the vector store."""
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )
    return qa


def main():
    load_dotenv()

    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError(
            "OPENAI_API_KEY not found. Set it in a .env file or your environment."
        )

    print(" Loading documents from ./docs ...")
    docs = load_documents("docs")

    print(" Building FAISS vector index ...")
    vectordb = build_vectorstore(docs)

    print("🤖 Starting mini RAG assistant. Type 'exit' to quit.\n")
    qa = build_qa_chain(vectordb)

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Assistant: Bye! ")
            break

        result = qa({"query": query})
        answer = result["result"]
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
