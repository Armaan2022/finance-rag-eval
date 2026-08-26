def _sentence_preview(text: str, min_chars: int = 150, max_chars: int = 400) -> str:
    """Return text up to the first sentence boundary after min_chars."""

    flat = text.replace("\n", " ").strip()
    if len(flat) <= min_chars:
        return flat
    window = flat[min_chars:max_chars]
    for punct in ".?!":
        idx = window.find(punct)
        if idx != -1:
            return flat[: min_chars + idx + 1].strip()
        
    # If no sentence end found, then cut at the last space to avoid incomplete sentences. 
    end = flat[:max_chars].rfind(" ")
    return flat[: end if end > min_chars else max_chars].strip()


# Connects the retriever to GPT-4o-mini. Answers are grounded in retrieved chunks only.

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI


# Prompt template
# {context} = retrieved chunks, {question} = user's query.
# "only use the context below" stops it from mixing in training knowledge.

SYSTEM_PROMPT = """You are a financial analyst assistant specializing in SEC filings.
Answer the user's question using ONLY the context provided below.
Do not use any outside knowledge or make up figures.
If the answer is not clearly stated in the context, say "I don't have enough information in the provided filings to answer that."

When answering:
- Be specific: include numbers, dates, and company names from the context
- Be concise: 2-4 sentences unless the question requires more detail
- Cite which company and filing year your answer comes from

Context:
{context}"""

HUMAN_PROMPT = "{question}"

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_PROMPT),
])


def format_docs(docs) -> str:
    """Prefix each chunk with ticker + date so the LLM knows which filing it's reading."""

    formatted = []
    for doc in docs:
        ticker = doc.metadata.get("ticker", "Unknown")
        date = doc.metadata.get("filing_date", "Unknown")
        formatted.append(f"[{ticker}, {date}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_rag_chain(retriever):
    """Build the RAG chain: retriever → format → prompt → LLM → string output."""

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0, 
    )

    # Run retrieval and pass the question through in parallel, then merge into one dict
    setup = RunnableParallel({
        "context": itemgetter("question") | retriever | format_docs,
        "question": itemgetter("question"),
    })

    chain = setup | PROMPT | llm | StrOutputParser()
    return chain



def ask(chain, retriever, question: str) -> dict:
    """Return the answer plus source metadata for the UI's source panel."""

    answer = chain.invoke({"question": question})

    # Retrieve to get source metadata (the chain only returns the answer string)
    source_docs = retriever.invoke(question)
    sources = [
        {
            "ticker": doc.metadata.get("ticker", "?"),
            "company_name": doc.metadata.get("company_name", ""),
            "filing_date": doc.metadata.get("filing_date", "?"),
            "form_type": doc.metadata.get("form_type", "10-K"),
            "section": doc.metadata.get("section", ""),
            "preview": _sentence_preview(doc.page_content),
        }
        for doc in source_docs
    ]

    return {"answer": answer, "sources": sources}



# Small test

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from src.ingestion.sec_edgar import ingest_tickers
    from src.processing.chunker import chunk_documents
    from src.retrieval.vector_store import build_vector_store, load_vector_store, get_retriever
    from pathlib import Path

    load_dotenv()

    CHROMA_DIR = Path(__file__).resolve().parents[2] / "chroma_db"

    print("=== Loading vector store ===")

    # Load from disk.
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        vector_store = load_vector_store()
        print("  Loaded from disk.")
    else:
        print("  Building from scratch...")
        documents = ingest_tickers(["AAPL", "MSFT"], max_filings_per_ticker=1)
        chunks = chunk_documents(documents, config_name="medium")
        vector_store = build_vector_store(chunks)

    retriever = get_retriever(vector_store, k=5)
    chain = build_rag_chain(retriever)

    print("\n=== Finance Q&A ===\n")

    questions = [
        "What was Apple's total net sales in fiscal year 2025?",
        "What are Microsoft's three main business segments?",
        "What risks does Apple identify related to its supply chain?",
        "How much did Microsoft spend on research and development?",
    ]

    for question in questions:
        print(f"Q: {question}")
        result = ask(chain, retriever, question)
        print(f"A: {result['answer']}")
        sources = [f"{s['ticker']} ({s['filing_date']})" for s in result['sources'][:2]]
        print(f"   Sources: {', '.join(sources)}")
        print()
