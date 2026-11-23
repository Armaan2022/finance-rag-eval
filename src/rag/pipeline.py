import re

# A sentence ends at . ? ! followed by whitespace or end of text, so decimals like "$ 7.5" don't count.
_SENTENCE_END_RE = re.compile(r"[.?!](?=\s|$)")


def _sentence_preview(text: str, min_chars: int = 150, max_chars: int = 400) -> str:
    """Return text up to the first sentence boundary after min_chars."""

    # Older chunks start with ". " (the splitter attached the separator to the next chunk), so strip it.
    flat = text.replace("\n", " ").strip().lstrip(".?! ")
    if len(flat) <= min_chars:
        return flat
    match = _SENTENCE_END_RE.search(flat, min_chars, max_chars)
    if match:
        return flat[: match.end()].strip()

    # If no sentence end found, then cut at the last space to avoid incomplete sentences. 
    end = flat[:max_chars].rfind(" ")
    return flat[: end if end > min_chars else max_chars].strip()


# Connects the retriever to GPT-4o-mini. Answers are grounded in retrieved chunks only.

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
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


def build_rag_chain():
    """
    Build the generation chain: {question, docs} → format → prompt → LLM → string.
    Retrieval is deliberately NOT inside the chain; ask() retrieves once and passes the docs in,
    so the answer, the sources panel and the eval all use the exact same chunks.
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    setup = RunnableParallel({
        "context": itemgetter("docs") | RunnableLambda(format_docs),
        "question": itemgetter("question"),
    })

    return setup | PROMPT | llm | StrOutputParser()


def source_info(doc) -> dict:
    """Metadata + preview for one chunk, as shown in the UI's sources panel."""

    return {
        "ticker": doc.metadata.get("ticker", "?"),
        "company_name": doc.metadata.get("company_name", ""),
        "filing_date": doc.metadata.get("filing_date", "?"),
        "form_type": doc.metadata.get("form_type", "10-K"),
        "section": doc.metadata.get("section", ""),
        "preview": _sentence_preview(doc.page_content),
    }


def ask(chain, retriever, question: str) -> dict:
    """
    Retrieve ONCE, then generate from those chunks.
    Returns the answer, the raw docs the model read (for evaluation) and their source metadata (for the UI).
    """

    docs = retriever.invoke(question)
    answer = chain.invoke({"question": question, "docs": docs})
    return {
        "answer": answer,
        "docs": docs,
        "sources": [source_info(doc) for doc in docs],
    }



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
        chunks = chunk_documents(documents)
        vector_store = build_vector_store(chunks)

    retriever = get_retriever(vector_store, k=5)
    chain = build_rag_chain()

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
