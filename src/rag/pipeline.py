"""
RAG Pipeline Module
-------------------
Connects the retriever (ChromaDB) to a language model (GPT-4o-mini)
to answer questions grounded in SEC 10-K filings.

Key concepts:
- Prompt template: a structured message that tells the LLM exactly how to
  behave. Without it, the LLM answers from training data, not your documents.
- Grounding / faithfulness: the LLM's answer should only use facts from the
  retrieved chunks. This is measurable — Week 3 uses RAGAS to score it.
- LangChain Expression Language (LCEL): the | pipe syntax chains components.
  retriever | prompt | llm | output_parser is the full RAG pipeline.
  Each component's output becomes the next component's input.
- Source attribution: we return which chunks were used so answers are
  auditable — critical for finance where accuracy matters.
"""

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

# This is the most important part of the RAG chain.
# The {context} placeholder gets filled with retrieved chunks.
# The {question} placeholder gets filled with the user's query.
#
# Why "only use the context below"?
#   Without this constraint, GPT-4o-mini will blend its training knowledge
#   with retrieved facts, making it impossible to know which source an answer
#   came from. For finance RAG, you want answers traceable to specific filings.
#
# Why "If the answer is not in the context, say you don't know"?
#   Hallucination is a real risk with financial data. Telling the model to
#   admit ignorance is safer than a confident wrong number.

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


# ---------------------------------------------------------------------------
# Format retrieved chunks into a single context string
# ---------------------------------------------------------------------------

def format_docs(docs) -> str:
    """
    Concatenate retrieved Document chunks into a single context string.

    Each chunk is prefixed with its source metadata so the LLM knows
    which company and filing year the text comes from.

    Why include metadata in the context string?
        The LLM only sees the context string — it can't read Document
        metadata directly. Prefixing each chunk with [AAPL, 2025-10-31]
        lets the LLM correctly attribute its answers.
    """
    formatted = []
    for doc in docs:
        ticker = doc.metadata.get("ticker", "Unknown")
        date = doc.metadata.get("filing_date", "Unknown")
        formatted.append(f"[{ticker}, {date}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


# ---------------------------------------------------------------------------
# Build the RAG chain
# ---------------------------------------------------------------------------

def build_rag_chain(retriever):
    """
    Assemble the full RAG pipeline using LangChain Expression Language (LCEL).

    The chain works as follows:
      1. RunnableParallel runs two things at once:
         - "context": retrieves relevant chunks and formats them as a string
         - "question": passes the original question through unchanged
      2. PROMPT: fills {context} and {question} into the template
      3. LLM: sends the filled prompt to GPT-4o-mini
      4. StrOutputParser: extracts the text string from the LLM response object

    Why GPT-4o-mini and not GPT-4o?
        For RAG, the retriever does the heavy lifting of finding relevant info.
        The LLM just needs to read and summarise — a smaller model handles
        this well at a fraction of the cost (~10x cheaper than GPT-4o).

    Args:
        retriever: A LangChain retriever (from vector_store.get_retriever())

    Returns:
        A runnable chain. Call chain.invoke({"question": "..."}) to get answers.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,   # 0 = deterministic, no creativity. You want consistent
                         # factual answers for finance, not creative paraphrasing.
    )

    # RunnableParallel runs both branches simultaneously, then merges results
    # into a dict: {"context": "..formatted chunks..", "question": "..query.."}
    setup = RunnableParallel({
        "context": itemgetter("question") | retriever | format_docs,
        "question": itemgetter("question"),
    })

    chain = setup | PROMPT | llm | StrOutputParser()
    return chain


# ---------------------------------------------------------------------------
# Convenience wrapper with source attribution
# ---------------------------------------------------------------------------

def ask(chain, retriever, question: str) -> dict:
    """
    Ask a question and return both the answer and the source chunks used.

    Returns a dict with:
      - answer: the LLM's response string
      - sources: list of (ticker, filing_date, text_preview) tuples

    Why return sources separately?
        In Week 4's Streamlit UI, you'll display a "Sources" sidebar
        showing which chunks backed each answer. This is a key portfolio
        feature — it shows the system is auditable, not a black box.
    """
    answer = chain.invoke({"question": question})

    # Re-retrieve to get source metadata (the chain only returns the answer string)
    source_docs = retriever.invoke(question)
    sources = [
        {
            "ticker": doc.metadata.get("ticker", "?"),
            "filing_date": doc.metadata.get("filing_date", "?"),
            "preview": doc.page_content[:150].replace("\n", " "),
        }
        for doc in source_docs
    ]

    return {"answer": answer, "sources": sources}


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

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
    # Load from disk — no re-embedding needed
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
