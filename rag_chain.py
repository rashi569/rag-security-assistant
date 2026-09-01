import os
from google import genai

from ingest import get_chroma_collection
from security.prompt_injection_detector import scan_with_defense_in_depth
from security.guardrails import build_prompt, check_response_for_leakage

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
TOP_K = 4 

if not os.environ.get("GEMINI_API_KEY"):
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Get a key from https://aistudio.google.com/apikey "
        "and set it, e.g.: export GEMINI_API_KEY=your-key-here"
    )

_client = genai.Client()

def retrieve_context(question: str, top_k: int = TOP_K) -> dict:
    """Retrieves the most relevant document chunks for a question."""
    collection = get_chroma_collection()
    results = collection.query(query_texts=[question], n_results=top_k)

    chunks = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []

    flagged_sources = [
        m["source"] for m, present in zip(metadatas, [True] * len(metadatas))
        if m.get("injection_flagged")
    ]

    context_text = "\n\n---\n\n".join(chunks) if chunks else "(no relevant documents found)"

    return {
        "context_text": context_text,
        "sources": [m.get("source", "unknown") for m in metadatas],
        "flagged_sources": list(set(flagged_sources)),
    }


def generate_answer(prompt: str) -> str:
    response = _client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text

def ask(question: str) -> dict:
    """
    Full pipeline for one question:
      1. Scan the incoming question itself for injection attempts
      2. Retrieve relevant document chunks
      3. Build a hardened prompt (context wrapped as data, not instructions)
      4. Generate an answer
      5. Scan the answer for signs of leakage/manipulation

    Returns a dict with the answer plus all security signals collected
    along the way, so the UI can surface warnings instead of hiding them.
    """
    query_scan = scan_with_defense_in_depth(question, use_ml_layer=True, use_llm_layer=True)

    retrieval = retrieve_context(question)
    prompt = build_prompt(retrieval["context_text"], question)
    answer = generate_answer(prompt)
    leakage_check = check_response_for_leakage(answer)

    flag_reasons = list(query_scan["regex_reasons"])
    if query_scan["ml_flagged"]:
        flag_reasons.append(f"ML classifier flagged (confidence={query_scan['ml_confidence']:.2f})")
    if query_scan["llm_flagged"]:
        flag_reasons.append("LLM layer flagged")

    return {
        "answer": answer,
        "sources": retrieval["sources"],
        "security": {
            "query_flagged": query_scan["flagged"],
            "query_flag_reasons": flag_reasons,
            "context_flagged_sources": retrieval["flagged_sources"],
            "response_leaked": leakage_check["leaked"],
            "response_leaked_phrases": leakage_check["leaked_phrases"],
        },
    }
