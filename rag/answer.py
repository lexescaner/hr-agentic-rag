"""
Policy answer generation.

Takes a user query, retrieves relevant chunks via rag.policy_rag.search(),
and calls an LLM (via OpenRouter, through LangChain's ChatOpenAI) to produce
a grounded, cited answer.

Uses the same ChatModel(ChatOpenAI) wrapper pattern as other projects
(real-estate-agent), pointed at OpenRouter's OpenAI-compatible endpoint.

This is where the three required RAG guardrails live, encoded directly in
the system prompt:
  1. Refuse/redirect out-of-corpus questions
  2. Limit unsupported claims (answer only from retrieved context)
  3. Distinguish policy facts from recommendations

Requires OPENROUTER_API_KEY in .env (same key used across other projects).
"""

import os
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from rag.policy_rag import search

load_dotenv()


class ChatModel(ChatOpenAI):
    """
    Creates a chat model from openrouter.ai using the OpenAI API.
    (Same wrapper pattern used in real-estate-agent.)
    """
    def __init__(
            self,
            model_name: str,
            openai_api_key: Optional[str] = None,
            openai_api_base: str = "https://openrouter.ai/api/v1",
            **kwargs: Any):
        openai_api_key = openai_api_key or os.getenv('OPENROUTER_API_KEY')
        super().__init__(
            openai_api_base=openai_api_base,
            openai_api_key=openai_api_key,
            model_name=model_name,
            **kwargs
        )


def get_model(model_name: Optional[str] = None) -> ChatModel:
    """
    Gets a reference to the answer-generation model.
    Falls back to LLM_MODEL from .env, then a known-working free OpenRouter
    model. Free-tier model availability rotates, so override via .env if
    this specific model gets delisted.
    """
    model_name = model_name or os.getenv("LLM_MODEL") or "google/gemma-4-26b-a4b-it:free"
    return ChatModel(
        model_name=model_name,
        max_tokens=512,
        temperature=0.2,  # low temperature: grounded, factual answers, not creative ones
    )


SYSTEM_PROMPT = """You are an HR policy assistant. You answer employee questions using ONLY the
policy excerpts provided to you as context below. Follow these rules strictly:

1. OUT-OF-CORPUS QUESTIONS: If the provided context does not contain information relevant to the
   question, say clearly that this isn't covered in the policy documents you have access to, and
   suggest the employee contact HR directly. Do NOT guess or use outside knowledge.

2. NO UNSUPPORTED CLAIMS: Every statement you make must be traceable to the provided context. Do
   not invent numbers, dates, or rules that are not explicitly present in the excerpts.

3. FACTS VS. RECOMMENDATIONS: Clearly distinguish between (a) policy facts stated directly in the
   documents, and (b) any suggestions or next steps you are recommending. Label recommendations
   explicitly, e.g. "Recommendation:" so the employee can tell the difference.

4. CITATIONS: After each factual claim, cite the source in the format [Document Title, Section].

Be concise and direct."""

PROMPT_TEMPLATE = ChatPromptTemplate([
    ("system", SYSTEM_PROMPT),
    ("human", "Policy context:\n\n{context}\n\n---\n\nEmployee question: {question}"),
])


def answer_question(query: str, top_k: int = 4, model: Optional[ChatModel] = None) -> dict:
    """
    Full RAG answer pipeline: retrieve -> build context -> call LLM -> return
    answer with the citations that were actually used.
    """
    retrieved = search(query, top_k=top_k)

    if not retrieved:
        return {
            "answer": (
                "I don't have any policy documents covering this topic. "
                "Please reach out to HR directly for guidance."
            ),
            "citations": [],
            "retrieved_chunks": [],
        }

    context_blocks = [
        f"[{r['doc_title']}, {r['section']}]\n{r['text']}" for r in retrieved
    ]
    context = "\n\n---\n\n".join(context_blocks)

    model = model or get_model()
    chain = PROMPT_TEMPLATE | model
    response = chain.invoke({"context": context, "question": query})

    citations = [
        {"doc_id": r["doc_id"], "doc_title": r["doc_title"], "section": r["section"]}
        for r in retrieved
    ]

    return {
        "answer": response.content,
        "citations": citations,
        "retrieved_chunks": retrieved,
    }


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "How many PTO days do I get per year?"
    result = answer_question(query)
    print("QUESTION:", query)
    print("\nANSWER:\n", result["answer"])
    print("\nCITATIONS USED:")
    for c in result["citations"]:
        print(f"  - {c['doc_title']} — {c['section']}")
