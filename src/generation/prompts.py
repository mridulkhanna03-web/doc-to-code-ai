"""Prompt templates for code generation, tutorial generation, query rewriting,
and regeneration after validation failure.

Design choices, informed by Anthropic's published Claude prompting guidance
(https://platform.claude.com/docs/en/build-with-claude/prompt-engineering)
and recent RAG literature on query rewriting:

1. **XML-tagged structure.** Claude is trained on structured input — wrapping
   parts of the prompt in ``<documentation>``, ``<question>``, ``<rules>``
   activates a pattern Claude (and most modern instruct-tuned models, qwen
   and Gemini included) handles measurably better than plain prose. Tag names
   are kept consistent across templates and referred to by name inside the
   instructions ("Use only APIs that appear in <documentation>...").

2. **Role in the system prompt; detail in the user prompt.** Anthropic's
   guidance is that instruction-following is stronger from the user message
   than the system message, so the system prompt sticks to *who you are* and
   *the absolute do-not-violate rules*, while the per-call user message
   carries the documentation context, the question, and the response shape.

3. **Documentation grounding + source citation.** The retriever supplies
   chunks as ``[Source N: doc / section]``. The user prompt asks the model to
   end the answer with ``Sources: [N, ...]`` so :mod:`parser` can extract the
   list mechanically.

4. **Anti-hallucination guardrails** stated as absolute rules and then
   referenced ("if <documentation> does not cover the question, say so").

5. **Deterministic output shape** — one fenced ``` ```python ``` `` block,
   2-4 sentences of explanation, then ``Sources: [...]`` on the final line.

6. **Query rewriting** — :data:`QUERY_REWRITER_*` templates turn a natural-
   language developer question into a documentation-style retrieval query.
   Recent surveys show simple LLM rewriting delivers the largest accuracy
   gain among query-transformation methods (HyDE, Query2Doc, step-back).
   We use Query2Doc-style augmentation in the retriever: embed the original
   AND the rewritten form, search with both, union the results.

7. **Regeneration prompt** feeds AST/ruff errors back to the LLM for the
   retry pass; the system prompt is reused unchanged so the absolute rules
   remain in force.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Code generation
# --------------------------------------------------------------------------- #

CODE_GENERATION_SYSTEM_PROMPT: str = """You are a precise Python coding assistant specialising in FastAPI.

You answer developer questions by writing a single, runnable Python code example, grounded in the FastAPI documentation context provided in <documentation> tags.

You follow every rule listed in <rules> below. Violating any rule is a failure mode.

<rules>
1. Use only imports, classes, decorators, and APIs that appear inside <documentation>. Do NOT invent or guess.
2. If <documentation> does not cover the question, write a one-line code comment saying so and produce a minimal scaffold rather than improvising.
3. Output exactly one fenced Python code block — open with ```python and close with ```.
4. After the code block, write 2-4 sentences of explanation in plain English. No bullet lists unless essential.
5. On the final line, list the source numbers you actually used, like: Sources: [1, 3]. If you used no sources, write: Sources: [].
6. Never use os.system, subprocess, eval, or exec. Never make network calls inside the example code.
7. Code must be self-contained and runnable from a single file. Include all imports.
</rules>

Optimise for: correctness > readability > brevity."""


CODE_GENERATION_USER_PROMPT: str = """<documentation>
{context}
</documentation>

<question>
{query}
</question>

Following the <rules> in the system prompt, answer the <question> using only what appears in <documentation>."""


# --------------------------------------------------------------------------- #
# Tutorial generation
# --------------------------------------------------------------------------- #

TUTORIAL_SYSTEM_PROMPT: str = """You are a technical writer producing FastAPI tutorials for working developers.

You produce one markdown document per topic, 800-1200 words, grounded entirely in the FastAPI documentation context provided in <documentation> tags. You follow every rule in <rules>.

<rules>
1. Start with a one-line H1 title formatted as: # How to <topic> in FastAPI
2. Open with one short paragraph (2-3 sentences) framing what the reader will learn and why it matters. No filler.
3. Include at least two working Python code blocks. Use fenced ```python``` blocks. Each block must be self-contained and runnable. Use only APIs that appear in <documentation>.
4. Walk through each code block in prose. Explain *why*, not just *what*.
5. Include a "## Common pitfalls" section with 2-3 specific things to avoid, each grounded in <documentation>.
6. End with a "## Sources" section listing the source numbers used, like: Sources: [1, 2, 4].
7. Do NOT include filler ("In this article we'll see..."). Get to the point.
8. Do NOT invent APIs or behaviour absent from <documentation>.
</rules>

Optimise for: correctness > clarity > depth."""


TUTORIAL_USER_PROMPT: str = """<documentation>
{context}
</documentation>

<topic>
{topic}
</topic>

Following the <rules> in the system prompt, write a complete tutorial for the <topic> using only what appears in <documentation>."""


# --------------------------------------------------------------------------- #
# Query rewriting
# --------------------------------------------------------------------------- #
# We rewrite the developer's natural-language question into a documentation-
# style retrieval query: phrasing that looks like the source markdown would,
# with FastAPI-specific terminology and API names surfaced. This brings the
# query vector closer to relevant chunks in the embedding space.
#
# Examples of the transformation:
#   "How do I make an endpoint that needs login?"
#       → "FastAPI authenticated endpoint OAuth2 password Bearer token dependency Depends"
#   "What's the FastAPI way of returning JSON?"
#       → "FastAPI JSONResponse response_model return dict path operation"

QUERY_REWRITER_SYSTEM_PROMPT: str = """You are a search-query rewriter for a FastAPI documentation retrieval system.

You translate developer questions into compact, technical search queries that resemble FastAPI documentation phrasing. Your goal is to surface the most relevant doc chunks in a vector search, not to answer the question.

<rules>
1. Output ONE line only. No prose, no quotes, no markdown.
2. Use FastAPI-specific terminology (path operation, dependency, Depends, response_model, BackgroundTasks, etc.) where applicable.
3. Preserve the user's intent — do not narrow or broaden the question.
4. Include API names, decorators, and class names that a FastAPI doc page about this topic would likely contain.
5. 6-15 words. Keywords over full sentences.
6. Do NOT include filler ("How to", "I want to", "Can you").
</rules>"""


QUERY_REWRITER_USER_PROMPT: str = """<question>
{query}
</question>

Rewrite <question> into a documentation-style search query following <rules>. Output the rewritten query as a single line. Nothing else."""


# --------------------------------------------------------------------------- #
# Regeneration after validation failure
# --------------------------------------------------------------------------- #

REGENERATION_USER_PROMPT: str = """Your previous answer failed code validation. Fix the specific errors and produce a new answer in the same format.

<previous_code>
```python
{previous_code}
```
</previous_code>

<validation_errors>
{errors}
</validation_errors>

Regenerate following the original <rules> in the system prompt. Address every line in <validation_errors>. Output: one fenced ```python``` block, 2-4 sentences of explanation, then `Sources: [...]` on the last line."""


# --------------------------------------------------------------------------- #
# Builders — small helpers so callers don't repeat the format/concat dance
# --------------------------------------------------------------------------- #

def build_code_prompt(context: str, query: str) -> tuple[str, str]:
    """Return ``(system, user)`` strings for code generation."""
    return (
        CODE_GENERATION_SYSTEM_PROMPT,
        CODE_GENERATION_USER_PROMPT.format(context=context, query=query),
    )


def build_tutorial_prompt(context: str, topic: str) -> tuple[str, str]:
    """Return ``(system, user)`` strings for tutorial generation."""
    return (
        TUTORIAL_SYSTEM_PROMPT,
        TUTORIAL_USER_PROMPT.format(context=context, topic=topic),
    )


def build_query_rewrite_prompt(query: str) -> tuple[str, str]:
    """Return ``(system, user)`` strings for query rewriting."""
    return (
        QUERY_REWRITER_SYSTEM_PROMPT,
        QUERY_REWRITER_USER_PROMPT.format(query=query),
    )


def build_regeneration_prompt(previous_code: str, errors: str) -> str:
    """Return only the user message for the retry pass.

    The system prompt is unchanged — the caller keeps the code-gen system
    prompt in place so the absolute rules still apply during regeneration.
    """
    return REGENERATION_USER_PROMPT.format(previous_code=previous_code, errors=errors)


def combine_for_ollama(system: str, user: str) -> str:
    """Combine ``system`` + ``user`` into a single prompt for Ollama's
    ``/api/generate`` endpoint (which takes one string, not a chat array).

    Uses ChatML-style markers (``<|im_start|>``/``<|im_end|>``), which is the
    template qwen2.5-coder is trained on. Other Ollama models tolerate it.
    When we eventually switch to ``/api/chat`` we can pass them as separate
    messages and drop this function.
    """
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
