"""
chatbot.py
----------
Interactive command-line chatbot for interrogating Prof. Devreotes' papers.

Usage:
    python chatbot.py

Special commands:
    /papers     — List all papers in the corpus
    /sources    — Show sources used for the last answer
    /memory     — Show current PCC memory status
    /reset      — Clear conversation history
    /help       — Show commands
    /quit       — Exit

Changes from original
---------------------
  ask() now returns a 4-tuple (answer, chunks, intent, memory_info).
  ask_stream() protocol normalised: REFUSAL arrives as a single
  (str, [], {}) yield; tokens arrive as (str, None, None); the final
  sentinel is (None, chunks, memory_info).
  All print() for diagnostic output replaced with logging.
  ask_stream() used for terminal output (real streaming, no fake delay).
"""

import os
import sys
import logging
import textwrap
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,                    # keep terminal clean for users
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from rag_engine import RAGEngine, REFUSAL_MESSAGE

WIDTH = 80


def hr(char: str = "-") -> None:
    print(char * WIDTH)


def print_wrapped(text: str, indent: int = 2) -> None:
    prefix = " " * indent
    for line in text.split("\n"):
        if line.strip():
            print(textwrap.fill(line, width=WIDTH - indent,
                                initial_indent=prefix, subsequent_indent=prefix))
        else:
            print()


def print_header() -> None:
    os.system("clear" if os.name == "posix" else "cls")
    hr("=")
    print("  GraphRAG Research Chatbot  (PCC Memory Enabled)")
    print("  Ask questions about the research papers")
    hr("=")
    print("  Commands: /papers  /sources  /memory  /reset  /help  /quit")
    hr("-")
    print()


def print_sources(chunks: list) -> None:
    hr("-")
    print(f"  Sources retrieved ({len(chunks)} chunks):\n")
    for i, c in enumerate(chunks, 1):
        authors = c.get("authors", [])
        author_str = ", ".join(authors[:2]) + (" et al." if len(authors) > 2 else "")
        print(f"  [{i}] {c.get('title','?')} ({c.get('year','?')}) — score: {c.get('score',0):.4f}")
        print(f"       {author_str}")
        print(f"       Chunk {c.get('chunk_index',0)+1}/{c.get('total_chunks',1)}")
        print()
    hr("-")


def print_memory_status(engine: RAGEngine) -> None:
    hr("-")
    status = engine.get_memory_status()
    if not status.get("pcc_enabled"):
        print("  PCC Memory: DISABLED")
    else:
        print(f"  PCC Memory: ENABLED")
        print(f"  User ID:           {status.get('user_id','?')}")
        print(f"  Conversation ID:   {status.get('conversation_id','?')}")
        print(f"  Short-term msgs:   {status.get('short_term_messages',0)}")
        summary = status.get("short_term_summary","")
        if summary:
            print(f"  Summary (preview): {summary[:120]}...")
    hr("-")


def print_papers(papers: list) -> None:
    hr("-")
    print(f"  Papers in corpus ({len(papers)}):\n")
    for p in papers:
        print(f"  * {p.get('year','?')} — {p.get('title','?')}")
        print(f"    {p.get('journal','?')}")
        authors = p.get("authors", [])
        print(f"    {', '.join(authors[:3])}{'  et al.' if len(authors) > 3 else ''}")
        if p.get("topics"):
            print(f"    Topics: {', '.join(p['topics'])}")
        print()
    hr("-")


def print_help() -> None:
    hr("-")
    print("""
  Available commands:

  /papers   Show all papers in the corpus
  /sources  Show source excerpts used in the last answer
  /memory   Show current PCC memory status
  /reset    Clear conversation history (start fresh)
  /help     Show this help message
  /quit     Exit

  Tips:
  * PCC Memory means follow-up questions "remember" earlier answers.
  * Ask about specific findings, methods, or molecular entities.
  * Use /memory to verify that context is being captured.
""")
    hr("-")


def main():
    print_header()

    try:
        engine = RAGEngine()
        engine.load()
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        print("  Run `python build_index.py` first.\n")
        sys.exit(1)
    except EnvironmentError as e:
        print(f"\n  ERROR: {e}\n")
        sys.exit(1)

    last_chunks: list = []
    print("  Ready! Type your question below.\n")

    while True:
        try:
            user_input = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye!\n")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("/quit", "/exit", "/q"):
            print("\n  Goodbye!\n")
            break
        elif cmd == "/help":
            print_help()
            continue
        elif cmd == "/papers":
            print_papers(engine.get_paper_list())
            continue
        elif cmd == "/sources":
            if last_chunks:
                print_sources(last_chunks)
            else:
                print("\n  No sources yet — ask a question first.\n")
            continue
        elif cmd == "/memory":
            print_memory_status(engine)
            continue
        elif cmd == "/reset":
            engine.reset_conversation()
            last_chunks = []
            print("\n  Conversation reset.\n")
            continue

        # RAG streaming query
        print()
        print(f"  {'Assistant':-<{WIDTH-2}}")
        try:
            print("  ", end="", flush=True)
            answer       = ""
            chunks: list = []
            memory_info  = {}

            for token, c, m in engine.ask_stream(user_input):
                if token is not None:
                    # Real Mistral streaming token
                    print(token, end="", flush=True)
                    answer += token
                elif c is not None:
                    # Final sentinel: (None, chunks, memory_info)
                    chunks      = c or []
                    memory_info = m or {}
                else:
                    # Early-exit REFUSAL: (str, [], {})
                    # This branch is hit when ask_stream yields (REFUSAL, [], {})
                    pass

            last_chunks = chunks
            print("\n")

        except Exception:
            logger.exception("Error during ask_stream.")
            print("\n  ERROR: see logs for details.\n")
            continue

        # Compact source attribution
        if answer and answer != REFUSAL_MESSAGE and chunks:
            unique_papers = {}
            for c in chunks:
                src = c.get("source", "")
                if src not in unique_papers:
                    unique_papers[src] = c.get("title", src)
            attrs = []
            for title in unique_papers.values():
                attrs.append(title[:45] + "..." if len(title) > 45 else title)
            print("  Sources: " + " | ".join(attrs))

        # Show PCC memory state (brief)
        if memory_info.get("pcc_enabled") and memory_info.get("short_term_messages", 0) > 0:
            print(f"  [PCC] {memory_info['short_term_messages']} msgs in memory"
                  + (f" | {memory_info.get('long_term_episodes',0)} long-term episodes"
                     if memory_info.get("long_term_episodes") else ""))

        print()
        hr("-")
        print()

    engine.close()


if __name__ == "__main__":
    main()
