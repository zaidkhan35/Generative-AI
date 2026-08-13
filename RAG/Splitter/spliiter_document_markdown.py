# Language -> lets the splitter know what TYPE of content it's splitting
# (Markdown, Python, JS, etc.) so it splits at logical boundaries for that format
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

# Sample Markdown text (like a README file) with headers, lists, and a code block
text = """
# Project Name: Smart Student Tracker
A simple Python-based project to manage and track student data, including their grades, age, and academic status.
## Features
- Add new students with relevant info
- View student details
- Check if a student is passing
- Easily extendable class-based design
## 🛠 Tech Stack
- Python 3.10+
- No external dependencies
## Getting Started
1. Clone the repo  
```bash
   git clone https://github.com/your-username/student-tracker.git
"""

# Initialize the splitter, but built specifically for Markdown structure
# from_language() gives it markdown-specific rules:
# it tries to split on headers (#, ##), then lists, then code blocks,
# instead of just splitting on generic paragraphs/sentences
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=200,
    chunk_overlap=0,
)

# Split the markdown text into logical chunks
chunks = splitter.split_text(text)

# Check how many chunks we got
print(len(chunks))

# Print the first chunk - likely the title + intro paragraph, kept together
print(chunks[0])
```

## Why this matters — language-aware splitting

A normal `RecursiveCharacterTextSplitter` doesn't know that `#`, `##`, or code blocks (` ``` `) have special meaning — it just sees them as regular characters. This could cut a chunk mid-code-block or separate a header from its content awkwardly.

`Language.MARKDOWN` tells the splitter: "headers and code blocks are meaningful boundaries — try to split around them, not through them." This produces much cleaner, more logically grouped chunks for structured content.

## Other `Language` options available

`Language` isn't just for Markdown — LangChain supports splitting rules for many programming languages too, like `Language.PYTHON`, `Language.JS`, `Language.HTML`, `Language.CPP`, etc. Each one knows the syntax patterns specific to that language:
- `Language.PYTHON` tries to split at function/class definitions (`def`, `class`) rather than mid-function
- `Language.HTML` tries to split at tag boundaries

## Why this is useful for you specifically

Since you often work with code repositories and documentation (READMEs, Python files), this is directly useful if you ever build a RAG system over your **own GitHub projects or codebase** — e.g., "explain what this function does" — the splitter would keep each function/class intact as one chunk instead of cutting it in half, which makes for much more accurate retrieval and answers.
