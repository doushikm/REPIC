import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tree_sitter_language_pack import get_parser

MAX_BYTES = 1500

EXT_TO_LANG = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "tsx", ".java": "java", ".go": "go",
    ".rs": "rust", ".c": "c", ".h": "c", ".cpp": "cpp", ".cs": "csharp",
    ".php": "php", ".rb": "ruby", ".html": "html", ".css": "css",
}

fallback = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def spans(node, max_bytes):
    """Return (start, end) byte ranges, each small enough, cut at syntax boundaries."""
    if node.end_byte - node.start_byte <= max_bytes or not node.children:
        return [(node.start_byte, node.end_byte)]

    out, cur = [], None
    for child in node.children:
        size = child.end_byte - child.start_byte
        if size > max_bytes:                      # too big: flush, then go deeper
            if cur:
                out.append(cur)
                cur = None
            out.extend(spans(child, max_bytes))
        elif cur is None:
            cur = (child.start_byte, child.end_byte)
        elif child.end_byte - cur[0] <= max_bytes:  # still fits: merge into current chunk
            cur = (cur[0], child.end_byte)
        else:                                     # doesn't fit: close chunk, start a new one
            out.append(cur)
            cur = (child.start_byte, child.end_byte)
    if cur:
        out.append(cur)
    return out

def chunk_document(doc):
    path = doc.metadata["source"]
    lang = EXT_TO_LANG.get(os.path.splitext(path)[1].lower())

    if lang:
        try:
            src = doc.page_content.encode("utf-8")
            tree = get_parser(lang).parse(src)
            chunks = []
            for start, end in spans(tree.root_node, MAX_BYTES):
                text = src[start:end].decode("utf-8", errors="ignore")
                if not text.strip():
                    continue
                start_line = src[:start].count(b"\n") + 1
                end_line = start_line + text.count("\n")
                header = f"# File: {path} | lines {start_line}-{end_line}\n"
                chunks.append(Document(
                    page_content=header + text,
                    metadata={**doc.metadata, "start_line": start_line, "end_line": end_line},
                ))
            if chunks:
                return chunks
        except Exception:
            pass   # unknown language or parse problem: use the plain splitter

    return fallback.split_documents([doc])