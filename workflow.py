import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
from openai import OpenAI
from pypdf import PdfReader
from docx import Document

CACHE_DIR = Path("cache")
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


def extract_document(file_path):
    """Extract text from PDF, TXT, or DOCX; keep file name and page where available."""
    path = Path(file_path)
    ext = path.suffix.lower()
    pages = []

    if ext == ".pdf":
        reader = PdfReader(str(path))
        for page_number, page in enumerate(reader.pages, start=1):
            pages.append({"text": page.extract_text() or "", "page": page_number})
    elif ext == ".txt":
        pages.append({"text": path.read_text(encoding="utf-8", errors="replace"), "page": None})
    elif ext == ".docx":
        doc = Document(str(path))
        pages.append({"text": "\n".join(p.text for p in doc.paragraphs), "page": None})
    else:
        raise ValueError("Supported file types are PDF, TXT, and DOCX.")

    extracted = []
    for item in pages:
        text = item["text"].strip()
        if text:
            extracted.append({
                "text": text,
                "metadata": {"file_name": path.name, "page": item["page"]},
            })
    return extracted


def split_text(extracted_pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split extracted text into overlapping chunks while preserving metadata."""
    chunks = []
    for item in extracted_pages:
        text = item["text"]
        metadata = item["metadata"]
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({"text": chunk, "metadata": dict(metadata)})
            if end >= len(text):
                break
            start = max(0, end - overlap)
    return chunks


def keyword_score(query, text):
    """Simple normalized word-overlap score."""
    words = lambda s: set(re.findall(r"\b[a-zA-Z0-9_]+\b", s.lower()))
    q_words = words(query)
    if not q_words:
        return 0.0
    return len(q_words & words(text)) / len(q_words)


class DocumentAssistant:
    def __init__(self, api_key=None):
        self.client = OpenAI(api_key=api_key) if api_key else None
        CACHE_DIR.mkdir(exist_ok=True)
        self.chunks = []
        self._load_cache()

    def _cache_path(self, file_name):
        safe = hashlib.sha256(file_name.encode("utf-8")).hexdigest()[:24]
        return CACHE_DIR / f"{safe}.json"

    def _load_cache(self):
        self.chunks = []
        for path in CACHE_DIR.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.chunks.extend(data.get("chunks", []))
            except (json.JSONDecodeError, OSError):
                continue

    def add_document(self, file_path):
        if not self.client:
            raise ValueError("OPENAI_API_KEY is missing.")
        path = Path(file_path)
        cache_path = self._cache_path(path.name)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()

        # Avoid embedding a file again if its content has not changed.
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("digest") == digest:
                    self._load_cache()
                    return len(cached.get("chunks", []))
            except (json.JSONDecodeError, OSError):
                pass

        extracted = extract_document(path)
        chunks = split_text(extracted)
        if not chunks:
            raise ValueError("No readable text found. Scanned PDFs may need OCR.")

        # Embed chunks in batches and save vectors plus metadata for reuse.
        batch_size = 64
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=[c["text"] for c in batch],
            )
            for chunk, item in zip(batch, response.data):
                chunk["embedding"] = item.embedding

        cache_path.write_text(
            json.dumps({"digest": digest, "chunks": chunks}, ensure_ascii=False),
            encoding="utf-8",
        )
        self._load_cache()
        return len(chunks)

    def delete_document(self, file_name):
        cache_path = self._cache_path(file_name)
        if cache_path.exists():
            cache_path.unlink()
        self._load_cache()

    def hybrid_search(self, question, top_k=5):
        """Combine cosine semantic similarity and simple keyword matching."""
        if not self.client:
            raise ValueError("OPENAI_API_KEY is missing.")
        if not self.chunks:
            return []

        query_vector = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[question],
        ).data[0].embedding
        q = np.asarray(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q) or 1.0
        results = []

        for chunk in self.chunks:
            vector = np.asarray(chunk.get("embedding", []), dtype=np.float32)
            if vector.size != q.size:
                continue
            denom = (np.linalg.norm(vector) * q_norm) or 1.0
            semantic = float(np.dot(vector, q) / denom)
            keyword = keyword_score(question, chunk["text"])
            # Both scores contribute to the final ranking.
            combined = 0.7 * semantic + 0.3 * keyword
            results.append({
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "semantic_score": semantic,
                "keyword_score": keyword,
                "combined_score": combined,
            })

        return sorted(results, key=lambda x: x["combined_score"], reverse=True)[:top_k]

    def answer(self, question, top_k=5):
        sources = self.hybrid_search(question, top_k=top_k)
        if not sources:
            return {
                "answer": "I couldn't find relevant information in the loaded documents.",
                "sources": [],
            }

        context_parts = []
        for i, source in enumerate(sources, 1):
            meta = source["metadata"]
            location = f"File: {meta['file_name']}"
            if meta.get("page"):
                location += f", page {meta['page']}"
            context_parts.append(f"[Source {i} | {location}]\n{source['text']}")
        context = "\n\n".join(context_parts)

        # Keep model behavior in a constant system prompt.
        system_prompt = (
            "You are a document question-answering assistant. Answer using only the "
            "provided document context. If the context does not contain the answer, "
            "say: 'I couldn't find that information in the provided documents.' "
            "Do not invent facts. Be clear and concise. Refer to source file names "
            "and page numbers when they are available."
        )
        response = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question:\n{question}\n\nDocument context:\n{context}"},
            ],
            temperature=0.1,
        )
        return {"answer": response.choices[0].message.content, "sources": sources}
