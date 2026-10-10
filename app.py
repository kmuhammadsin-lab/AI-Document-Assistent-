import os
import streamlit as st
from dotenv import load_dotenv
from workflow import DocumentAssistant

load_dotenv()
st.set_page_config(page_title="Screen Document Assistant", page_icon="📄", layout="wide")
st.title("📄 Screen Document Assistant")
st.caption("Ask questions about PDF, TXT, and DOCX documents using hybrid search.")

api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key:
    st.warning("Add OPENAI_API_KEY to your local .env file, then restart the app.")

assistant = DocumentAssistant(api_key=api_key)
os.makedirs("documents", exist_ok=True)

with st.sidebar:
    st.header("Add documents")
    uploaded = st.file_uploader(
        "Upload PDF, TXT, or DOCX files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )
    if st.button("Process uploaded documents", type="primary"):
        if not uploaded:
            st.info("Choose at least one document first.")
        elif not api_key:
            st.error("OPENAI_API_KEY is missing.")
        else:
            progress = st.progress(0)
            for i, file in enumerate(uploaded):
                path = os.path.join("documents", os.path.basename(file.name))
                with open(path, "wb") as f:
                    f.write(file.getbuffer())
                try:
                    count = assistant.add_document(path)
                    st.success(f"{file.name}: processed into {count} chunks.")
                except Exception as e:
                    st.error(f"{file.name}: {e}")
                progress.progress((i + 1) / len(uploaded))

    st.divider()
    st.subheader("Local documents")
    local_files = sorted(
        f for f in os.listdir("documents")
        if f.lower().endswith((".pdf", ".txt", ".docx"))
    )
    if not local_files:
        st.caption("No documents in the documents/ folder yet.")
    for name in local_files:
        col1, col2 = st.columns([3, 1])
        col1.write(name)
        if col2.button("Delete", key=f"delete_{name}"):
            # Delete the source file and its cached chunks/embeddings.
            os.remove(os.path.join("documents", name))
            assistant.delete_document(name)
            st.rerun()

```python
st.subheader("Ask your documents")

question = st.text_input(
    "Question",
    placeholder="What does the document say about ...?"
)

if st.button("Ask", type="primary") and question.strip():

    if not api_key:
        st.error("Set OPENAI_API_KEY in your .env first.")

    else:
        with st.spinner("Searching documents and generating an answer..."):

            try:
                result = assistant.answer(question.strip())

                st.markdown("### Answer")
                st.write(result["answer"])

                st.caption(
                    f"Retrieved {len(result['sources'])} source chunk(s)."
                )

                st.markdown("### Sources")

                if not result["sources"]:
                    st.write("No relevant document chunks were found.")

                for i, source in enumerate(result["sources"], 1):
                    page = source["metadata"].get("page")
                    page_text = f" · Page {page}" if page else ""

                    with st.expander(
                        f"{i}. {source['metadata']['file_name']}{page_text}"
                    ):
                        st.write(source["text"])

                        st.caption(
                            f"Semantic score: {source['semantic_score']:.3f} | "
                            f"Keyword score: {source['keyword_score']:.3f}"
                        )

            except Exception as e:
                st.error(f"An error occurred: {e}")

st.markdown("---")
```

