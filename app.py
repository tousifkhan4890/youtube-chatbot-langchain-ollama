import base64
import re
from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ---------- Page setup (must be the first Streamlit call) ----------
st.set_page_config(page_title="YouTube ChatBot", page_icon="🎬", layout="centered")


# ---------- Helpers ----------
def clean_youtube_url(text: str):
    """Extract the 11-character video ID from any YouTube link and rebuild a clean URL."""
    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})", text)
    return f"https://www.youtube.com/watch?v={match.group(1)}" if match else None


def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


# ---------- Background + styling ----------
def set_background(image_path: str):
    if Path(image_path).exists():
        data = base64.b64encode(Path(image_path).read_bytes()).decode()
        bg = (
            "linear-gradient(rgba(0,0,0,0.25), rgba(0,0,0,0.45)), "
            f'url("data:image/jpg;base64,{data}")'
        )
    else:  # fallback if the picture is missing
        bg = "linear-gradient(135deg, #0b1020, #1a2440)"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: {bg};
            background-size: cover;
            background-position: right center;
            background-attachment: fixed;
        }}
        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stSidebar"] {{
            background: rgba(8,12,28,0.85);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255,255,255,0.08);
        }}
        [data-testid="stChatMessage"] {{
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 14px;
            padding: 14px;
            backdrop-filter: blur(6px);
        }}
        .title {{
            font-size: 2.3rem; font-weight: 700; text-align: center;
            background: linear-gradient(90deg, #4F8BF9, #9b6bff);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 0;
        }}
        .subtitle {{
            text-align: center; color: #b8c0d4; margin-bottom: 1.5rem;
        }}
        .notice {{
            background: rgba(245,158,11,0.12);
            border: 1px solid rgba(245,158,11,0.45);
            border-radius: 10px;
            padding: 10px 12px;
            margin-bottom: 14px;
            font-size: 0.82rem;
            line-height: 1.5;
            color: #f3e3c3;
        }}
        .notice-title {{
            font-weight: 700;
            color: #fbbf24;
            margin-bottom: 4px;
        }}
        .author {{
            text-align: center;
            padding-top: 4px;
        }}
        .author-label {{
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #7c869c;
        }}
        .author-name {{
            font-size: 1rem;
            font-weight: 600;
            color: #7fa8ff;
        }}
        .loader-box {{
            display: flex; align-items: center; gap: 14px;
            background: rgba(79,139,249,0.10);
            border: 1px solid rgba(79,139,249,0.35);
            border-radius: 12px; padding: 14px 18px; color: #cfe0ff;
        }}
        .spinner {{
            width: 22px; height: 22px; flex-shrink: 0;
            border: 3px solid rgba(127,168,255,0.25);
            border-top-color: #7fa8ff; border-radius: 50%;
            animation: spin 0.9s linear infinite;
        }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
        .dots span {{ animation: blink 1.4s infinite both; }}
        .dots span:nth-child(2) {{ animation-delay: 0.2s; }}
        .dots span:nth-child(3) {{ animation-delay: 0.4s; }}
        @keyframes blink {{ 0%, 80%, 100% {{ opacity: 0.2; }} 40% {{ opacity: 1; }} }}
        .loader-title {{ font-weight: 600; }}
        .loader-sub {{ font-size: 0.8rem; color: #9aa3b8; }}
        .ready-box {{
            background: rgba(29,158,117,0.15);
            border: 1px solid rgba(29,158,117,0.5);
            border-radius: 12px; padding: 12px 18px; color: #9FE1CB;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


set_background("background.jpg")


# ---------- RAG pipeline (cached, so it is built only once per video) ----------
@st.cache_resource(show_spinner=False)
def build_chain(video_url: str):
    loader = YoutubeLoader.from_youtube_url(
        video_url, add_video_info=False, language=["en"]
    )
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    llm = ChatOllama(model="qwen2.5:7b", temperature=0.2)
    prompt = PromptTemplate(
        template="""
          You are a helpful assistant.
          Answer ONLY from the provided transcript context.
          If the context is insufficient, just say you don't know.

          {context}
          Question: {question}
        """,
        input_variables=["context", "question"],
    )

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })
    main_chain = parallel_chain | prompt | llm | StrOutputParser()
    return main_chain, retriever, len(chunks)


# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "video_url" not in st.session_state:
    st.session_state.video_url = None

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")

    st.markdown(
        """
        <div class="notice">
            <div class="notice-title">⚠️ Compatibility notice</div>
            Works only with YouTube videos that have
            <b>English captions (subtitles)</b>.
            Look for the <b>CC</b> button on the video player.
        </div>
        """,
        unsafe_allow_html=True,
    )

    url_input = st.text_input(
        "YouTube video URL",
        value="",
        placeholder="Paste a YouTube link here",
    )
    if st.button("Load video", use_container_width=True, type="primary"):
        cleaned = clean_youtube_url(url_input)
        if cleaned:
            st.session_state.video_url = cleaned
            st.session_state.messages = []
        else:
            st.error("That doesn't look like a YouTube link. Paste a full video URL.")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []

    st.divider()
    st.markdown(
        """
        <div class="author">
            <span class="author-label">Developed by</span><br>
            <span class="author-name">Tousif Khan</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- Header ----------
st.markdown('<p class="title">🎬 YouTube ChatBot</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Ask questions about any YouTube video, powered by local AI</p>',
    unsafe_allow_html=True,
)

# ---------- Main ----------
if not st.session_state.video_url:
    st.info("Enter a YouTube URL in the sidebar and click **Load video** to start.")
    st.stop()

status = st.empty()
status.markdown(
    """
    <div class="loader-box">
        <div class="spinner"></div>
        <div>
            <div class="loader-title">Analyzing video<span class="dots"><span>.</span><span>.</span><span>.</span></span></div>
            <div class="loader-sub">Getting everything ready. This may take a minute for long videos.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    chain, retriever, n_chunks = build_chain(st.session_state.video_url)
except Exception as e:
    status.empty()
    st.error(f"Could not load the video: {e}")
    st.stop()

status.markdown(
    '<div class="ready-box">Video ready. Ask your first question below.</div>',
    unsafe_allow_html=True,
)

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources used"):
                for i, s in enumerate(msg["sources"], 1):
                    st.markdown(f"**Source {i}:** {s}")

# Chat input
if question := st.chat_input("Ask something about the video..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = chain.invoke(question)
            sources = [d.page_content for d in retriever.invoke(question)]
        st.markdown(answer)
        with st.expander("📄 Sources used"):
            for i, s in enumerate(sources, 1):
                st.markdown(f"**Source {i}:** {s}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )