from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ---------- Step 1: Indexing ----------
video_url = input("Enter the YouTube video URL: ")

loader = YoutubeLoader.from_youtube_url(
    video_url,
    add_video_info=False,   # avoids extra dependency/errors
    language=["en"],
)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
print(f"Total chunks: {len(chunks)}")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = FAISS.from_documents(chunks, embeddings)

# ---------- Step 2: Retrieval ----------
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# ---------- Step 3: Augmentation ----------
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

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

# ---------- Step 4: Chain + Generation ----------
parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
})

parser = StrOutputParser()
main_chain = parallel_chain | prompt | llm | parser

# ---------- Chat loop ----------
if __name__ == "__main__":
    print("\nYouTube ChatBot ready. Type 'exit' to quit.\n")
    while True:
        q = input("You: ")
        if q.lower() in ("exit", "quit"):
            break
        print("Bot:", main_chain.invoke(q), "\n")