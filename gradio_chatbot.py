import os
import time
import gradio as gr
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

# ──────────────────────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore_dir = "vectorstore"
merged_db = None
for foldername in os.listdir(vectorstore_dir):
    folderpath = os.path.join(vectorstore_dir, foldername)
    if os.path.isdir(folderpath):
        db = FAISS.load_local(folderpath, embedding, allow_dangerous_deserialization=True)
        if merged_db is None:
            merged_db = db
        else:
            merged_db.merge_from(db)

# ──────────────────────────────────────────────────────────────────────────────
llm = LlamaCpp(
    model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_ctx=4048,
    temperature=0.7,
    top_p=0.95,
    n_threads=12,
    n_gpu_layers=0,
    verbose=True,  # Set to True for backend logs
)

template = """[INST]
You are a helpful assistant. Use the following context to answer the question elaborately with bullet points.
If the answer is not in the context, just say "I don't know".

Context:
{context}

Question:
{question}
[/INST]"""

prompt = PromptTemplate(input_variables=["context", "question"], template=template)
llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

# ──────────────────────────────────────────────────────────────────────────────
def chat(user_input, session_history):
    session_history.append(f"User: {user_input}")
    docs_with_scores = merged_db.similarity_search_with_score(user_input, k=10, score_threshold=1.0)

    filtered_docs = []
    for doc, score in docs_with_scores:
        print(f"📈 Score: {score:.4f} for document: {doc.metadata.get('source', 'unknown')}")
        filtered_docs.append(doc)

    summary = "\n".join(session_history[-5:])
    doc_context = "\n\n".join(doc.page_content for doc in filtered_docs)
    total_context = f"{summary}\n\n{doc_context}"
    total_context = " ".join(total_context.split()[:4048])  # trim to token limit

    print("\n📝 Context (first 500 chars):\n", total_context[:500], "...")

    result = llm_chain.invoke({
        "context": total_context,
        "question": user_input
    })

    assistant_response = result["text"]
    print("\n🤖 Mistral Response:\n", assistant_response)

    session_history.append(f"Assistant: {assistant_response}")
    chat_pairs = [(session_history[i], session_history[i+1]) for i in range(0, len(session_history)-1, 2)]
    return chat_pairs, session_history

# ──────────────────────────────────────────────────────────────────────────────
with gr.Blocks() as demo:
    gr.Markdown("### 🤖 Mistral 7B Chatbot (Local)")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Ask something", placeholder="Type your question and press Enter...")
    clear = gr.Button("Clear")

    session_state = gr.State([])

    msg.submit(chat, [msg, session_state], [chatbot, session_state])
    clear.click(lambda: ([], []), None, [chatbot, session_state])

demo.launch(inbrowser=True)
