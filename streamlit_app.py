import streamlit as st
import requests
import json
from langchain import hub
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, pipeline

st.set_page_config(layout="wide")
st.title("UPM IRS",anchor=False)
st.markdown(
    "*Search through the UPM IRS' Computer Science thesis database.*"
)
st.write("This search engine allows for keyword queries or natural language queries." \
" Currently there are only Computer Science theses in the system. These theses span across topics like applications in the medical field, " \
"information systems and Computer Science concepts.")
st.write('Some sample queries could be: "Theses about information management systems", "What is an Epitope", "How to perform the CYK algorithm".')
st.markdown("Notes: \n - The system may hallucinate (provide wrong factual information), so be careful! If ever it hallucinates please take note of that since it is also something I'm trying to assess in the survey later. \n- The system excels at topics where there are many papers on the topic or concept, otherwise it will struggle (i.e. hallucinate) on topics that do not have any documents within the current system. \n- The primary purpose of this system is to be a search engine and NOT a chatbot. It will not remember the previous questions you ask it, nor can I guarantee its responses to chatbot question be what you expect.")
st.divider()

from pinecone.grpc import PineconeGRPC as Pinecone

pc = Pinecone(api_key="pcsk_5wMDLk_7Jd3cRhkX49Lhgre8APWNgaD1fAe6xUg8ZzC8d5BQSR8KdbdW7qyE94nTbd5D1e")

index = pc.Index(host="https://cas-pdfs-1h52h4a.svc.aped-4627-b74a.pinecone.io")

API_URL = "https://snfbddx96bbfouqa.us-east-1.aws.endpoints.huggingface.cloud"
headers = {
	"Accept" : "application/json",
	"Authorization": "Bearer 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",
	"Content-Type": "application/json" 
}

def api_call(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    retrieved: int

def retrieve(state: State):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    user_input = state["question"]

    score_cutoff=0.35

    embedded_input = model.encode(user_input)

    results = index.query(
        namespace="cs_pdf_data",
        vector=embedded_input,
        top_k=50,
        include_values=False,
        include_metadata=True
    )

    unique_doc_ids = []
    unique_docs = []
    for result in results.get("matches"):
        doc_id = result.id.split("#")[0]
        if doc_id not in unique_doc_ids:
            unique_doc_ids.append(doc_id)
            unique_docs.append(result)

    final_documents = []

    for result in unique_docs:
        document = Document(
            page_content=result.metadata.get("text"),
            metadata={
                "title": result.metadata.get("title"), "author": result.metadata.get("author"),
                "month": result.metadata.get("month"), "year": result.metadata.get("year"),
                "id": result.id.split("#")[0], "score": result['score']}
        )
        final_documents.append(document)

    filtered_results = [result for result in final_documents if result.metadata.get("score") >= score_cutoff]
    print("Found "+str(len(final_documents))+" documents. Only "+str(len(filtered_results))+" pass the score cutoff.")
    return {"context": filtered_results, "retrieved": len(filtered_results)}

def generate(state: State):
    if len(state["context"]) == 0: 
        return {"answer": "Sorry, I could not find any papers in my data :("}
    
    given_context = state["context"][:5]
    docs_content = "".join(f"{i}. {doc.page_content}\n" for i, doc in enumerate(given_context, start=1))

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    chat_prompt = tokenizer.apply_chat_template([
        {"role": "system", "content": "Answer the question below with the given context. After answering the question, give a quick overview of the retrieved documents."},
        {"role": "user", "content": "Question: "+state["question"] + "\n\nContext:\n" + docs_content}
    ], tokenize=False, add_generation_prompt=False)

    response = api_call({
        "inputs": f"{chat_prompt}",
        "parameters": {
            "max_new_tokens": 500,
            "return_full_text": False
	    }
    })
    return {"answer": response}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

col1, col2 = st.columns((1,1))
result_list = []
with col1:
    with st.form("search"):
        query = st.text_input("Search bar")
        submitted = st.form_submit_button("Submit")
        if submitted:
            response = graph.invoke({"question": query})
            try:
                if len(response["context"]) > 0:
                    st.markdown((response["answer"][0]["generated_text"].removeprefix("assistant")).strip())
                else:
                    st.text(response["answer"])
            except:
                st.error("LLM server is down! Please contact Tim about server availability.\nError Message:\n"+response["answer"]["error"])
            result_list = response["context"]

with col2:
    st.text("Found " + str(len(result_list)) + " documents!")
    for doc in result_list:
        st.text(doc.metadata.get("score"))
        st.markdown("**"+doc.metadata.get("title").rstrip() + "**")
        st.text("by " + doc.metadata.get("author") + " | " + doc.metadata.get("month") + " " + str(doc.metadata.get("year")).replace(".0",""))
        with st.expander("Check document snippet"):
            st.text('...'+doc.page_content+'...')
        st.divider()