from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain

documents = ["./exam_catalog.pdf"]

def load_documents(documents:list)->list:
    all_documents = []
    
    # for doc_path in documents:
    #     document_loader = PyPDFLoader(doc_path)
    #     loaded_document = document_loader.load() 
    #     all_documents += loaded_document 
    document_loader = PyPDFLoader(documents[0])
    loaded_document = document_loader.load()     
    print(type(load_documents))
    return loaded_document


def preprocess_uploaded_documents(docs: list):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    preprocess_docs = text_splitter.split_documents(docs)
    return [doc.page_content for doc in preprocess_docs] 


def load_embedding_model(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"):
    embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
            )
    return embedding_model
    

def create_vector_db(text_chunks, embedding_model):
    vector_store = FAISS.from_texts(
        texts=text_chunks,
        embedding=embedding_model
    )
    return vector_store


def ask_llm_with_retrievaled_doc(llm_model, retriever, query):
    qa_chain = RetrievalQAWithSourcesChain.from_llm(llm=llm_model, retriever=retriever)
    result = qa_chain({"question": query})
    return result


# Load documents and preprocess
docs = load_documents(documents)
preprocessed_docs = preprocess_uploaded_documents(docs)

# Load embedding model and create vector store
embedding_model = load_embedding_model()
vector_store = create_vector_db(preprocessed_docs, embedding_model)

# Set up HuggingFacePipeline for LLaMA 2
llm_pipeline = pipeline(
    "text-generation",
    model="meta-llama/Llama-2-7b-chat-hf",
    tokenizer="meta-llama/Llama-2-7b-chat-hf",
    device=-1,  # Use GPU if available,

    
)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

query = "welche themen sind wichtig für die ap2 prüfung? antworte auf deutsch?"
qa_chain = load_qa_chain(llm, chain_type="stuff")

retrieved_docs = vector_store.similarity_search(query, k=10)
# Create retriever from the vector store

response = qa_chain.run(input_documents=retrieved_docs, question=query)


print(response)