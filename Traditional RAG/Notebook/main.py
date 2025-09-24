import os
import zipfile
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chains import RetrievalQA



load_dotenv() 

# Step 1: Setup LLM (Mistral-7B via Groq)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#Extract zip files from data folder
DATA_PATH='Data/'

def extract_zip_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.zip'):
            zip_path=os.path.join(folder_path, filename)
            with zipfile.ZipFile(zip_path, 'r')as zip_ref:
                zip_ref.extractall(folder_path)
                print(f'Extracted: {filename}')



extract_zip_files(DATA_PATH)


#Step-1: load all pdfs (even inside subfolders)

def load_pdf_files(data):
    loader=DirectoryLoader(data,
                           glob='*.pdf',
                           loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents

documents=load_pdf_files(data=DATA_PATH)
print('Length of PDF Pages', len(documents))

#Step: 2
def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(extracted_data=documents)
print('Length of Text Chunks', len(text_chunks))

#Step:3
def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embedding_model

embedding_model=get_embedding_model()


def load_llm():
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama3-70b-8192",  # You can also try "llama3-70b-8192" or "gemma-7b-it"
        temperature=0.5,
        max_tokens=512
    )
    return llm

# Step 2: Create Custom Prompt
CUSTOM_PROMPT_TEMPLATE = """

Your-custom-template

Context:
{context}

Question:
{question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

# Step 3: Load FAISS Vectorstore
DB_FAISS_PATH = 'vectorstore/db_faiss'
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
vectorstore=FAISS.from_documents(text_chunks, embedding_model)
vectorstore_retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
keyword_retriever=BM25Retriever.from_documents(text_chunks, embedding_model)
keyword_retriever.k=3
ensemble_retriever=EnsembleRetriever(retrievers=[vectorstore_retriever, keyword_retriever], weights=[0.5, 0.5] )

# Step 4: Create RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=ensemble_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Step 5: Ask a question
user_query = input("Write Query Here: ")
response = qa_chain.invoke({"query": user_query})

# Step 6: Print results
print("\nRESULT:\n", response["result"])
print("\nSOURCE DOCUMENTS:\n", response["source_documents"]) 