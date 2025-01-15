from django.shortcuts import render
from django.http import JsonResponse

from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.ollama import Ollama
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA

# Create your views here.


def loading_and_embeddings(path_to_pdf):
    """_summary_

    Args:
        path_to_pdf (string): path to the pdf document that we want to retrieve 

    Returns:
        vector_store: the vector store that we will be using for the retrieval
    """
    #We first load the PDF
    doc = PyPDFLoader(path_to_pdf).load()
    #We then split the document into paragraphs
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100,
    )
    split_doc = splitter.split_documents(doc)
    #We will be doing the embeddings 
    embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(split_doc, embeddings)

    return vector_store



def handle_question(request):
    """_summary_

    Args:
        request (request): the request that we are getting from the user

    Returns:
        JsonResponse: the response that we will be sending back to the user
    """
    question = request.GET.get("question")
    if not question:
        return JsonResponse({'error': 'Please provide a question'}, status=400)
    
    path_to_pdf = request.GET.get("path_to_pdf")
    vector_store = loading_and_embeddings(path_to_pdf)
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=Ollama(model="tinyllama"),
        retriever=retriever,
        chain_type="stuff"
    )
    answer = qa_chain.run(question)


    return JsonResponse({"answer": answer})

