from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA

import os
import time
import request

# Serve the frontend template
def index(request):
    return render(request, 'index.html')


def loading_and_embeddings(path_to_pdf):
    """
    Loads a PDF, splits it into chunks, and embeds using a local embedding model.
    """
    # Load PDF
    doc = PyPDFLoader(path_to_pdf).load()

    # Split document
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    split_doc = splitter.split_documents(doc)

    # Embeddings (keep using SentenceTransformer or switch to Ollama if you have an embedding-capable model)
    embeddings = OllamaEmbeddings(model="all-minilm:l6-v2")
    vector_store = FAISS.from_documents(split_doc, embeddings)

    return vector_store

@csrf_exempt
def handle_question(request):
    if request.method == 'POST':
        question = request.POST.get('question')

        if not question:
            return JsonResponse({'error': 'We could\'t retrieve your question, please provide a question again'}, status=400)
        

@csrf_exempt
def handle_question_with_pdf(request):
    if request.method == 'POST':
        # Get the uploaded file and question
        pdf_file = request.FILES.get('pdf')
        question = request.POST.get('question')

        if not pdf_file or not question:
            return JsonResponse({'error': 'Please provide both a PDF and a question'}, status=400)

        # Save PDF to temporary path
        pdf_path = f'/tmp/{pdf_file.name}'
        with open(pdf_path, 'wb') as f:
            f.write(pdf_file.read())

        try:
            # Load embeddings
            vector_store = loading_and_embeddings(pdf_path)

            # QA with model from Ollama
            retriever = vector_store.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(
                llm=Ollama(model="hf.co/MaziyarPanahi/Chocolatine-3B-Instruct-DPO-Revised-GGUF:Q4_K_M"),
                retriever=retriever,
                chain_type="stuff"
            )

            answer = qa_chain.run(question)

            # Print to terminal
            print(f"Question: {question}")
            print(f"Answer: {answer}")

            return JsonResponse({"answer": answer})

        finally:
            # Always clean up the file
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

    return JsonResponse({'error': 'Invalid request method'}, status=405)


#On va créer une fonction qui va mesurer le temps pris pour générer une réponse
def time_measure(){
    return time.perf_counter()
}


#Embedding de manière natif ollama
def embedding_ollama(query):
    data = {'model':f'{EMBEDDING_MODEL_NAME}','prompt': query}
    response = request.post(

    ) 