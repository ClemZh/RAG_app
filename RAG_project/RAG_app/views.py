from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
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
import requests
import logging
import ollama


#MODEL_OLLAMA = "hf.co/MaziyarPanahi/Chocolatine-3B-Instruct-DPO-Revised-GGUF:Q4_K_M"
EMBEDDING_MODEL_NAME = "all-minilm:l6-v2"

# Serve the frontend template
def ChatAPP(request):
    return render(request, 'ChatAPP.html')

def loading_and_embeddings_pdf_doc(path_to_pdf):
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
    start_time = time_measure()
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    end_time = time_measure()
    vector_store = FAISS.from_documents(split_doc, embeddings)

    return vector_store, end_time-start_time

@csrf_exempt
def handle_question_streaming(request):
    if request.method == 'POST':
        question = request.POST.get('question')
        model= request.POST.get('model-select')
        #print("azeopjazfpojapzjfvpojazfg\n",model)


        if not question:
            return JsonResponse({'error': 'Please provide a question'}, status=400)

        def stream_response():
            start_time = time.perf_counter()
            try:
                for chunk in ollama.chat(
                    model=model,
                    messages=[{"role": "user", "content": question}],
                    stream=True
                ):
                    content = chunk.get('message', {}).get('content', '')
                    yield content
            except Exception as e:
                yield f"\n[Error]: {str(e)}"

            end_time = time.perf_counter()
            total_time = end_time - start_time
            yield f"\n\n[TimeTaken]: {total_time:.2f} seconds"
        return StreamingHttpResponse(stream_response(), content_type='text/plain')

    return JsonResponse({'error': 'Invalid request method'}, status=405)


@csrf_exempt
def handle_question(request):
    """
    """
    if request.method == 'POST':
        question = request.POST.get('question')
        model= request.POST.get('model-select')
        #print("azeopjazfpojapzjfvpojazfg\n",model)

    if not question:
        return JsonResponse({'error': 'We couldn’t retrieve your question, please provide a question again'}, status=400)
    try:
        start_time = time.perf_counter()
        chat_completion = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": question
                }
            ],
            stream=False # Pas de streaming ici
        )

        end_time = time.perf_counter()
        total_time = end_time - start_time

        logging.debug("La question a bien été transmise au LLM par OLLAMA")

        # Récupération du contenu
        answer = chat_completion['message']['content']

        return JsonResponse({
            "answer": answer,
            "time_taken": f"{total_time:.2f} seconds"
         })

    except Exception as e:
        error_msg = f"Error in Ollama call: {str(e)}"
        logging.error(error_msg)
        return JsonResponse({'error': error_msg}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)


        

@csrf_exempt
def handle_question_with_pdf(request):
    if request.method == 'POST':
        # Get the uploaded file and question
        pdf_file = request.FILES.get('pdf')
        question = request.POST.get('question')
        model= request.POST.get('model-select')
        #print("azeopjazfpojapzjfvpojazfg\n",model)

        if not pdf_file or not question:
            return JsonResponse({'error': 'Please provide both a PDF and a question'}, status=400)

        # Save PDF to temporary path
        pdf_path = f'/tmp/{pdf_file.name}'
        with open(pdf_path, 'wb') as f:
            f.write(pdf_file.read())

        try:
            # Load embeddings
            vector_store, time_taken_for_embeddings = loading_and_embeddings_pdf_doc(pdf_path)
            print(f"Time taken for embedding of the relevant document = {time_taken_for_embeddings:.2f} seconds")

            # QA with model from Ollama
            retriever = vector_store.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(
                llm=Ollama(model=model),
                retriever=retriever,
                chain_type="stuff"
            )
            start_time=time_measure()
            answer = qa_chain.run(question)
            end_time = time_measure()

            time_taken_for_inference = end_time-start_time
            # Print to terminal
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f"Time:{time_taken_for_inference}")
            return JsonResponse({
                "answer": answer,
                "time_taken": f"{time_taken_for_inference:.2f} seconds"
            })

        finally:
            # Always clean up the file
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

    return JsonResponse({'error': 'Invalid request method'}, status=405)


#On va créer une fonction qui va mesurer le temps pris pour générer une réponse
def time_measure():
    return time.perf_counter()


#Embedding de manière natif ollama
def embedding_ollama(query):
    header = {
        "Content-Type": "application/json",
    }
    data = {
        'model':EMBEDDING_MODEL_NAME,
        'prompt': query
    }
    response = request.post(
        'http://localhost:11434/api/embeddings',headers=headers, json=data
    )
    return json.loads(response.text)