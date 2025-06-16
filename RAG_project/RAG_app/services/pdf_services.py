# import os
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OllamaEmbeddings

# def loading_and_embeddings_pdf_doc(path_to_pdf):
#     """
#     Loads a PDF, splits it into chunks, and embeds using a local embedding model.
#     """
#     # Load PDF
#     doc = PyPDFLoader(path_to_pdf).load()

#     # Split document
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=100,
#     )
#     split_doc = splitter.split_documents(doc)

#     start_time = time_measure()
#     embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
#     end_time = time_measure()
#     vector_store = FAISS.from_documents(split_doc, embeddings)
#     return vector_store, end_time-start_time
