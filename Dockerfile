FROM python:3.10.12-slim-buster

# Set the working directory
WORKDIR /app

# # Copy the requirements and install them
# RUN pip install -r requirements.txt

#Running the packages installation
RUN pip install django==5.1.5 \
    langchain==0.3.14 \
    sentence-transformers==3.3.1 \
    faiss-cpu==1.9.0.post1 \
    pypdf==5.1.0 


# Expose port 8000 and run the app
EXPOSE 8000
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]