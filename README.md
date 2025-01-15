This app permits different users to interact with a llm model (in this case we use tinyllama because of the hardware limitations).
The app consists of one functionnality which is RAG. The user can give a pdf document and enquire information about said document. The llm will then give a response in accordance to the question asked by the user.

To do so, we have to follow the following steps:
1. Install the requirements
```bash
pip install django
pip install langchain
pip install sentence-transformers
pip install faiss-cpu
pip install pypdf
```

2. Run the app
Make sure that a port is available when running the app
```bash
python manage.py runserver
```

3. Build the docker image
```bash
docker build -t rag-app .
```

4. Use the app from local
Go to the following link: http://http://127.0.0.1:8000/
you will end up on a page where you can give the pdf document and ask a question.
The response will take time to be generated

5. Use the app from docker
```bash
docker run -p 8000:8000 rag-app
```

The docker image creation wasnt tested because i have problems with the installation of the requirements. The app was tested locally and works fine.
In the steps.txt document I detail the steps I followed to create the app.

