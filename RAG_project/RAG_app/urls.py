from django.urls import path
from . import views
from .services import pdf_services, llm_clients, prompt_services


urlpatterns = [
    path('ask_pdf/', views.handle_question_streaming, name='handle_question_with_pdf'),
    path('ask/', views.handle_question, name = 'handle_question'),
    path('ask_streaming/', views.handle_question_streaming, name='handle_question_streaming'),
    path('',views.ChatAPP, name='chatbot'), #frontend page

    path('get_available_models/', llm_clients.get_available_models, name='get_list_models' ),
]
