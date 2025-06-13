from django.urls import path
from . import views

urlpatterns = [
    path('ask_pdf/', views.handle_question_streaming, name='handle_question_with_pdf'),
    path('ask/', views.handle_question, name = 'handle_question'),
    path('ask_streaming/', views.handle_question_streaming, name='handle_question_streaming'),
    path('', views.index, name='index'),  # New frontend path
    path('chat/',views.ChatAPP, name='chatbot'), #second frontend page
]
