from django.urls import path
from . import views

urlpatterns = [
    path('ask_pdf/', views.handle_question, name='handle_question_with_pdf'),
    path('ask/', views.handle_question, name = 'handle_question'),
    path('', views.index, name='index'),  # New frontend path
]
