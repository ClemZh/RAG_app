from django.urls import path
from . import views

urlpatterns = [
    path('ask/', views.handle_question, name='handle_question'),
    path('', views.index, name='index'),  # New frontend path
]
