from django.urls import path
from . import views

urlpatterns = [
    path('main_logic', views.main_logic)
]
