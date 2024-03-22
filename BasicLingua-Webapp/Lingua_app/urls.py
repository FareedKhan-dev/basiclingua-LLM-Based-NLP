from django.urls import path
from .views import *

urlpatterns = [
    path('', Base, name='base'),
    
]
