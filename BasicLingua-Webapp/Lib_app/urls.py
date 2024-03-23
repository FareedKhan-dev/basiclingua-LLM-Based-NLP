from django.urls import path
from .views import *

urlpatterns = [
    path('text-translation/', translate_view, name='text-translation'),
    path('text-replace/', text_replace_view, name='text-replace'),
    path('text-correction/', text_correction_view, name='text-correction'),
    path('extract-patterns/', extract_pattern_view, name='extract-patterns'),
    
    
    
    
]
