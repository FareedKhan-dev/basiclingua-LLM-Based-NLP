from django.shortcuts import render
from .utils import *
from .forms import *
from django.http import JsonResponse

    
def translate_view(request):
    translated_text = None

    if request.method == 'POST':
        form = TranslationForm(request.POST)
        if form.is_valid():
            api_key = form.cleaned_data['api_key']
            user_input = form.cleaned_data['user_input']
            target_lang = form.cleaned_data['target_lang']
            translated_text = Translate(api_key, user_input, target_lang)  
            
            return JsonResponse({'translated_text': translated_text})
    else:
        form = TranslationForm()

    return render(request, 'Text_Trans.html', {'form': form, 'translated_text': translated_text})

def text_replace_view(request):
    answer = None

    if request.method == 'POST':
        form = TextReplacementForm(request.POST)
        if form.is_valid():
            api_key = form.cleaned_data['api_key']
            user_input = form.cleaned_data['user_input']
            replacement_rules = form.cleaned_data['replacement_rules']
            answer = Text_Replace(api_key, user_input, replacement_rules) 

            return JsonResponse({'answer': answer})
    else:
        form = TextReplacementForm()

    return render(request, 'Text_Replace.html', {'form': form, 'answer': answer})


def text_correction_view(request):
    corrected_text = None

    if request.method == 'POST':
        form = TextCorrectionForm(request.POST)
        if form.is_valid():
            api_key = form.cleaned_data['api_key']
            user_input = form.cleaned_data['user_input']
            corrected_text = Text_Correction(api_key, user_input) 

            return JsonResponse({'corrected_text': corrected_text})
    else:
        form = TextCorrectionForm()

    return render(request, 'Text_Correction.html', {'form': form, 'corrected_text': corrected_text})

def extract_pattern_view(request):
    extracted_patterns = None
    
    if request.method == 'POST':
        form = ExtractPatternForm(request.POST)
        if form.is_valid():
            api_key = form.cleaned_data['api_key']
            user_input = form.cleaned_data['user_input']
            patterns = form.cleaned_data['patterns']
            extracted_patterns = ExtractPattern(api_key, user_input, patterns) 

            return JsonResponse({'extracted_patterns': extracted_patterns})
    else:
        form = ExtractPatternForm()

    return render(request, 'Extract_Pattern.html', {'form': form, 'extracted_patterns': extracted_patterns})