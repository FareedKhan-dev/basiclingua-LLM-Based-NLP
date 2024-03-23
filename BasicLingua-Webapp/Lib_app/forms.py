from django import forms

class TranslationForm(forms.Form):
    api_key = forms.CharField(label='API Key')
    user_input = forms.CharField(label='User Input')
    target_lang = forms.CharField(label='Target Language')
    
class TextReplacementForm(forms.Form):
    api_key = forms.CharField(label='API Key')
    user_input = forms.CharField(label='User Input')
    replacement_rules = forms.CharField(label='Replacement Rules')

class TextCorrectionForm(forms.Form):
    api_key = forms.CharField(label='API Key')
    user_input = forms.CharField(label='User Input')
    
class ExtractPatternForm(forms.Form):
    api_key = forms.CharField(label='API Key')
    user_input = forms.CharField(label='User Input')
    patterns = forms.CharField(label='Patterns')
    
# class Upocomingsoon(forms.Form):
    