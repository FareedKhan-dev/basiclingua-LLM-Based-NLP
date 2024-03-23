from basiclingua import BasicLingua


def Translate(api_key, user_input, target_lang):
    client = BasicLingua(api_key=api_key)
    translated_text = client.text_translate(user_input, target_lang)
    return translated_text

def Text_Replace(api_key, user_input, replacement_rules):
    client = BasicLingua(api_key=api_key)
    answer = client.text_replace(user_input, replacement_rules)
    return answer

def Text_Correction(api_key, user_input):
    client = BasicLingua(api_key=api_key)
    corrected_text = client.text_spellcheck(user_input)
    return corrected_text

def ExtractPattern(api_key, user_input, patterns):
    client = BasicLingua(api_key=api_key)
    extracted_patterns = client.extract_patterns(user_input, patterns)
    return extracted_patterns

