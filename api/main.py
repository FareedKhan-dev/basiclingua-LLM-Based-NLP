from enum import Enum
from typing import Optional, List, Tuple, Any, Dict, Annotated, Union

from basiclingua import BasicLingua
from fastapi import FastAPI, Body, Header
from pydantic import BaseModel

from config import settings

app = FastAPI(title="BasicLingua API")


class StrListResult(BaseModel):
    results: Optional[List[str]]


class OptionalListResult(BaseModel):
    results: Optional[List[Any]]


class TextResult(BaseModel):
    results: Optional[str]


class TupleListResult(BaseModel):
    results: List[Tuple[str, str]]


class TupleResult(BaseModel):
    results: Tuple


class DictResult(BaseModel):
    results: Optional[Dict]


class BoolResult(BaseModel):
    results: bool = False


@app.post("/text/translate/{target_lang}", response_model=TextResult)
async def text_translate_wrapper(
        target_lang: str,
        api_key: Annotated[Union[str, None], Header()] = settings.gemini_api_key,
        user_input: str = Body(..., media_type="text/plain")
):
    """
    translate the given text into the target language.

    `Parameters`:
    1. user_input (str): The input sentence to be translated.
    Example: "The phone number is 123-456-7890."

    2. target_lang (str): The target language for translation.
    Example: "french".

    `Returns`:
    str: The translated text in the target language.
    """
    client = BasicLingua(api_key=api_key)
    translated_text = client.text_translate(user_input, target_lang)
    return TextResult(results=translated_text)


@app.post("/text/patterns/{pattern_params}", response_model=StrListResult)
async def detect_patterns(
        pattern_params: str,
        api_key: Annotated[Union[str, None], Header()] = settings.gemini_api_key,
        user_input: str = Body(..., media_type="text/plain")
):
    """
    Extracts patterns from the given input sentence.

    `Parameters`:
    1. user_input (str): The input sentence containing information to be extracted.
    Example: "The phone number is 123-456-7890."

    2. patterns (str): Comma-separated patterns to be extracted.
    Example: "email, name, phone number, address, date of birth".

    `Returns`:
        list: A list containing the extracted patterns. If no pattern is found, returns None.
    """
    client = BasicLingua(api_key=api_key)
    extracted_patterns = client.extract_patterns(pattern_params, user_input)
    return StrListResult(results=extracted_patterns)


@app.post("/text/replace/{replacement_rules}", response_model=TextResult)
async def text_replace_wrapper(
        replacement_rules: str,
        api_key: Annotated[Union[str, None], Header()] = settings.gemini_api_key,
        user_input: str = Body(..., media_type="text/plain")
):
    """
            Replace words in the original text according to the replacement rules provided.

            Parameters:
            1. user_input (str): The input sentence to be modified.
                Example: "I love Lamborghini, but Bugatti is even better. Although, Mercedes is a class above all."

            2. replacement_rules (str): A detailed prompt specifying the replacement rules.
                Example: "all mentioned cars with mehran but mercerdes with toyota"

            Returns:
            str: The modified text with replacements.
            """
    client = BasicLingua(api_key=api_key)
    replacement_result = client.text_replace(user_input, replacement_rules)
    return TextResult(results=replacement_result)


@app.post("/text/detect/ner/{ner_tags}", response_model=TupleListResult)
async def detect_ner_wrapper(
        ner_tags: Optional[str],
        api_key: Annotated[Union[str, None], Header()] = settings.gemini_api_key,
        user_input: str = Body(..., media_type="text/plain")
):
    """
    Perform Named Entity Recognition (NER) detection on the input text.

    Parameters:
    1. user_input (str): The input sentence to be modified.
        Example: "I love Lamborghini, but Bugatti is even better. Although, Mercedes is a class above all."

    2. ner_tags (str, optional): A comma-separated string specifying the NER tags.
        Example: "organization, date, time"
        Default: "person, location, organization, date, time, money, percent"

    Returns:
    list: A list of tuples containing the detected NER entities.
    """
    client = BasicLingua(api_key=api_key)
    ner_result = client.detect_ner(user_input, ner_tags)
    return TupleListResult(results=ner_result)


@app.post("/text/summarize/{summary_length}", response_model=TextResult)
async def text_summarize_wrapper(
        summary_length: Optional[str],
        api_key: Annotated[Union[str, None], Header()] = settings.gemini_api_key,
        user_input: str = Body(..., media_type="text/plain")
):
    """
            Generate a summary of the input text.

            Parameters:
            1. user_input (str): The input sentence to be summarized.
                Example: "I love Lamborghini, but Bugatti is even better"

            2. summary_length (str, optional): The length of the summary.
                Values (str): "short", "medium" or "long"
                Default: "short"

            Returns:
            str: The generated summary.
            """
    client = BasicLingua(api_key=api_key)
    summary = client.text_summarize(user_input, summary_length)
    return TextResult(results=summary)


@app.post("/text/qna/{question}", response_model=TextResult)
async def text_qna_wrapper(
        question: Optional[str],
        api_key: Annotated[Union[str, None], Header()] = settings.gemini_api_key,
        user_input: str = Body(..., media_type="text/plain")
):
    """
            answer the given question based on the input text.

            Parameters:
            1. user_input (str): The input sentence on which the question is based.
                Example: "OpenAI has hosted a hackathon for developers to build AI models. The event took place on 15th October 2022.\"

            2. question (str): question to be answered
                Example: "When did the event happen?"

            Returns:
            str: The generated summary.
            """
    client = BasicLingua(api_key=api_key)
    answer = client.text_qna(user_input, question)
    return TextResult(results=answer)


@app.post("/text/intent", response_model=StrListResult)
async def text_intent_wrapper(
        api_key: Annotated[Union[str, None], Header()] = settings.gemini_api_key,
        user_input: str = Body(..., media_type="text/plain")
):
    """
            Identify the intent of the user input.

            Parameters:
            1. user_input (str): The input sentence of which the intent is to be identified.
                Example: "OpenAI has hosted a hackathon for developers to build AI models."

            Returns:
            str: The identified intent.
            """
    client = BasicLingua(api_key=api_key)
    results = client.text_intent(user_input)
    return StrListResult(results=results)


class LemStemOp(str, Enum):
    stemming = "stemming"
    lemming = "lemmatization"


@app.post("/text/lemstem/{operation}", response_model=TextResult)
async def text_lemstem_wrapper(
        operation: LemStemOp,
        api_key: Annotated[Union[str, None], Header()] = settings.gemini_api_key,
        user_input: str = Body(..., media_type="text/plain")
):
    client = BasicLingua(api_key=api_key)
    results = client.text_lemstem(user_input, operation.value)
    return TextResult(results=results)


@app.post("/text/tokenize", response_model=StrListResult)
async def text_tokenize_wrapper(
        break_point: str = " ",
        api_key: Annotated[Union[str, None], Header()] = settings.gemini_api_key,
        user_input: str = Body(..., media_type="text/plain")
):
    client = BasicLingua(api_key=api_key)
    results = client.text_tokenize(user_input, break_point)
    return StrListResult(results=results)


class EmbedTaskType(str, Enum):
    RETRIEVAL_QUERY = "RETRIEVAL_QUERY"
    RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
    SEMANTIC_SIMULARITY = "SEMANTIC_SIMILARITY"
    CLASSIFICATION = "CLASSIFICATION"
    CLUSTERING = "CLUSTERING"


@app.post("/text/embed/{task_type}", response_model=OptionalListResult)
async def text_embed_wrapper(
        task_type: EmbedTaskType = EmbedTaskType.RETRIEVAL_DOCUMENT,
        api_key: Annotated[Union[str, None], Header()] = settings.gemini_api_key,
        user_input: str = Body(..., media_type="text/plain")
):
    """
            Perform stemming or lemmatization on the input text.

            Parameters:
            1. user_input (str): The input sentence to be processed.
                Example: "OpenAI has hosted a hackathon for developers to build AI models."

            2. type (str, optional): The type of text processing to be performed.
                Values (str): "stemming" or "lemmatization"
                Default: "stemming"

            Returns:
            str: The processed sentence.
            """
    client = BasicLingua(api_key=api_key)
    results = client.text_embedd(user_input, task_type.value)
    return OptionalListResult(results=results)


@app.post("/text/generate/{length}", response_model=TextResult)
async def text_generate_wrapper(
        length: Optional[str],
        api_key: Annotated[Union[str, None], Header()] = settings.gemini_api_key,
        user_input: str = Body(..., media_type="text/plain")):
    """
    Generate text based on the input text.

    Parameters:
    1. user_input (str): The input sentence to generate text from.
        Example: "I love Lamborghini, but Bugatti is even better"

    2. ans_length (str, optional): The length of the generated text.
        Values (str): "short", "medium" or "long"
        Default: "short"

    Returns:
    str: The generated text.
    """
    client = BasicLingua(api_key=api_key)
    results = client.text_generate(user_input, length)
    return TextResult(results=results)


@app.post("/text/detect_spam", response_model=DictResult)
async def detect_spam_wrapper(
        detect_classes: str = "spam, not_spam, unknown",
        explain: bool = True,
        api_key: Annotated[Union[str, None], Header()] = settings.gemini_api_key,
        user_input: str = Body(..., media_type="text/plain")
):
    """
            Perform spam detection on the input text.

            Parameters:
            1. user_input (str): The input sentence to perform spam detection on.
                Example: "Congratulations! You have won a lottery of $1,000,000!"

            2. num_classes (str, optional): The number of classes for spam detection.
                Default: "spam, not_spam, unknown"

            3. explanation (bool, optional): Whether to include an explanation in the result.
                Default: True

            Returns:
            dict: A dictionary containing the prediction and explanation (if available).
            """
    client = BasicLingua(api_key=api_key)
    results = client.detect_spam(user_input, detect_classes, explain)
    return DictResult(results=results)


@app.post("/text/clean", response_model=TextResult)
async def text_clean_wrapper(
        clean_info: str,
        api_key: Annotated[Union[str, None], Header()] = settings.gemini_api_key,
        user_input: str = Body(..., media_type="text/plain")
):
    """
    Clean the input text based on the given information.

    Parameters:
    1. user_input (str): The input sentence to be cleaned.
        Example:
        ```
        <h1>Heading</h1> <p>para</p> visit to this website https://www.google.com for more information
        ```

    2. clean_info (str): The information on how to clean the text.
        Example: "remove h1 tags but keep their inner text and remove links and fullstop"

    Returns:
    str: The cleaned text.
    """
    client = BasicLingua(api_key=api_key)
    results = client.text_clean(user_input, clean_info)
    return TextResult(results=results)


class NormalizeMode(str, Enum):
    UPPERCASE = "uppercase"
    LOWERCASE = "lowercase"


@app.post("/text/normalize", response_model=TextResult)
async def text_normalize_wrapper(
        mode: NormalizeMode = NormalizeMode.UPPERCASE,
        api_key: Annotated[Union[str, None], Header()] = settings.gemini_api_key,
        user_input: str = Body(..., media_type="text/plain")
):
    """
    Transform user input to either uppercase or lowercase string.

    Parameters:
    1. user_input (str): The string to be transformed.

    2. mode (str): The transformation mode. Valid values are 'uppercase' or 'lowercase'.
    Default: "uppercase"

    Returns:
    str: The transformed string.
    """

    client = BasicLingua(api_key=api_key)
    results = client.text_normalize(user_input, mode)
    return TextResult(results=results)


@app.post("/text/spellcheck", response_model=TextResult)
async def text_spellcheck_wrapper(
        api_key: Annotated[Union[str, None], Header()] = settings.gemini_api_key,
        user_input: str = Body(..., media_type="text/plain")
):
    """
    Correct the misspelled words in the input text.

    Parameters:
    1. user_input (str): The input sentence to perform spell correction on.
       Example: "we wlli oderr pzzia adn buregsr at nghti"

    Returns:
    str: The corrected version of the input sentence with all misspelled words replaced by their correct spellings.
    """
    client = BasicLingua(api_key=api_key)
    results = client.text_spellcheck(user_input)
    return TextResult(results=results)


@app.post("/text/srl", response_model=DictResult)
async def text_srl_wrapper(
        api_key: Annotated[Union[str, None], Header()] = settings.gemini_api_key,
        user_input: str = Body(..., media_type="text/plain")
):
    """
    Perform Semantic Role Labeling (SRL) on the input text.

    Parameters:
    1. user_input (str): The input sentence to perform SRL on.
        Example: "John ate an apple."

    Returns:
    dict: A dictionary containing the detected SRL entities.
    """
    client = BasicLingua(api_key=api_key)
    results = client.text_srl(user_input)
    return DictResult(results=results)


@app.post("/text/cluster", response_model=DictResult)
async def text_cluster_wrapper(
        api_key: Annotated[Union[str, None], Header()] = settings.gemini_api_key,
        user_input: str = Body(..., media_type="text/plain")
):
    """
    Cluster the sentences based on their similarity.

    Parameters:
    1. user_input (str): The input sentences to be clustered.
        Example: '''
        "sentence 1, sentence 2, sentence 3, ..."

    Returns:
    dict: A dictionary where each key-value pair represents a cluster.
        The key is the cluster number, and the value is a list containing similar sentences.
    """
    client = BasicLingua(api_key=api_key)
    results = client.text_cluster(user_input)
    return DictResult(results=results)


@app.post("/text/sentiment", response_model=DictResult)
async def text_sentiment_wrapper(
        num_classes: str = "positive, negative, neutral",
        explanation: bool = True,
        api_key: Annotated[Union[str, None], Header()] = settings.gemini_api_key,
        user_input: str = Body(..., media_type="text/plain")
):
    """
    Perform sentiment detection on the input text.

    Parameters:
    1. user_input (str): The input sentence to perform sentiment detection on.
        Example: "Congratulations! You have won a lottery of $1,000,000!"

    2. num_classes (str, optional): The number of categories for sentiment detection.
        Default: "positive, negative, neutral"

    3. explanation (bool, optional): Whether to include an explanation in the result.
        Default: True

    Returns:
    dict: A dictionary containing the prediction and explanation (if available).
    """
    client = BasicLingua(api_key=api_key)
    results = client.text_sentiment(user_input, num_classes=num_classes, explanation=explanation)
    return DictResult(results=results)


@app.post("/text/topic", response_model=DictResult)
async def text_topic_wrapper(
        num_classes: str,
        explanation: bool = True,
        api_key: Annotated[Union[str, None], Header()] = settings.gemini_api_key,
        user_input: str = Body(..., media_type="text/plain")
):
    """
    Perform topic detection on the input text.

    Parameters:
    1. user_input (str): The input sentence to perform topic detection on.
        Example: "Congratulations! You have won a lottery of $1,000,000!"

    2. num_classes (str, optional): The number of categories for topic detection.
        Default: "story, horror, comedy"

    3. explanation (bool, optional): Whether to include an explanation in the result.
        Default: True

    Returns:
    dict: A dictionary containing the prediction and explanation (if available).
    """
    client = BasicLingua(api_key=api_key)
    results = client.text_topic(user_input, num_classes, explanation)
    return DictResult(results=results)


@app.post("/text/parts_of_speech", response_model=OptionalListResult)
async def text_parts_of_speech_wrapper(
        pos_tags: Optional[str] = None,
        api_key: Annotated[Union[str, None], Header()] = settings.gemini_api_key,
        user_input: str = Body(..., media_type="text/plain")
):
    """
    Perform Part-of-Speech (POS) detection on the input text.

    Parameters:
    1. user_input (str): The input sentence to be analyzed.
        Example: "I love Lamborghini, but Bugatti is even better. Although, Mercedes is a class above all."

    2. pos_tags (str, optional): A comma-separated string specifying the POS tags.
        Example: "noun, verb, adjective"
        Default: "More than 50 TAGS already defined"

    default_pos_tags = 'noun, verb, adjective, adverb, pronoun, preposition, conjunction, interjection, determiner, cardinal, foreign, number, date, time, ordinal, money, percent, symbol, punctuation, emoticon, hashtag, email, url, mention, phone, ip, cashtag, entity, noun_phrase, verb_phrase, adjective_phrase, adverb_phrase, pronoun_phrase, preposition_phrase, conjunction_phrase, interjection_phrase, determiner_phrase, cardinal_phrase, foreign_phrase, number_phrase, date_phrase, time_phrase, ordinal_phrase, money_phrase, percent_phrase, symbol_phrase, punctuation_phrase, emoticon_phrase, hashtag_phrase, email_phrase, url_phrase, mention_phrase, phone_phrase, ip_phrase, cashtag_phrase, entity_phrase'

    Returns:
    list: A list of tuples containing the detected POS entities.
    """
    client = BasicLingua(api_key=api_key)
    results = client.detect_pos(user_input, pos_tags=pos_tags)
    return OptionalListResult(results=results)


@app.post("/text/paraphrase", response_model=DictResult)
async def text_paraphrase_wrapper(
        user_input: List[str],
        api_key: Annotated[Union[str, None], Header()] = settings.gemini_api_key,
        explanation: bool = True
):
    """
    Determine if two sentences are paraphrases of each other.

    Parameters:
    1. user_input (list): A list containing two sentences to be checked for paraphrasing.
        Example: ["OpenAI has hosted a hackathon for developers.", "The event was a huge success with over 1000 participants."]

    2. explanation (bool, optional): Whether to include an explanation in the result.
        Default: True

    Returns:
    str: The prediction of whether the sentences are paraphrases or not.
    """
    client = BasicLingua(api_key=api_key)
    results = client.text_paraphrase(user_input, explanation)
    return DictResult(results=results)


# @app.post("/text/ocr", response_model=TextResult)
# async def text_ocr_wrapper(image: str, prompt: str):
#     """
#     Extract text from an image using the genai library.
#
#     Parameters:
#     1. image_path (str): The path to the image file.
#         Example: "path/to/image.jpg"
#
#     2. prompt (str): The prompt to be used for text extraction.
#         Example: "Extract the text from the image."
#
#     Returns:
#     str: The extracted text from the image.
#     """
#     client = BasicLingua(api_key=settings.gemini_api_key)
#     ocr_result = client.text_ocr(image, prompt)
#     return TextResult(results=ocr_result)

@app.post("/text/segment", response_model=OptionalListResult)
async def text_segment_wrapper(
        logical: bool = True,
        api_key: Annotated[Union[str, None], Header()] = settings.gemini_api_key,
        user_input: str = Body(..., media_type="text/plain")
):
    """
    Segment the given text into individual sentences separated by full stops.

    Parameters:
    1. text_paragraph (str): The input text paragraph(s) to be segmented into sentences.
        Example: "The sun gently rose ..."

    2. logical (bool, optional): Whether to logically segment the text into sentences.
        If True, the prompt code will be used. If False, the text will be split at full stops.
        Default: True

    Returns:
    list: A Python list of sentences.
    """
    client = BasicLingua(api_key=api_key)
    segment_result = client.text_segment(user_input, logical)
    return OptionalListResult(results=segment_result)


class ToxicInputType(str, Enum):
    PROFANITY = "profanity"
    BIAS = "bias"
    SARCASM = "sarcasm"


class ThresholdType(str, Enum):
    BLOCK_NONE = "BLOCK_NONE"
    BLOCK_ONLY_HIGH = "BLOCK_ONLY_HIGH"
    BLOCK_MEDIUM_AND_ABOVE = "BLOCK_MEDIUM_AND_ABOVE"
    BLOCK_LOW_AND_ABOVE = "BLOCK_LOW_AND_ABOVE"


@app.post("/text/toxic/{analysis_type}", response_model=BoolResult)
async def text_badness_wrapper(
        analysis_type: ToxicInputType,
        threshold: ThresholdType = ThresholdType.BLOCK_NONE,
        api_key: Annotated[Union[str, None], Header()] = settings.gemini_api_key,
        user_input: str = Body(..., media_type="text/plain")
):
    """
    Check if the user input contains profanity, biased language, or sarcastic language based on the given analysis type and threshold.

    Parameters:
    1. user_input (str): The input text to be analyzed.

    2. analysis_type (str): The type of analysis to be performed.
        Values (str): "profanity", "bias", "sarcasm"

    3. threshold (str, optional): The threshold level for blocking the respective language.
        Values (str): "BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE"
        Default: "BLOCK_NONE"

    Returns:
    bool: True if the user input contains the respective language, False otherwise.
    """
    client = BasicLingua(api_key=api_key)
    found_toxicity = client.text_badness(user_input, analysis_type, threshold)
    return BoolResult(results=found_toxicity)


@app.post("/text/replace_emojis", response_model=TextResult)
async def text_emojis_wrapper(
        api_key: Annotated[Union[str, None], Header()] = settings.gemini_api_key,
        user_input: str = Body(..., media_type="text/plain")
):
    """
    Replace emojis with their meaning and full form in the given user input.

    Parameters:
    1. user_input (str): The input user input containing emojis.

    Returns:
    str: The user input with emojis replaced by their meaning and full form.
    """

    client = BasicLingua(api_key=api_key)
    cleaned_text = client.text_emojis(user_input)
    return TextResult(results=cleaned_text)


class TFIDFOutputType(str, Enum):
    TFIDF = "tfidf"
    NGRAMS = "ngrams"
    ALL = "all"


@app.post("/text/tfidf", response_model=TupleResult)
async def text_tfidf_wrapper(
        documents: List[str],
        ngram_size: int,
        output_type: TFIDFOutputType,
        api_key: Annotated[Union[str, None], Header()] = settings.gemini_api_key
):
    """
    Calculate the TF-IDF matrix or unique n-grams for a given list of documents and n-gram size.

    Parameters:
    1. documents (list): A list of documents.

    2. ngrams_size (int): The size of n-grams.

    3. output_type (str): The type of output to be generated. Values can be "tfidf", "ngrams", or "all".

    Returns:
    tuple: A tuple containing the TF-IDF matrix, the set of unique n-grams, or both based on the output_type.
    """

    client = BasicLingua(api_key=api_key)
    results = client.text_tfidf(documents, ngram_size, output_type)
    return TupleResult(results=results)


@app.post("/text/idioms", response_model=OptionalListResult)
async def text_idioms_wrapper(user_input: str = Body(..., media_type="text/plain")):
    """
    Identify and extract any idioms present in the given sentence.

    Parameters:
    1. user_input (str): The input sentence.

    Returns:
    list: A list of extracted idioms. If no idiom is found, returns None.
    """
    client = BasicLingua(api_key=settings.gemini_api_key)
    idioms = client.text_idioms(user_input)
    return OptionalListResult(results=idioms)


@app.post("/text/sense_disambiguation", response_model=OptionalListResult)
async def text_sense_disambiguation_wrapper(
        word_to_disambiguate: str,
        api_key: Annotated[Union[str, None], Header()] = settings.gemini_api_key,
        user_input: str = Body(..., media_type="text/plain")
):
    """
    Perform word sense disambiguation for a given input sentence and word to disambiguate.

    Parameters:
    1. user_input (str): The input sentence.

    2. word_to_disambiguate (str): The word to disambiguate.

    Returns:
    list: A list of meanings and their explanations based on the context in the input sentence.
    If the word does not appear in the input sentence, returns None.
    """
    client = BasicLingua(api_key=api_key)
    disambiguations = client.text_sense_disambiguation(user_input, word_to_disambiguate)
    return OptionalListResult(results=disambiguations)


@app.post("/text/frequency", response_model=DictResult)
async def text_frequency_wrapper(
        words: Optional[List[str]] = None,
        api_key: Annotated[Union[str, None], Header()] = settings.gemini_api_key,
        user_input: str = Body(..., media_type="text/plain")
):
    """
    Calculate the frequency of specific words or all words in the given user input.

    Parameters:
    1. user_input (str): The input user input.

    2. words (list, optional): The list of words to calculate the frequency for.
    If None is provided, the function will calculate the frequency for all words.
    Default: None

    Returns:
    dict: A dictionary where the key is the word and the value is its frequency.
    """
    client = BasicLingua(api_key=api_key)
    frequency_result = client.text_word_frequency(user_input, words)
    return DictResult(results=frequency_result)


@app.post("/text/anomaly", response_model=OptionalListResult)
async def text_anomaly_wrapper(
        api_key: Annotated[Union[str, None], Header()] = settings.gemini_api_key,
        user_input: str = Body(..., media_type="text/plain")
):
    """
    Detect any anomalies or outliers in the given input text.

    Parameters:
    1. user_input (str): The input text to be analyzed.

    Returns:
    list: A list of detected anomalies with explanations of how they are anomalous.
    If no anomalies are found, returns None.
    """
    client = BasicLingua(api_key=api_key)
    anomaly_result = client.text_anomaly(user_input)
    return OptionalListResult(results=anomaly_result)


@app.post("/text/core_reference", response_model=OptionalListResult)
async def text_core_reference_wrapper(
        api_key: Annotated[Union[str, None], Header()] = settings.gemini_api_key,
        user_input: str = Body(..., media_type="text/plain")
):
    """
    Perform coreference resolution on the given text to identify who (pronoun) refers to what/whom.

    Parameters:
    1. user_input (str): The input text to perform coreference resolution on.

    Returns:
    list: A list of resolved coreferences in the format "Pronoun refers to Entity".
    If no pronouns are found or if the resolved references cannot be determined, returns None.
    """
    client = BasicLingua(api_key=api_key)
    try:
        core_references = client.text_coreference(user_input)
    except ValueError as ve:
        core_references = []
    return OptionalListResult(results=core_references)
