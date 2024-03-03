<!-- omit in toc -->
# BasicLingua
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aWOEpzco-7zevu2ma7djkiCpcV_Rk2Iv?usp=sharing) [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/fareedkhan557/basiclingua-nlp) [![GitHub](https://img.shields.io/badge/GitHub-Notebook-blue?logo=github)](https://github.com/FareedKhan-dev/basic_lingua/blob/main/backend_notebook.ipynb) [![Documentation](https://img.shields.io/badge/Documentation-Link-blue)](https://basiclingua-docs.streamlit.app/) [![License](https://img.shields.io/badge/License-MIT-blue)](https://opensource.org/licenses/MIT)

![Basic-Lingua Logo](https://i.ibb.co/smMH4dR/logo.png)

Basiclingua is a Gemini LLM based Python library that provides functionalities for linguistic tasks such as tokenization, stemming, lemmatization, and many others.'

**Why Building this Project?**

The problem that we plan to tackle is the increasing complexity and difficulty of handling text data as its size and complexity increase. NLP libraries that offer solutions are either limited in their ability to solve the required problem or require a great deal of human intervention to handle text data. We have used Gemini Language Model, which have demonstrated promising results in dealing with text data, to address the complexity that no NLP library has yet solved. As a result, we will be able to handle text-related tasks with minimal human intervention. We have created a powerful NLP library capable of solving any type of human text-related task, producing accurate results.

**Updates**
- **`2024/3/3`** We have released the first version of the library. The library is now available for use. We are currently working on the documentation and the next version of the library. We are also working on the integration of the library with other LLMs.
- **`2024/2/10`** We have released the baby version of this library containing limited number of pre-processing features.

<!-- omit in toc -->
## Table of Content
- [Installation](#installation)
- [Initialization](#initialization)
- [Usage](#usage)
- [Features of the library](#features-of-the-library)
- [BasicLingua Playground](#basiclingua-playground)
- [Contributors](#contributors)
- [Acknowledgements](#acknowledgements)

## Installation

To install the library, you can use pip:ff

```bash
pip install basiclingua
```

## Initialization

To initialize the client, you need to import the library and initialize the client with your Gemini API key. You can get your Gemini API key from [here](https://makersuite.google.com/app/apikey).

```python
# Import the library
from basiclingua import BasicLingua

# Initialize the client
client = BasicLingua("YOUR_GEMINI_API_KEY")
```

Replace `YOUR_API_KEY` with your Gemini API key.


## Usage

The library provides a wide range of functionalities for linguistic tasks such as tokenization, stemming, lemmatization, and many others. Here are some examples of how to use the library:

In the following example, we will use the extract_patterns method to extract patterns such as email, phone number, and name from the user_input instead of using regex.

```python
# user_input is the text from which you want to extract patterns
user_input = '''The phone number of fareed khan and asad rizvi are 
                123-456-7890 and 523-456-7892. Please call for
                assistance and email me at x123@gmail.com'''

# patterns is the list of patterns you want to extract from the user_input
patterns = '''email, phone number, name'''

# Extract patterns from the user_input
extracted_patterns = client.extract_patterns(user_input, patterns)

# Print the extracted patterns
print(extracted_patterns)

######## Output ########
['123-456-7890', '523-456-7892', 'fareed khan', 'asad rizvi', 'x123@gmail.com']
```
In the following example, we will use the text_intent method to find the intent of the user from the user_input.

```python
# user_input is the text from which you want to find intent of the user
user_input = '''let's book a flight for our vacation and reserve 
                a table at a restaurant for dinner, Also going 
                to watch football match at 8 pm.'''

# Find the intent of the user
intent = client.text_intent(user_input)

# Print the intent
print(intent)

######## Output ########
['Book Flight', 'Reserve Restaurant', 'Watch Football Match']
```

In the following example, we will use the detect_ner method to find the NER tags from the user_input.

```python
# user_input is the text from which you want to find NER tags
user_input = '''I love Lamborghini, but Bugatti is even better.
                Although, Mercedes is a class above all and I work in Google'''

# ner_tags is the list of NER tags you want to find from the user_input
ner_tags="cars, date, time"

# Find the NER tags from the user_input
answer = client.detect_ner(user_input, ner_tags)

# Print the NER tags
print(answer)

######## Output ########
[('Lamborghini', 'cars'), ('Bugatti', 'cars'), 
('Mercedes', 'cars'), ('Google', 'organization')]
```

There are many other functionalities provided by the library such as tokenization, stemming, lemmatization, and many others.


## Features of the library

There are total **31** functionalities provided by the library. Here is the list of all the functionalities provided by the library. You can find example of each functionality in the [documentation](https://basiclingua-docs.streamlit.app/). The list of functionalities is as follows:

| Function Name           | Python Function Name  | Parameters                                 | Returns                                                                  |
|-------------------------|-----------------------|--------------------------------------------|--------------------------------------------------------------------------|
| Initialize the Library  | BasicLingua           | api_key                                    | An instance of the BasicLingua class                                    |
| Extract Patterns        | extract_patterns      | user_input, patterns                      | A list of extracted patterns from the input sentence                     |
| Text Translation        | text_translate        | user_input, target_lang                   | Translated text in the target language                                   |
| Text Replace            | text_replace          | user_input, replacement_rules             | Modified text with replacements applied                                  |
| Detect NER              | detect_ner            | user_input, ner_tags                      | A list of detected Named Entity Recognition (NER) entities              |
| Text Summarize          | text_summarize        | user_input, summary_length                | A summary of the input text                                              |
| Text Q&A                | text_qna              | user_input, question                      | The answer to the given question based on the input text                 |
| Text Intent             | text_intent           | user_input                                 | A list of identified intents from the input sentence                     |
| Text Lemstem            | text_lemstem          | user_input, task_type                     | Stemmed or lemmatized text                                               |
| Text Tokenize           | text_tokenize         | user_input, break_point                   | A list of tokens from the input text                                     |
| Text Embedd             | text_embedd           | user_input, task_type                     | Embeddings of the input text                                             |
| Text Generate           | text_generate         | user_input, ans_length                    | Generated text based on the input text                                   |
| Detect Spam             | detect_spam           | user_input, num_classes, explanation      | Prediction and (optionally) explanation of spam detection               |
| Text Clean              | clean_text            | user_input, clean_info                    | Cleaned text based on the given information                              |
| Text Normalize          | text_normalize        | user_input, mode                          | Transformed string to either uppercase or lowercase                      |
| Text Spellcheck         | text_spellcheck       | user_input                                 | Corrected version of the input sentence                                  |
| Text SRL                | text_srl              | user_input                                 | A dictionary containing the detected Semantic Role Labeling (SRL) entities|
| Text Cluster            | text_cluster          | user_input                                 | A dictionary where each key-value pair represents a cluster              |
| Text Sentiment          | text_sentiment        | user_input, num_classes, explanation      | A dictionary containing the sentiment prediction and (optionally) explanation|
| Text Topic              | text_topic            | user_input, num_classes, explanation      | A dictionary containing the topic prediction and (optionally) explanation|
| Detect POS              | detect_pos            | user_input, pos_tags                      | A list of detected Part-of-Speech (POS) entities                        |
| Text Paraphrase         | text_paraphrase       | user_input, explanation                   | Prediction of whether two sentences are paraphrases                      |
| Text OCR                | text_ocr              | image_path, prompt                        | Extracted text from the image                                            |
| Text Segment            | text_segment          | user_input, logical                       | A Python list of sentences                                               |
| Text Emojis             | text_emojis           | user_input                                 | User input with emojis replaced by their meaning and full form           |
| Text TF-IDF             | text_tfidf            | documents, ngrams_size, output_type       | A tuple containing the TF-IDF matrix and the set of unique n-grams       |
| Text Idioms             | text_idioms           | user_input                                 | A list of extracted idioms from the input sentence                       |
| Text Sense Disambiguation| text_sense_disambiguation | user_input, word_to_disambiguate      | A list of possible senses of the word and their explanations             |
| Text Word Frequency     | text_word_frequency   | user_input, words                         | A dictionary where the key is the word and the value is its frequency    |
| Text Anomaly            | text_anomaly          | user_input                                 | A list of detected anomalies with explanations                           |
| Text Coreference        | text_coreference      | user_input                                 | A list of resolved coreferences in the format "Pronoun refers to Entity"|
| Text Badness            | text_badness          | user_input, analysis_type, threshold      | True if the input contains the respective language, False otherwise      |


## BasicLingua Playground
[[Try the Colab Demo](https://colab.research.google.com/drive/1aWOEpzco-7zevu2ma7djkiCpcV_Rk2Iv?usp=sharing)] &nbsp; [[Try Kaggle Demo](https://www.kaggle.com/code/fareedkhan557/basiclingua-nlp)] &nbsp; [[Learn from Documentation](https://basiclingua-docs.streamlit.app/)]

Since this library is available under the MIT license, you can use it in your projects. You can also contribute to the library by adding new functionalities or improving the existing ones. All the backend code is available in the [this notebook](https://github.com/FareedKhan-dev/basic_lingua/blob/main/backend_notebook.ipynb).

## Contributors

<a href="https://github.com/FareedKhan-dev/basic_lingua/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=FareedKhan-dev/basic_lingua" />
</a>

Made with [contrib.rocks](https://contrib.rocks).

##  Acknowledgements

- [Gemini Multi-Model](https://blog.google/technology/ai/google-gemini-ai/)