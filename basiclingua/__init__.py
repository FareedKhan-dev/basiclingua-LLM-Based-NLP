import google.generativeai as genai
import re
import math

class BasicLingua:
    def __init__(self, api_key: str):
        """
        Initializes the BasicLingua class with the given API key.

        `Parameters`:
        1. api_key (str): The API key to be used for the Gemini AI model.
        """

        # Configuring Gemini AI with API key
        genai.configure(api_key=api_key)

        # text generation model
        self.model = genai.GenerativeModel('gemini-1.0-pro-latest')
        
        # vision model
        self.v_model = genai.GenerativeModel('gemini-1.0-pro-vision-latest')


    ########################## Extract Patterns ##########################
    def extract_patterns(self, user_input: str, patterns: str) -> list:
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

        # Generate the prompt template
        prompt_template = f'''
        Given the input sentence:
        user input: {user_input}

        __________________________

        extract {patterns} from it
        Output must only contain the extracted patterns but not what they are
        output must not contain any bullet points
        if no pattern found returns None
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(patterns, str):
            raise TypeError("patterns must be of type str")
        
        
        # Generate response using the provided model (assuming it's defined elsewhere)
        try:
            # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.model.generate_content(prompt_template)
            # Extract the output list
            output_list = response.text.split('\n')
            return output_list
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")


    ########################## Text Translate ##########################
    def text_translate(self, user_input: str, target_lang: str) -> str:

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

        # Generate the prompt template
        prompt_template = f'''Translate the following text into {target_lang}:
        source text: {user_input}

        __________________________

        Output must only contain the translated text.
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(target_lang, str):
            raise TypeError("target_lang must be of type str")
        
        try:
            # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.model.generate_content(prompt_template)
            return response.text
        except Exception as e:
            return "Translation failed. Only the most popular languages are supported. Actively working to add more."
        

    ########################## Text Replace ##########################
    def text_replace(self, user_input: str, replacement_rules: str) -> str:
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

        # Generate the prompt template
        prompt_template = f'''
        Given the original text:
        user input: {user_input}

        And the replacement rule:
        replacement rule: {replacement_rules}
        __________________________

        Replace words in the original text according to the replacement rules provided.
        Apply the rules to modify the text.
        Only provide the output that has the modified text with replacements, nothing else.
        Replace words even when sentence does not make sense.
        make sure all mentioned words must be replaced
        replace word even if change the meaning of the sentence or does not make sense
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(replacement_rules, str):
            raise TypeError("replacement_rules must be of type str")

        try:
            # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.model.generate_content(prompt_template)
            return response.text
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")
        

    ########################## Named Entity Recognition ##########################
    def detect_ner(self, user_input: str, ner_tags: str = "") -> list:
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

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(ner_tags, str):
            raise TypeError("ner_tags must be of type str")
        
        # default NER tags
        default_ner_tags = "person, location, organization, date, time, money, percent"

        # if ner_tags is not provided, use default tags else add the provided tags
        if not ner_tags:
            ner_tags = default_ner_tags
        else:
            ner_tags = default_ner_tags + ", " + ner_tags

        # Generate the prompt template
        prompt_template = f'''Given the input text, perform NER detection on it
        NER Tags are: {ner_tags}
        {user_input}
        answer must be in the format
        word: entity
        '''

        try:
            # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.model.generate_content(prompt_template)

            # Split the string by '\n' to get individual lines
            lines = response.text.split('\n')

            # Initialize an empty list to store tuples
            tuples_answer = []

            # Iterate through each line and split by ':'
            for line in lines:
                if line.strip():  # Check if line is not empty
                    key, value = line.split(':')
                    tuples_answer.append((key.strip(), value.strip()))

            return tuples_answer

        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")
        

    ########################## Text Summarize ##########################
    def text_summarize(self, user_input: str, summary_length: str = "short") -> str:
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
        # Check if the summary_length value is valid
        if summary_length not in ["short", "medium", "long"]:
            raise ValueError("Invalid summary_length value. Allowed values are 'short', 'medium', or 'long'.")
        
        # Define the prompt template
        prompt_template = f'''
        Given the input text:
        Input: {user_input}

        __________________________________________

        Produce a {summary_length} summary of the text.

        Ensure that each summary:
        - Captures the main ideas and key points of the text.
        - Is coherent and concise.
        - Does not include verbatim sentences from the original text.

        Do not produce the summary in bullet points or numbers ever. 
        If a summary is not applicable (e.g., the text is too short), return None for that summary.
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(summary_length, str):
            raise TypeError("summary_length must be of type str")

        try:
            # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.model.generate_content(prompt_template)
            return response.text
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")
        

    ########################## Text QnA ##########################
    def text_qna(self, user_input: str, question: str) -> str:
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
        # Define the prompt template
        prompt_template = f'''
        Given the input text:
        Input: {user_input}

        __________________________________________

        Answer the following question: 
        {question}

        
        The answer should be relevant and concise, without any additional information.
        Ensure that the answer is grammatically correct and directly addresses the question.
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(question, str):
            raise TypeError("question must be of type str")

        try:
            # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.model.generate_content(prompt_template)
            return response.text
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")
        

    ########################## Text Intent ##########################
    def text_intent(self, user_input: str) -> str:
        """
        Identify the intent of the user input.

        Parameters:
        1. user_input (str): The input sentence of which the intent is to be identified.
            Example: "OpenAI has hosted a hackathon for developers to build AI models."

        Returns:
        str: The identified intent.
        """
        
        # Define the prompt template
        prompt_template = f'''
        Given the input sentence:
        user input: {user_input}

        __________________________

        Identify the intent of the user input.

        If no clear intent can be determined from the input, return None.
        If the output intent contains multiple words, separate them with spaces.

        each intent must be new line:
        <intent>
        ...
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        

        try:
            # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.model.generate_content(prompt_template)

            # Get string after "Intent: "
            return response.text.split("\n")
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")        
            

    ########################## Text Lemmatization/Stemming ##########################
    def text_lemstem(self, user_input: str, task_type: str = "stemming") -> str:
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

        # Check if the type value is valid
        if task_type not in ["stemming", "lemmatization"]:
            raise ValueError("Invalid type value. Allowed values are 'stemming' or 'lemmatization'.")

        # Define the prompt template
        prompt_template = f'''
        Given the input sentence, perform {task_type} on it
        {user_input}
        output must be the {task_type} sentence
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")


        try:
            # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.model.generate_content(prompt_template)
            return response.text
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")


    ########################## Text Tokenization ##########################
    def text_tokenize(self, user_input: str, break_point: str = " ") -> list:
        """
        Perform tokenization on the input text.

        Parameters:
        1. user_input (str): The input sentence to be tokenized.
            Example: "OpenAI has hosted a hackathon for developers to build AI models."

        2. break_point (str, optional): The break point to split the sentence into tokens.
            Default: " "

        Returns:
        list: A list of tokens.
        """

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(break_point, str):
            raise TypeError("break_point must be of type str")

        # break the sentence into tokens
        tokens = user_input.split(break_point)

        return tokens


    ########################## Text Embedding ##########################
    def text_embedd(self, user_input: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list:
        """
        Perform tokenization on the input text.

        Parameters:
        1. user_input (str): The input sentence to be tokenized.
            Example: "OpenAI has hosted a hackathon for developers to build AI models."

        2. task_type (str, optional): The task type for embedding.
            Values (str): "RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT", "SEMANTIC_SIMILARITY", "CLASSIFICATION", "CLUSTERING"
            Default: "RETRIEVAL_DOCUMENT"

        Returns:
        list: Embeddings of the input text.
        """

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be of type str")

        try:
            embeddings = genai.embed_content(model='models/embedding-001', content=user_input, task_type=task_type)["embedding"]
            return embeddings
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")


    ########################## Text Generate ##########################
    def text_generate(self, user_input: str, ans_length: str = "short") -> str:
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
        # Check if the ans_length value is valid
        if ans_length not in ["short", "medium", "long"]:
            raise ValueError("Invalid ans_length value. Allowed values are 'short', 'medium', or 'long'.")
        
        # Define the prompt template
        prompt_template = f'''
        Given the input text:
        Input: {user_input}

        __________________________________________

        Generate text based on the input text of {ans_length} length.
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(ans_length, str):
            raise TypeError("ans_length must be of type str")
        
        
        try:
            # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.model.generate_content(prompt_template)
            return response.text
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")
        

    ########################## Spam Analysis ##########################
    def detect_spam(self, user_input: str, num_classes: str = "spam, not_spam, unknown", explanation: bool = True) -> dict:
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
        if explanation:
            format_answer = "prediction:\nexplanation:"
        else:
            format_answer = "prediction:"
        
        # Question to be asked
        prompt_template = f'''Given the input text, perform spam detection on it
        {user_input}
        num_classes: {num_classes}

        _________________________
        You must not provide any other information than the format
        {format_answer}
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(num_classes, str):
            raise TypeError("num_classes must be of type str")
        if not isinstance(explanation, bool):
            raise TypeError("explanation must be of type bool")

        try:
            # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.model.generate_content(prompt_template)

            result = {}
            for line in response.text.split("\n"):
                if line.strip():
                    key, value = line.split(": ", 1)
                    result[key.strip()] = value.strip()

            return result
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")
        

    ########################## Text Clean ##########################
    def text_clean(self, user_input: str, clean_info: str) -> str:
        """
        Clean the input text based on the given information.

        Parameters:
        1. user_input (str): The input sentence to be cleaned.
            Example: "<h1>Heading</h1> <p>para</p> visit to this website https://www.google.com for more information"

        2. clean_info (str): The information on how to clean the text.
            Example: "remove h1 tags but keep their inner text and remove links and fullstop"

        Returns:
        str: The cleaned text.
        """
        prompt_template = f'''
        Given the input sentence:
        {user_input}

        _______________________

        Clean the following input sentence based on the given information: {clean_info}
        you must clean only the asked information but not do other things
        Ensure that only the instructed modifications are made, and all other information remains unchanged.
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(clean_info, str):
            raise TypeError("clean_info must be of type str")

        try:
            # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.model.generate_content(prompt_template)
            return response.text.strip()
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")
        

    ########################## Text Normalize ##########################
    def text_normalize(self, user_input: str, mode: str = "uppercase") -> str:
        """
        Transform user input to either uppercase or lowercase string.

        Parameters:
        1. user_input (str): The string to be transformed.
        
        2. mode (str): The transformation mode. Valid values are 'uppercase' or 'lowercase'.
        Default: "uppercase"

        Returns:
        str: The transformed string.
        """

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(mode, str):
            raise TypeError("mode must be of type str")
        
        # Check if the mode value is valid
        if mode not in ["uppercase", "lowercase"]:
            raise ValueError("Invalid mode value. Allowed values are 'uppercase' or 'lowercase'.")

        if mode == "uppercase":
            transformed_string = user_input.upper()
        else:
            transformed_string = user_input.lower()

        return transformed_string


    ########################## Text Spellcheck ##########################
    def text_spellcheck(self, user_input: str) -> str:
        """
        Correct the misspelled words in the input text.

        Parameters:
        1. user_input (str): The input sentence to perform spell correction on.
            Example: "we wlli oderr pzzia adn buregsr at nghti"

        Returns:
        str: The corrected version of the input sentence with all misspelled words replaced by their correct spellings.
        """
        
        prompt_template = f'''Given the input text:
        user input: {user_input}

        __________________________

        Correct the misspelled words in the sentence.
        Output must only contain the corrected version of the sentence with all misspelled words replaced by their correct spellings.
        Ensure that the corrected version maintains the original sentence structure and punctuation.
        If no misspelled words are found, return the input text unchanged.
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")

        try:
            # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.model.generate_content(prompt_template)

            return response.text.strip()
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")
        

    ########################## Text Semantic Role Labeling ##########################
    def text_srl(self, user_input: str) -> dict:
        """
        Perform Semantic Role Labeling (SRL) on the input text.

        Parameters:
        1. user_input (str): The input sentence to perform SRL on.
            Example: "John ate an apple."

        Returns:
        dict: A dictionary containing the detected SRL entities.
        """

        prompt_template = f'''Given the input sentence:
        user input: {user_input}

        __________________________

        Perform Semantic Role Labeling (SRL) on the input sentence to identify the predicate, agent, and theme.
        - Predicate: The action or state described by the verb.
        - Agent: The entity performing the action.
        - Theme: The entity that is affected by the action.

        Ensure the output follows this format:
        - Predicate: [predicate]
        - Agent: [agent]
        - Theme: [theme]

        If any component is not present or cannot be identified, return None for that component.
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")

        try:
            # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.model.generate_content(prompt_template)

            result = {}
            for line in response.text.split("\n"):
                if line.strip():
                    key, value = line.split(": ", 1)
                    result[key.strip()] = value.strip()

            # Remove "- " from the keys
            result = {key.replace("- ", ""): value for key, value in result.items()} 

            return result
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")
        

    ########################## Clustering Analysis ##########################
    def text_cluster(self, user_input: str) -> dict:
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

        prompt_template = f'''Given the input sentences:
        input_sentences = {user_input}

        _____________________________________________

        Cluster the sentences based on their similarity.
        Each sentence is represented as a string in inverted commas in {user_input}.
        Sentences are separated by commas.

        Output must only return a dictionary where each key-value pair represents a cluster.
        The key is the cluster number, and the value is a list containing similar sentences.
        Please do not return the answer in markdown format or anything else other than the output.

        Ensure that sentences with similar meanings or topics are grouped together.
        The sentences which are not similar to any other sentence must be put in different clusters.
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")

        try:
            import ast
            # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.model.generate_content(prompt_template)

            # Extract the response text
            response_text = response.text

            # Parse the response text to extract the clustered sentences
            start_index = response_text.find("{")
            end_index = response_text.rfind("}") + 1
            clustered_sentences = ast.literal_eval(response_text[start_index:end_index])

            return clustered_sentences

        except Exception as e:
            print("An error occurred. Please try again.")


    ########################## Text Sentiment ##########################
    def text_sentiment(self, user_input: str, num_classes: str = "positive, negative, neutral", explanation: bool = True) -> dict:
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
        if explanation:
            format_answer = "prediction:\nexplanation:"
        else:
            format_answer = "prediction:"
        
        # Question to be asked
        prompt_template = f'''Given the input text, perform sentiment detection on it
        {user_input}
        num_classes: {num_classes}

        _________________________
        You must not provide any other information than the format
        {format_answer}
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(num_classes, str):
            raise TypeError("num_classes must be of type str")
        if not isinstance(explanation, bool):
            raise TypeError("explanation must be of type bool")

        try:
            # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.model.generate_content(prompt_template)

            result = {}
            for line in response.text.split("\n"):
                if line.strip():
                    key, value = line.split(": ", 1)
                    result[key.strip()] = value.strip()

            return result
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")


    ########################## Text Topic ##########################
    def text_topic(self, user_input: str, num_classes: str = "story, horror, comedy", explanation: bool = True) -> dict:
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
        if explanation:
            format_answer = "prediction:\nexplanation:"
        else:
            format_answer = "prediction:"
        
        # Question to be asked
        prompt_template = f'''Given the input text, perform topic detection on it
        {user_input}
        topic are: {num_classes}

        _________________________
        You must not provide any other information than the format
        {format_answer}
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(num_classes, str):
            raise TypeError("num_classes must be of type str")
        if not isinstance(explanation, bool):
            raise TypeError("explanation must be of type bool")

        try:
            # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.model.generate_content(prompt_template)

            result = {}
            for line in response.text.split("\n"):
                if line.strip():
                    key, value = line.split(": ", 1)
                    result[key.strip()] = value.strip()

            return result
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")
        


    ########################## POS Tagging ##########################
    def detect_pos(self, user_input: str, pos_tags: str = "") -> list:
        """
        Perform Part-of-Speech (POS) detection on the input text.

        Parameters:
        1. user_input (str): The input sentence to be analyzed.
            Example: "I love Lamborghini, but Bugatti is even better. Although, Mercedes is a class above all."

        2. pos_tags (str, optional): A comma-separated string specifying the POS tags.
            Example: "noun, verb, adjective"
            Default: "More than 50 TAGS already defined"

        Returns:
        list: A list of tuples containing the detected POS entities.
        """
        
        # Default POS tags
        default_pos_tags = 'noun, verb, adjective, adverb, pronoun, preposition, conjunction, interjection, determiner, cardinal, foreign, number, date, time, ordinal, money, percent, symbol, punctuation, emoticon, hashtag, email, url, mention, phone, ip, cashtag, entity, noun_phrase, verb_phrase, adjective_phrase, adverb_phrase, pronoun_phrase, preposition_phrase, conjunction_phrase, interjection_phrase, determiner_phrase, cardinal_phrase, foreign_phrase, number_phrase, date_phrase, time_phrase, ordinal_phrase, money_phrase, percent_phrase, symbol_phrase, punctuation_phrase, emoticon_phrase, hashtag_phrase, email_phrase, url_phrase, mention_phrase, phone_phrase, ip_phrase, cashtag_phrase, entity_phrase'

        # If pos_tags is not provided, use default tags; otherwise, add the provided tags
        if not pos_tags:
            pos_tags = default_pos_tags
        else:
            pos_tags = default_pos_tags + ", " + pos_tags

        # Generate the prompt template
        prompt_template = f'''Given the input text, perform POS detection on it
        POS Tags are: {pos_tags}
        {user_input}
        Answer must be in the format:
        word: POS_TAG
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(pos_tags, str):
            raise TypeError("pos_tags must be of type str")
        
        try:
            # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.model.generate_content(prompt_template)

            # Split the string by '\n' to get individual lines
            lines = response.text.split('\n')

            # Initialize an empty list to store tuples
            tuples_answer = []

            # Iterate through each line and split by ':'
            for line in lines:
                if line.strip():  # Check if line is not empty
                    key, value = line.split(':')
                    tuples_answer.append((key.strip(), value.strip()))

            return tuples_answer
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")
        

    ########################## Text Paraphrase Detection ##########################
    def text_paraphrase(self, user_input: list, explanation: bool = True) -> str:
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

        if explanation:
            format_answer = "prediction:\nexplanation:"
        else:
            format_answer = "prediction:"

        # Check if the user_input list contains exactly two sentences
        if len(user_input) != 2:
            raise ValueError("Invalid input. Please provide exactly two sentences in a list for paraphrasing.")

        # Define the prompt template
        prompt_template = f'''Given the input text, determine if two sentences are paraphrases of each other.
        Sentence 1: {user_input[0]}
        Sentence 2: {user_input[1]}
        Prediction must be 'yes' or 'no'.
        {format_answer}
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, list):
            raise TypeError("user_input must be of type list")
        if not isinstance(explanation, bool):
            raise TypeError("explanation must be of type bool")

        try:
            # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.model.generate_content(prompt_template)

            result = {}
            for line in response.text.split("\n"):
                if line.strip():
                    key, value = line.split(": ", 1)
                    result[key.strip()] = value.strip()

            return result
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")


    ########################## Text OCR ##########################
    def text_ocr(self, image_path: str, prompt: str) -> str:
        """
        Extract text from an image using the genai library.

        Parameters:
        1. image_path (str): The path to the image file.
            Example: "path/to/image.jpg"
        
        2. prompt (str): The prompt to be used for text extraction.
            Example: "Extract the text from the image."

        Returns:
        str: The extracted text from the image.
        """

        prompt_template = f'''{prompt}
        if no text found returns None
        '''

        # check if parameters are of correct type
        if not isinstance(image_path, str):
            raise TypeError("image_path must be of type str")
        if not isinstance(prompt, str):
            raise TypeError("prompt must be of type str")

        try:
            # Import the Image class from IPython.display
            from IPython.display import Image

            # Load the image using IPython.display.Image
            img = Image(filename=image_path)

            # if the image is not found, raise an error
            if not img:
                raise FileNotFoundError("The image file was not found.")

            # Create a generative model
            v_model = genai.GenerativeModel('gemini-1.0-pro-vision-latest')

            # Generate content to extract text from the image
            response = v_model.generate_content([prompt_template, img])

            # Return the extracted text
            return response.text
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")
        

    ########################## Text Segment ##########################
    def text_segment(self, user_input: str, logical: bool = True) -> list:
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

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(logical, bool):
            raise TypeError("logical must be of type bool")

        # check logical can only be true or false
        if logical not in [True, False]:
            raise ValueError("Invalid logical value. Allowed values are 'True' or 'False'.")


        if logical:
            prompt_template = f'''Given the input text paragraph(s):
            {user_input}

            __________________________

            Segment the given text into individual sentences separated by full stops.
            Output should only be a Python list of sentences, such as:

            ["Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "Sed ut magna ac orci condimentum vestibulum.",
            "Mauris luctus leo eget sapien vehicula, quis mattis sapien viverra.",
            "Donec euismod purus in turpis scelerisque, vitae efficitur justo malesuada.",
            "Duis non lacinia arcu.",
            ...

            Your task is to correctly segment the input paragraph(s) into individual sentences based on the presence of full stops.

            Ensure that each sentence in the output list ends with a full stop. If there's a sentence fragment at the end of the input text without a full stop, include it as a separate sentence.
            If the input text contains no sentences (e.g., it's an empty string), the output should be an empty list.
            The output must not contain bullet points or numbering.
            Note: The input text may contain abbreviations, acronyms, or other instances where periods are not indicative of the end of a sentence. Your task is to identify and separate actual sentences based on context.
            '''
            try:
                    
                # Generate response using the provided model (assuming it's defined elsewhere)
                response = self.model.generate_content(prompt_template)

                # Extract the output list
                output_list = eval(response.text)

                return output_list
            except Exception as e:
                raise ValueError("Please provide a correct API key or try again.")
        else:
            # Split the text at full stops
            output_list = user_input.split(". ")
            return output_list


    ########################## Text Badness ##########################
    def text_badness(self, user_input: str, analysis_type: str, threshold: str = "BLOCK_NONE") -> bool:
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

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(analysis_type, str):
            raise TypeError("analysis_type must be of type str")
        if not isinstance(threshold, str) or threshold not in ["BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE"]:
            raise TypeError("threshold must be of type str and one of 'BLOCK_NONE', 'BLOCK_ONLY_HIGH', 'BLOCK_MEDIUM_AND_ABOVE', 'BLOCK_LOW_AND_ABOVE'")

        safety_settings = [
            {
                "category": "HARM_CATEGORY_DANGEROUS",
                "threshold": threshold,
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": threshold,
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": threshold,
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": threshold,
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": threshold,
            },
        ]

        if analysis_type == "profanity":
            prompt_template = f'''Given the input text paragraph(s):
            {user_input}

            __________________________

            Does the user input contain profanity words or not?
            Answer must only contain True or False.
            '''
        elif analysis_type == "bias":
            prompt_template = f'''Given the input text paragraph(s):
            {user_input}

            __________________________

            Given a user input, analyze whether it contains biased language or not. Biased language refers to words or phrases that unfairly or inaccurately portray individuals or groups based on characteristics such as race, gender, ethnicity, religion, sexual orientation, disability, socioeconomic status, etc
            if userinput given priority to any class on top of other classes then it is biased language
            Answer must only contain True or False.
            '''
        elif analysis_type == "sarcasm":
            prompt_template = f'''Given the input text paragraph(s):
            {user_input}

            __________________________

            Given a user input, determine if it contains sarcastic remarks. Sarcastic remarks are characterized by expressing the opposite of what is actually meant, often conveyed through tone, context, or specific word choices.
            Answer must only contain True or False.
            '''
        else:
            raise ValueError("Invalid analysis type. Please choose either 'profanity', 'bias', or 'sarcasm'.")

        try:
            # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.model.generate_content(prompt_template, safety_settings=safety_settings)

            # Convert string to boolean
            return response.text.strip() == "True"
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")
        

    ########################## Text Emojis ##########################
    def text_emojis(self, user_input: str) -> str:
        """
        Replace emojis with their meaning and full form in the given user input.

        Parameters:
        1. user_input (str): The input user input containing emojis.

        Returns:
        str: The user input with emojis replaced by their meaning and full form.
        """

        prompt_template = f'''Given the input text paragraph(s):
        {user_input}

        __________________________

        Replace Emojis with their meaning and full form and provide updated output.
        output must not contain markdown
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")

        try:
            # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.model.generate_content(prompt_template)
            return response.text
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")


    ########################## Text TFIDF ##########################
    def text_tfidf(self, documents: list, ngrams_size: int = 2, output_type: str = "tfidf") -> tuple:
        """
        Calculate the TF-IDF matrix or unique n-grams for a given list of documents and n-gram size.

        Parameters:
        1. documents (list): A list of documents.

        2. ngrams_size (int): The size of n-grams.

        3. output_type (str): The type of output to be generated. Values can be "tfidf", "ngrams", or "all".

        Returns:
        tuple: A tuple containing the TF-IDF matrix, the set of unique n-grams, or both based on the output_type.
        """

        # check if parameters are of correct type
        if not isinstance(documents, list):
            raise TypeError("documents must be of type list")
        if not isinstance(ngrams_size, int):
            raise TypeError("ngrams_size must be of type int")
        if not isinstance(output_type, str):
            raise TypeError("output_type must be of type str")

        if type(ngrams_size) is not int:
            raise ValueError("Invalid ngram value. ngram must be an integer.")

        unique_ngrams = set()
        tf_idf_matrix = []

        for document in documents:
            words = re.findall(r'\b\w+\b', document.lower())
            ngrams_list = zip(*[words[i:] for i in range(ngrams_size)])
            ngrams = [" ".join(ngram) for ngram in ngrams_list]
            unique_ngrams.update(ngrams)

        for document in documents:
            tf_idf_vector = []
            total_words = len(document.split())

            for ngram in unique_ngrams:
                word_count = document.count(ngram)
                tf = word_count / total_words

                document_count = sum(1 for doc in documents if ngram in doc)
                idf = math.log(len(documents) / (1 + document_count))

                tf_idf = tf * idf
                tf_idf_vector.append(tf_idf)

            tf_idf_matrix.append(tf_idf_vector)

        if output_type == "tfidf":
            return tf_idf_matrix
        elif output_type == "ngrams":
            return unique_ngrams
        elif output_type == "all":
            return tf_idf_matrix, unique_ngrams
        else:
            raise ValueError("Invalid output_type. Please choose either 'tfidf', 'ngrams', or 'all'.")
        

    ########################## Text Idioms ##########################
    def text_idioms(self, user_input: str) -> list:
        """
        Identify and extract any idioms present in the given sentence.

        Parameters:
        1. user_input (str): The input sentence.

        Returns:
        list: A list of extracted idioms. If no idiom is found, returns None.
        """

        prompt_template = f'''Given the input sentence:
        user input: {user_input}

        __________________________

        Identify and extract any idioms present in the sentence.
        Output must only contain the extracted idioms.
        Output must not contain any bullet points.
        If there is more than one idiom found, return both in new lines.
        If no idiom is found, return None.
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")

        try:
            # Generate response using the provided model (assuming it's defined elsewhere)            
            response = self.model.generate_content(prompt_template)
            return response.text.split("\n")
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")
        

    ########################## Text Sense Disambiguation ##########################
    def text_sense_disambiguation(self, user_input: str, word_to_disambiguate: str) -> list:
        """
        Perform word sense disambiguation for a given input sentence and word to disambiguate.

        Parameters:
        1. user_input (str): The input sentence.

        2. word_to_disambiguate (str): The word to disambiguate.

        Returns:
        list: A list of meanings and their explanations based on the context in the input sentence.
        If the word does not appear in the input sentence, returns None.
        """

        prompt_template = f'''Given the input sentence:
        user input: {user_input}

        __________________________

        For the word "{word_to_disambiguate}" in the input sentence:
        - Provide its possible senses
        - Explain each sense briefly based on the context it is used for in the input sentence
        - If the word has multiple meanings, provide explanations for each meaning separately
        - Ensure the explanations are concise and easy to understand
        - If the word has only one meaning in the given context, provide its explanation directly
        - If the word does not appear in the input sentence, return None

        - Output should be like:
            Word to Disambiguate
            Meaning1: output (in the first sentence)
            Meaning2: output (in the second sentence)
            ...
        dont provide other information than the output format
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(word_to_disambiguate, str):
            raise TypeError("word_to_disambiguate must be of type str")

        try:    
            # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.model.generate_content(prompt_template)

            # Extract Word to Disambiguate: and Meaning 1 and Meaning 2
            information = response.text.split("\n")
            information = [meaning for meaning in information if meaning]

            return information
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")
        


    ########################## Text Word Frequency ##########################
    def text_word_frequency(self, user_input: str, words: list = None) -> dict:
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

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if words is not None and not isinstance(words, list):
            raise TypeError("words must be of type list")

        # remove punctuation from the user input
        user_input = re.sub(r'[^\w\s]', '', user_input)

        # calculate the frequency of specific words or all words in the given user input
        words_list = user_input.split()
        frequency_dict = {}

        if words is None:
            words = words_list

        for word in words:
            frequency_dict[word] = words_list.count(word)

        return frequency_dict
    

    ########################## Text Anomaly ##########################
    def text_anomaly(self, user_input: str) -> list:
        """
        Detect any anomalies or outliers in the given input text.

        Parameters:
        1. user_input (str): The input text to be analyzed.

        Returns:
        list: A list of detected anomalies with explanations of how they are anomalous.
        If no anomalies are found, returns None.
        """

        prompt_template = f'''Given the input sentence:
        user input: {user_input}

        __________________________

        Detect any anomalies or outliers in the text. An anomaly is any part of the text that deviates significantly from what would be considered normal or expected. This could include:
        - Uncommon or improbable events
        - Strange or unusual combinations of words
        - Inconsistent or contradictory information
        - Unexpected context or setting

        Your output should include the detected anomalies with explanations of how they are anomalous.

        If no anomalies are found, return None.
        output must not be markdown and must be in single string
        output format does not include explanation
        anaomaly:
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        
        try:
            # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.model.generate_content(prompt_template)

            # break it into list
            information = response.text.split("\n")

            return information
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")
        

    ########################## Text Coreference ##########################
    def text_coreference(self, user_input: str) -> list:
        """
        Perform coreference resolution on the given text to identify who (pronoun) refers to what/whom.

        Parameters:
        1. user_input (str): The input text to perform coreference resolution on.

        Returns:
        list: A list of resolved coreferences in the format "Pronoun refers to Entity".
        If no pronouns are found or if the resolved references cannot be determined, returns None.
        """
        
        prompt_template = f'''Given the input paragraph:
        user input: {user_input}

        __________________________

        Perform coreference resolution to identify what each pronoun in the paragraph is referring to. 
        Output must only contain the resolved references for each pronoun, without any additional context.
        Output must not contain any bullet points.
        If no referent is found for a pronoun, return 'None'.

        Output must be like:
        ['Pronoun1: Entity', 'Pronoun2: Entity', ...]
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")

        try:
                
            response = self.model.generate_content(prompt_template)

            # Extract resolved coreferences
            coreferences = response.text.split("\n")
            coreferences = [coreference for coreference in coreferences if coreference]

            return coreferences
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")