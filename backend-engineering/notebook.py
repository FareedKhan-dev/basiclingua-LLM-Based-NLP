######### IMPORTING LIBRARIES #########
import google.generativeai as genai
import re
import emoji
import fitz
import ast
from openai import OpenAI

######### GEMINI LINGUA CLASS #########
class GeminiLingua:

    ############## INITIALIZATION ##############
    def __init__(self, api_key: str, model_name: str = 'gemini-1.0-pro-latest', vision_model_name: str = 'models/gemini-1.5-pro-latest'):
        """
        Initializes the BasicLingua class with the provided API key and model names (optional).

        Args:
            api_key (str): A string representing the API key for the Generative AI model.
            model_name (str): If not provided, the default value is 'gemini-1.0-pro-latest'.
            vision_model_name (str): If not provided, the default value is 'gemini-1.0-pro-vision-latest'.
        """

        # Configuring Gemini AI with API key
        genai.configure(api_key=api_key)

        # text generation model
        self.model = genai.GenerativeModel(model_name)
        
        # vision model
        self.v_model = genai.GenerativeModel(vision_model_name)

    
    ############## EXTRACT PATTERNS ##############
    def extract_patterns(self, user_input: str, patterns: str) -> dict:
        """
        Extracts patterns from the given input.

        Example:
        >>> extract_patterns("The phone number of fareed khan and asad are 123-67-325 and 242-176-921", "phone numbers, person names")
        {'phone numbers': ['123-67-325', '242-176-921'], 'person names': ['fareed khan', 'asad']}

        Args:
            user_input (str): A string representing the input sentence.
            patterns (str): A string of comma-separated patterns to be extracted from the input.

        Returns:
            dict: A dictionary containing the extracted patterns with their values.
        """

        # Generate the prompt template
        prompt_template = f'''
        Given the input text:
        user input: {user_input}

        extract following patterns from it: {patterns}

        output must be in this format:
        pattern_name: pattern_values
        ...
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(patterns, str):
            raise TypeError("patterns must be of type str")
        

        # check if parameters are not empty
        if not user_input:
            raise ValueError("user_input cannot be empty")
        if not patterns:
            raise ValueError("patterns cannot be empty")
        
        # running Gemini model
        try:
            response = self.model.generate_content(prompt_template)
            x = response.text.split('\n')
            x = [i.split(':') for i in x if i]
            x = {i[0].strip(): [j.strip() for j in i[1].split(',')] for i in x}
            return x
        except Exception as e:
            raise ValueError("Please provide correct API key or try again")


    ############## TEXT TRANSLATE ##############        
    def text_translate(self, user_input: str, target_lang: str) -> str:
        """
        Translates the given input text into the target language.

        Example:
        >>> text_translate("Farid khan and asad are coming to my house at 5 pm", "urdu")
        'فرید خان اور اسد شام 5 بجے میرے گھر آ رہے ہیں'

        Args:
            user_input (str): A string representing the input sentence.
            target_lang (str): A string representing the target language to translate the input into.

        Returns:
            str: A string representing the translated text.
        """

        # Generate the prompt template
        prompt_template = f'''
        Given the input text:
        user input: {user_input}

        convert it into {target_lang} language
        output must contain only the translated text
        '''
        
        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(target_lang, str):
            raise TypeError("target_lang must be of type str")
        
        # check if parameters are not empty
        if not user_input:
            raise ValueError("user_input cannot be empty")
        if not target_lang:
            raise ValueError("target_lang cannot be empty")
        
        try:
            # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.model.generate_content(prompt_template)
            return response.text
        except Exception as e:
            return "Translation failed. Only the most popular languages are supported. Actively working to add more."


    ############## TEXT REPLACE ##############
    def text_replace(self, user_input: str, replacement_rules: str) -> str:
        """
        Replaces words in the given input text according to the replacement rules provided.
        
        Example:
        >>> text_replace("Fareed Phone number is 123-67-325", "sensitive information such as phone numbers:XXXXX")
        'Fareed Phone number is XXXXX'
        
        Args:
            user_input (str): A string representing the input sentence.
            replacement_rules (str): A string representing the replacement rules in the format "word_to_replace:replacement_word".

        Returns:
            str: A string representing the modified text with replacements.
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

        output format:
        word_to_replace: replacement_word
        '''


        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(replacement_rules, str):
            raise TypeError("replacement_rules must be of type str")

        try:
            response = self.model.generate_content(prompt_template)
            response = response.text.split('\n')

            # remove white spaces from the list
            response = [line.strip() for line in response]

            # Create a list comprehension to extract key-value pairs from each line
            result = [(line.split(":")[0], line.split(":")[1]) for line in response if line.strip()]

            # Replace the misspelled words with the corrected words
            for word, corrected_word in result:
                corrected_user_input = user_input.replace(word, corrected_word)

            return corrected_user_input
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")


    ############## NER EXTRACTION ##############
    def detect_ner(self, user_input: str, ner_tags: str = "") -> dict:
        """
        Perform Named Entity Recognition (NER) detection on the input text.

        Example:
        >>> detect_ner("I love Lamborghini, but Bugatti is even better. Although, Mercedes is a class above all. and i work in Google", "cars, date, time")
        {'cars': ['Lamborghini', 'Bugatti', 'Mercedes'], 'date': [], 'time': []}

        Args:
            user_input (str): A string representing the input sentence.
            ner_tags (str): A string of comma-separated NER tags to be detected from the input.

        Returns:
            dict: A dictionary containing the detected NER tags with their values.
        """

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(ner_tags, str):
            raise TypeError("ner_tags must be of type str")
        
        # check if parameters are not empty
        if not user_input:
            raise ValueError("user_input cannot be empty")
        
        # user ner tags
        if ner_tags != "":
            user_ner_tags = f'''NER TAGS: {ner_tags}'''
        else:
            user_ner_tags = f'''NER TAGS: FAC, CARDINAL, NUMBER, DEMONYM, QUANTITY, TITLE, PHONE_NUMBER, NATIONAL, JOB, PERSON, LOC, NORP, TIME, CITY, EMAIL, GPE, LANGUAGE, PRODUCT, ZIP_CODE, ADDRESS, MONEY, ORDINAL, DATE, EVENT, CRIMINAL_CHARGE, STATE_OR_PROVINCE, RELIGION, DURATION, URL, WORK_OF_ART, PERCENT, CAUSE_OF_DEATH, COUNTRY, ORG, LAW, NAME, COUNTRY, RELIGION, TIME'''

        # Generate the prompt template
        prompt_template = f'''
        Example:
        I love Lamborghini, but Bugatti is even better. Although, Mercedes is a class above all. and i work in Google

        Output:
        cars: Lamborghini, Bugatti, Mercedes
        ORG: Google

        _______________________________________

        Given the input text:
        user input: {user_input}

        perform NER detection on it.
        {user_ner_tags}
        '''

        try:
            # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.model.generate_content(prompt_template)
            x = response.text.split('\n')
            x = [i.split(':') for i in x if i]
            x = {i[0].strip(): [j.strip() for j in i[1].split(',')] for i in x}
            return x

        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")
        

    ############## TEXT SUMMARIZATION ##############
    def text_summarize(self, user_input: str, summary_length: int = 4) -> str:
        """
        Generate a summary of the input text.

        Example:
        >>> text_summarize("Beneath the canopy of a vast and ancient forest, where the towering trees stand as silent sentinels of time, the night unfolds with a grandeur unmatched by any other hour. The moon, a luminous orb hanging in the sky like a celestial lantern, casts its gentle light upon the forest floor, illuminating the intricate tapestry of life that thrives in the shadows. Each leaf, each blade of grass, seems to shimmer with a silvered glow, as if touched by the hand of magic.", 1)
        'The moon, a luminous orb hanging in the sky like a celestial lantern, casts its gentle light upon the forest floor, illuminating the intricate tapestry of life that thrives in the shadows.'

        Args:
            user_input (str): A string representing the input text.
            summary_length (int): An integer representing the length of the summary.

        Returns:
            str: A string representing the summary of the input text.
        """
        
        # Define the prompt template
        prompt_template = f'''
        Given the input text:
        Input: {user_input}
        Produce a {summary_length} sentences length summary of the text.
        Captures the main ideas and key points of the text.
        Summary Does not include verbatim sentences from the original text.
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(summary_length, int):
            raise TypeError("summary_length must be of type int")

        try:
            # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.model.generate_content(prompt_template)
            return response.text
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")


    ############## TEXT QUESTION ANSWERING ##############
    def text_qna(self, user_input: str, question: str) -> str:
        """
        answer the given question based on the input text.

        Example:
        >>> text_qna("OpenAI has hosted a hackathon for developers to build AI models. The event took place on 15th October 2022. The event was a huge success with over 1000 participants from around the world.", "When did the event happen?")
        '15th October 2022'

        Args:
            user_input (str): A string representing the input text.
            question (str): A string representing the question to be answered based on the input text.

        Returns:
            str: A string representing the answer to the question based on the input text.
        """
        # Define the prompt template
        prompt_template = f'''
        Given the input text:
        Input: {user_input}
        Answer the following question: 
        {question}
        The answer should be relevant and concise, without any additional information.
        Ensure that the answer directly addresses the question.
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


    ############## TEXT INTENT DETECTION ##############
    def text_intent(self, user_input: str) -> list:
        """
        Identify the intent of the user input.

        Example:
        >>> text_intent("let's book a flight for our vacation and reserve a table at a restaurant for dinner. also going to watch football match at 8 pm.")
        ['book a flight', 'reserve a table', 'watch football match']

        Args:
            user_input (str): A string representing the input sentence.

        Returns:
            list: A list of strings representing the detected intents from the input text.
        """
            
        # Define the prompt template
        prompt_template = f'''
        Given the input sentence:
        user input: {user_input}
        Identify the intent of the text.
        If no clear intent can be determined from the input, return None.
        If the output intent contains multiple words, separate them with comma.
        output must be in this format -> Intent: intent1, intent2, ...
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        

        try:
            response = self.model.generate_content(prompt_template)
            response1 = response.text
            # make those intent in a list
            response1 = response1.split("Intent: ")[1].split(", ")
            return response1
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")


    ############## TEXT EMBEDDING ##############
    def text_embedd(self, user_input: str, task_type: str = "RETRIEVAL_DOCUMENT", embedd_model: str = "models/embedding-001") -> list:
        """
        Generate embeddings for the given input text.

        Example:
        >>> text_embedd("OpenAI has hosted a hackathon for developers to build AI models.")
        [0.123, 0.456, 0.789, ..., 0.987]

        Args:
            user_input (str): A string representing the input text.
            embedd_model (str): A string representing the embedding model to use. If not provided, the default value is "models/embedding-001".
            task_type (str): The task type for embedding. Values can be "RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT", "SEMANTIC_SIMILARITY", "CLASSIFICATION", "CLUSTERING". Defaults to "RETRIEVAL_DOCUMENT".

        Returns:
            list: Embeddings of the input text.
        """

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be of type str")

        try:
            embeddings = genai.embed_content(model=embedd_model, content=user_input, task_type=task_type)["embedding"]
            return embeddings
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")


    ############## TEXT SPAM ##############
    def detect_spam(self, user_input: str, num_classes: str = "spam, not_spam, unknown", explanation: bool = True) -> dict:
        """
        Perform spam detection on the input text.

        Example:
        >>> detect_spam("Congratulations! You have won a lottery of $1,000,000!", "harmed, not_harmed, unknown", explanation=False)
        {'prediction': 'harmed'}

        Args:
            user_input (str): A string representing the input text.
            num_classes (str): A string representing the number of classes for spam detection.
            explanation (bool): A boolean value representing whether to include explanation in the output.

        Returns:
            dict: A dictionary containing the prediction of the spam detection.
        """
        if explanation:
            format_answer = "prediction:your_answer\nexplanation:your_answer"
        else:
            format_answer = "prediction:your_answer\n"
        
        # Question to be asked
        prompt_template = f'''Given the input text, perform spam detection on it
        {user_input}
        num_classes: {num_classes}
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
            # # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.model.generate_content(prompt_template)
            response1 = response.text

            result = {}
            for line in response1.split("\n"):
                if line.strip():
                    key, value = line.split(": ", 1)
                    result[key.strip()] = value.strip()

            return result
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")


    ############## TEXT SPELLCHECK ##############
    def text_spellcheck(self, user_input: str) -> str:
        """
        Correct the spelling of the input text.

        Example:
        >>> text_spellcheck("I am a good speler")
        'I am a good speller'

        Args:
            user_input (str): A string representing the input text.

        Returns:
            str: A string representing the corrected spelling of the input text.
        """
        
        prompt_template = f'''Given the input text:
        user input: {user_input}
        output must be in this format:
        misspelled_word:corrected_word
        ...
        output must not contain any other information than the format
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")

        try:
            response = self.model.generate_content(prompt_template)
            response1 = response.text

            response1 = response1.split("\n")
            result = [(line.split(":")[0], line.split(":")[1]) for line in response1 if line.strip()]

            # replace the misspelled words with the corrected words
            for word, corrected_word in result:
                user_input = user_input.replace(word, corrected_word)

            return user_input
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")

    ############## TEXT SEMANTIC ROLE LABELING ##############
    def text_srl(self, user_input: str) -> dict:
        """
        Perform Semantic Role Labeling (SRL) on the input text.

        Example:
        >>> text_srl("John ate an apple")
        {'Predicate': 'ate', 'Agent': 'John', 'Theme': 'apple'}

        Args:
            user_input (str): A string representing the input sentence.

        Returns:
            dict: A dictionary containing the Predicate, Agent, and Theme of the input sentence.
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

    ############## TEXT TOPIC ##############
    def text_topic(self, user_input: str, explanation: bool = True) -> dict:
        """
        Perform topic detection on the input text.

        Example:
        >>> text_topic("OpenAI has hosted a hackathon for developers to build AI models. The event took place on 15th October 2022. The event was a huge success with over 1000 participants from around the world.", explanation=False)
        {'topic': 'technology'}

        Args:
            user_input (str): A string representing the input text.
            explanation (bool): A boolean value representing whether to include explanation in the output.

        Returns:
            dict: A dictionary containing the detected topic from the input text.
        """
        if explanation:
            format_answer = "topic:your_answer\nexplanation:your_answer"
        else:
            format_answer = "topic:your_answer\n"
        
        # Question to be asked
        prompt_template = f'''Given the input text, perform topic detection on it
        {user_input}
        You must not provide any other information than the format
        {format_answer}
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(explanation, bool):
            raise TypeError("explanation must be of type bool")

        try:
            # Generate response using the provided model (assuming it's defined elsewhere)
            # # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.model.generate_content(prompt_template)

            response1 = response.text

            result = {}
            for line in response1.split("\n"):
                if line.strip():
                    try:
                        key, value = line.split(": ", 1)
                        result[key.strip()] = value.strip()
                    except ValueError:
                        key, value = line.split(":", 1)
                        result[key.strip()] = value.strip()

            return result
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")


    ############## POS DETECTION ##############
    def detect_pos(self, user_input: str, pos_tags: str = "") -> dict:
        """
        Perform Part-of-Speech (POS) detection on the input text.

        Example:
        >>> detect_pos("I love Lamborghini, but Bugatti is even better. Although, Mercedes is a class above all. and i work in Google", "noun, verb, adjective")
        {'noun': ['Lamborghini', 'Bugatti', 'Mercedes', 'Google'], 'verb': ['love', 'is', 'work'], 'adjective': ['better', 'above', 'all']}

        Args:
            user_input (str): A string representing the input sentence.
            pos_tags (str): A string of comma-separated POS tags to be detected from the input.

        Returns:
            dict: A dictionary containing the detected NER tags with their values.
        """

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(pos_tags, str):
            raise TypeError("ner_tags must be of type str")
        
        # check if parameters are not empty
        if not user_input:
            raise ValueError("user_input cannot be empty")
        
        # user ner tags
        if pos_tags != "":
            user_pos_tags = f'''POS TAGS: {pos_tags}'''
        else:
            user_pos_tags = f'''POS TAGS:noun, verb, adjective, adverb, pronoun, preposition, conjunction, interjection, determiner, cardinal, foreign, number, date, time, ordinal, money, percent, symbol, punctuation, emoticon, hashtag, email, url, mention, phone, ip, cashtag, entity,'''

        # Generate the prompt template
        prompt_template = f'''Given the input text:
        user input: {user_input}
        
        perform POS detection on it.
        {user_pos_tags}
        answer must be in the format
        tag:value
        '''

        try:
            # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.model.generate_content(prompt_template)

            # convert x to a dictionary and values must be a list
            x = response.text.split('\n')
            x = [i.split(':') for i in x if i]
            x = {i[0].strip(): [j.strip() for j in i[1].split(',')] for i in x}
            return x

        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")


    ############## TEXT BADNESS ##############
    def text_badness(self, user_input: str, threshold: str = "BLOCK_NONE") -> dict:
        """
        Perform badness detection on the input text.

        Example:
        >>> text_badness("I hate you!", "BLOCK_ONLY_HIGH")
        {'harassment': False, 'harassment_threatening': False, ...}

        Args:
            user_input (str): A string representing the input text.
            threshold (str): A string representing the threshold for badness detection. Values can be "BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE".

        Returns:
            dict: A dictionary containing the badness detection results.
        """

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
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


        prompt_template = f'''Given the input text paragraph(s):
        {user_input}

        __________________________

        Given a user input, determine the following moderation if it exist or not. Answer for each would be string True or False and dont provide any other information.
        harassment:
        harassment_threatening':
        hate:
        hate_threatening:
        self_harm:
        self_harm_instructions:
        self_harm_intent:
        sexual:
        violence:
        violence_graphic:
        self-harm:
        hate/threatening:
        violence/graphic:
        self-harm/intent:
        self-harm/instructions:
        harassment/threatening:
        '''

        try:
            # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.model.generate_content(prompt_template, safety_settings=safety_settings)
            # Split the string into lines and then split each line by ":"
            data_list = [line.strip().split(":") for line in response.text.strip().split("\n")]

            # Convert values to boolean
            data_dict = {key.strip(): value.strip() == 'True' for key, value in data_list}

            return data_dict

        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")


   ############## EMOJIS TEXT ############## 
    def text_emojis(self, user_input: str, high_cost: bool = False) -> str:
        """
        Detect the badness level of the input text.

        Example:
        >>> text_emojis("I love 🍕 and 🍔", high_cost=False)
        'I love pizza and burger'

        Args:
            user_input (str): A string representing the input text.
            high_cost (bool): A boolean value representing whether to use the high-cost model.

        Returns:
            str: A string representing the text with emojis replaced by their text representation.
        """

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(high_cost, bool):
            raise TypeError("high_cost must be of type bool")
        
        # check if parameters are not empty
        if not user_input:
            raise ValueError("user_input cannot be empty")
        

        try:
            if high_cost:
                prompt_template = f'''Given the input text:
                user input: {user_input}
                Identify the emojis in the text and replace them with their text representation.
                output must be the updated text with emojis replaced by their text representation.
                output must not contain any other information than the updated text.
                '''
                response = self.model.generate_content(prompt_template)
                return response.text

            else:
                x = []
                y = ''
                for c in user_input.split():
                    if c in emoji.EMOJI_DATA:
                        # get c and its index
                        x.append((user_input.index(c)))
                        y += c+'\n'

                if x == []:
                    return "No emojis found in the text."
                prompt_template = f'''Given the below emojis
                {y}
                Replace each emoji with its text representation in the same order
                output format must be like this:
                text respresentation
                ...
                output must mot contain any other information than the format
                '''

                response = self.model.generate_content(prompt_template)

                ya = response.text.split('\n')

                # Convert the text string into a list of characters
                text_list = list(user_input)

                # Replace emojis at positions specified in x with corresponding items from ya
                for i in range(len(x)):
                    if x[i] < len(text_list):
                        text_list[x[i]] = ya[i]

                # Join the list of characters back into a string
                modified_text = ''.join(text_list)

                return modified_text

        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")

    ############## TEXT IDIOMS ##############
    def text_idioms(self, user_input: str) -> list:
        """
        Identify and extract any idioms present in the given sentence.

        Example:
        >>> text_idioms("I am over the moon")
        ['over the moon']

        Args:
            user_input (str): A string representing the input sentence.

        Returns:
            list: A list of strings containing the extracted idioms from the input sentence.
        """

        prompt_template = f'''Given the input sentence:
        user input: {user_input}

        __________________________

        Identify and extract any idioms present in the sentence.
        Output must only contain the extracted idioms.
        Output must not contain any bullet points.
        If there is more than one idiom found, return both in new lines.
        If no idiom is found, return None.

        output must be in this format:
        extracted idioms
        ...
        output must not contain any other information than the extracted idioms.
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

    ############## TEXT ANOMALY ##############
    def text_anomaly(self, user_input: str) -> list:
        """
        Detect any anomalies or outliers in the given input text.

        Example:
        >>> text_anomaly("man is walking on the road while eating a book")
        ['eating a book']

        Args:
            user_input (str): A string representing the input sentence.

        Returns:
            list: A list of strings containing the detected anomalies or outliers in the input text.
        """

        prompt_template = f'''Given the input sentence:
        user input: {user_input}
        Detect any anomalies or outliers in the text.
        output only the detected anomalies and dont provide any other information
        dont use bullet points or any other formatting

        output must be in this format:
        detected anomalies
        ...
        output must not contain any other information than the detected anomalies.
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        
        try:
            response = self.model.generate_content(prompt_template)
            return response.text.split("\n")
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")


    ############## TEXT COREFERENCE ##############
    def text_coreference(self, user_input: str) -> dict:
        """
        Perform coreference resolution on the input text to identify the resolved references for each pronoun.

        Example:
        >>> text_coreference("John loves his dog. He takes care of it.")
        {'his': 'John', 'it': 'dog'}

        Args:
            user_input (str): A string representing the input paragraph.

        Returns:
            dict: A dictionary containing the resolved references for each pronoun in the input text.
        """
        
        prompt_template = f'''Given the input paragraph:
        user input: {user_input}

        __________________________

        Perform coreference resolution to identify what each pronoun in the paragraph is referring to. 
        Output must only contain the resolved references for each pronoun, without any additional context.
        Output must not contain any bullet points.
        If no referent is found for a pronoun, return 'None'.

        Output must be in this format:
        Pronoun: Referent
        ...
        outpust must not contain anyother information.
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")

        try:
                
            response = self.model.generate_content(prompt_template)

            # Extract resolved coreferences
            coreferences = response.text.split("\n")
            coreferences = [coreference for coreference in coreferences if coreference]
            coreferences_dict = {item.split(': ')[0]: item.split(': ')[1] for item in coreferences}

            return coreferences_dict
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")


    ############## PDF CHAT ##############
    def chat_pdf(self, user_input: str, pdf_path: str, pages: str = "all") -> str:
        """
        Answer the given question based on the information extracted from the provided PDF file.

        Example:
        >>> chat_pdf("Who is the CEO of Apple Inc.?", "apple.pdf", "1-2")
        'Tim Cook'

        Args:
            user_input (str): A string representing the question to be answered based on the information extracted from the PDF file.
            pdf_path (str): A string representing the path to the PDF file.
            pages (str): A string representing the range of pages to extract information from. If set to "all", extract information from all pages. format should be "integer-integer".
        
        Returns:
            str: A string representing the answer to the question based on the information extracted from the PDF file.
        """

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(pdf_path, str):
            raise TypeError("pdf_path must be of type str")
        if not isinstance(pages, str):
            raise TypeError("pages must be of type str")
        
        # check if parameters are not empty
        if not user_input:
            raise ValueError("user_input cannot be empty")
        if not pdf_path:
            raise ValueError("pdf_path cannot be empty")
        
        
        if pages != "all":
            # pages string must patch this format integer-integer
            if "-" not in pages:
                raise ValueError("Please provide pages in the format 'integer-integer'")
            pages = pages.split("-")
            pdf_text = ''
            doc = fitz.open(pdf_path)
            for page_num in range(int(pages[0]), int(pages[1])):
                doc = doc.load_page(page_num)
                pdf_text += doc.get_text()
        else:
            pdf_text = ''
            doc = fitz.open(pdf_path)
            for page in doc:
                pdf_text += page.get_text()


        
        prompt_template = f'''Given the input paragraph:
        Information: {pdf_text}

        __________________________

        Answer the below question based on the information provided. if information is not enough to answer the question, return No information found.
        {user_input}
        '''

        try:
            response = self.model.generate_content(prompt_template)
            return response.text
        except Exception as e:
            raise ValueError("Please provide a correct PdF path/API Key or try again.")

    ############## TEXT OCR ##############
    def text_ocr(self, image_path: str, user_input: str) -> str:
        """
        Extract text from an image using Optical Character Recognition (OCR).

        Example:
        >>> text_ocr("image.jpg", "Extract text from the image.")
        'The extracted text from the image.'

        Args:
            image_path (str): A string representing the path to the image file.
            prompt (str): A string representing the prompt to extract text from the image.

        Returns:
            str: A string representing the extracted text from the image.
        """

        prompt_template = f'''{user_input}
        if no text found returns None
        '''

        # check if parameters are of correct type
        if not isinstance(image_path, str):
            raise TypeError("image_path must be of type str")
        if not isinstance(user_input, str):
            raise TypeError("prompt must be of type str")
        
        # check if parameters are not empty
        if not image_path:
            raise ValueError("image_path cannot be empty")
        if not user_input:
            raise ValueError("prompt cannot be empty")

        try:
            # Import the Image class from IPython.display
            from IPython.display import Image

            # Load the image using IPython.display.Image
            img = Image(filename=image_path)

            # if the image is not found, raise an error
            if not img:
                raise FileNotFoundError("The image file was not found.")
            
            # Generate content to extract text from the image
            response = self.v_model.generate_content([prompt_template, img])

            # Return the extracted text
            return response.text
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")
        

    ############## TEXT SENTIMENT ##############
    def text_sentiment(self, user_input: str, num_classes: str = "positive, negative, neutral", explanation: bool = True) -> dict:
        """
        Perform sentiment detection on the input text.

        Example:
        >>> text_sentiment("I love this product. It works great!", "positive, negative, neutral", explanation=False)
        {'prediction': 'positive'}

        Args:
            user_input (str): A string representing the input text.
            num_classes (str): A string representing the number of classes for sentiment detection.
            explanation (bool): A boolean value representing whether to include explanation in the output.

        Returns:
            dict: A dictionary containing the prediction of the sentiment detection.
        """
        if explanation:
            format_answer = "prediction:your_answer\nexplanation:your_answer"
        else:
            format_answer = "prediction:your_answer\n"
        
        # Question to be asked
        prompt_template = f'''Given the input text, perform sentiment detection on it
        {user_input}
        num_classes: {num_classes}
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
            # # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.model.generate_content(prompt_template)

            result = {}
            for line in response.text.split("\n"):
                if line.strip():
                    key, value = line.split(": ", 1)
                    result[key.strip()] = value.strip()

            return result
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")

############## ANYSCALE LINGUA CLASS ##############
class AnyScaleLingua:


    ############## INITIALIZATION ##############
    def __init__(self, api_key: str, model_name: str = "meta-llama/Llama-3-70b-chat-hf"):
        """
        Initializes the OpenAILingua class with the provided API key and model names (optional).

        Args:
            api_key (str): A string representing the API key for the Generative AI model.
            model_name (str): If not provided, the default value is "meta-llama/Llama-3-70b-chat-hf". Supported models and their pricing is found at documenation.
        """

        # Configuring Generative AI with API key
        self.client = OpenAI(base_url = "https://api.endpoints.anyscale.com/v1", api_key=api_key)


        # text generation model
        self.model_name = model_name

    ############## EXTRACT PATTERNS ##############
    def extract_patterns(self, user_input: str, patterns: str) -> dict:
        """
        Extracts patterns from the given input.

        Example:
        >>> extract_patterns("The phone number of fareed khan and asad are 123-67-325 and 242-176-921", "phone numbers, person names")
        {'phone numbers': ['123-67-325', '242-176-921'], 'person names': ['fareed khan', 'asad']}

        Args:
            user_input (str): A string representing the input sentence.
            patterns (str): A string of comma-separated patterns to be extracted from the input.

        Returns:
            dict: A dictionary containing the extracted patterns with their values.
        """

        # Generate the prompt template
        prompt_template = f'''Given the input text: 
        {user_input}
        Extract the following patterns from it: {patterns}

        Output the result as a Python dictionary. Do not include any additional text, explanations, or markdown formatting. Only the dictionary should be present in the output.
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(patterns, str):
            raise TypeError("patterns must be of type str")
        

        # check if parameters are not empty
        if not user_input:
            raise ValueError("user_input cannot be empty")
        if not patterns:
            raise ValueError("patterns cannot be empty")
        
        # running Gemini model
        try:
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": "you are a NLP python library that only returns that asked formatted answer but not a chatbot like response"},{"role": "user", "content": prompt_template}],
            )
            response1 = response.choices[0].message.content
            # Use a regular expression to capture everything between the first and last curly braces
            pattern = r"\{.*\}"
            match = re.search(pattern, response1, re.DOTALL)

            if match:
                dict_str = match.group(0)  # Get the matched portion
            else:
                return response1

            try:
                extracted_dict = ast.literal_eval(dict_str)  # Safely evaluate the string to a dictionary
            except (SyntaxError, ValueError) as e:
                return response1
            
            return extracted_dict

        except Exception as e:
            raise ValueError("Please provide correct AnyScale API key or try again")

    ############## TRANSLATE TEXT ##############
    def text_translate(self, user_input: str, target_lang: str) -> str:
        """
        Translates the given input text into the target language.

        Example:
        >>> text_translate("Farid khan and asad are coming to my house at 5 pm", "urdu")
        'فرید خان اور اسد شام 5 بجے میرے گھر آ رہے ہیں'

        Args:
            user_input (str): A string representing the input sentence.
            target_lang (str): A string representing the target language to translate the input into.

        Returns:
            str: A string representing the translated text.
        """

        # Generate the prompt template
        prompt_template = f'''
        Given the input text:
        user input: {user_input}

        convert it into {target_lang} language
        output must contain only the translated text
        '''
        
        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(target_lang, str):
            raise TypeError("target_lang must be of type str")
        
        # check if parameters are not empty
        if not user_input:
            raise ValueError("user_input cannot be empty")
        if not target_lang:
            raise ValueError("target_lang cannot be empty")
        
        try:
            # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": "you are a NLP python library that only returns that asked formatted answer but not a chatbot like response"},{"role": "user", "content": prompt_template}],
            )
            response1 = response.choices[0].message.content
            return response1
        except Exception as e:
            return "Translation failed. Only the most popular languages are supported. Actively working to add more."

    ############## TEXT REPLACEMENT ##############
    def text_replace(self, user_input: str, replacement_rules: str) -> str:
        """
        Replaces words in the given input text according to the replacement rules provided.
        
        Example:
        >>> text_replace("Fareed Phone number is 123-67-325", "sensitive information such as phone numbers:XXXXX")
        'Fareed Phone number is XXXXX'
        
        Args:
            user_input (str): A string representing the input sentence.
            replacement_rules (str): A string representing the replacement rules in the format "word_to_replace:replacement_word".

        Returns:
            str: A string representing the modified text with replacements.
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

        output format:
        word_to_replace: replacement_word

        output must be in dictionary format, dont provide output text
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(replacement_rules, str):
            raise TypeError("replacement_rules must be of type str")

        # running Gemini model
        try:
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": "you are a NLP python library that only returns that asked formatted answer but not a chatbot like response"},{"role": "user", "content": prompt_template}],
            )
            response1 = response.choices[0].message.content
            # Use a regular expression to capture everything between the first and last curly braces
            pattern = r"\{.*\}"
            match = re.search(pattern, response1, re.DOTALL)

            if match:
                dict_str = match.group(0)  # Get the matched portion
            else:
                return response1

            try:
                extracted_dict = ast.literal_eval(dict_str)  # Safely evaluate the string to a dictionary
            except (SyntaxError, ValueError) as e:
                return response1
            
            
            for word, replacement in extracted_dict.items():
                user_input = user_input.replace(word, replacement)
            
            return user_input

        except Exception as e:
            raise ValueError("Please provide correct AnyScale API key or try again")

    ############## NER EXTRACTION ##############
    def detect_ner(self, user_input: str, ner_tags: str = "") -> dict:
        """
        Perform Named Entity Recognition (NER) detection on the input text.

        Example:
        >>> detect_ner("I love Lamborghini, but Bugatti is even better. Although, Mercedes is a class above all. and i work in Google", "cars, date, time")
        {'cars': ['Lamborghini', 'Bugatti', 'Mercedes'], 'date': [], 'time': []}

        Args:
            user_input (str): A string representing the input sentence.
            ner_tags (str): A string of comma-separated NER tags to be detected from the input.

        Returns:
            dict: A dictionary containing the detected NER tags with their values.
        """

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(ner_tags, str):
            raise TypeError("ner_tags must be of type str")
        
        # check if parameters are not empty
        if not user_input:
            raise ValueError("user_input cannot be empty")
        
        # user ner tags
        if ner_tags != "":
            user_ner_tags = f'''NER TAGS: {ner_tags}'''
        else:
            user_ner_tags = f'''NER TAGS: FAC, CARDINAL, NUMBER, DEMONYM, QUANTITY, TITLE, PHONE_NUMBER, NATIONAL, JOB, PERSON, LOC, NORP, TIME, CITY, EMAIL, GPE, LANGUAGE, PRODUCT, ZIP_CODE, ADDRESS, MONEY, ORDINAL, DATE, EVENT, CRIMINAL_CHARGE, STATE_OR_PROVINCE, RELIGION, DURATION, URL, WORK_OF_ART, PERCENT, CAUSE_OF_DEATH, COUNTRY, ORG, LAW, NAME, COUNTRY, RELIGION, TIME'''

        # Generate the prompt template
        prompt_template = f'''Given the input text:
        user input: {user_input}

        perform NER detection on it.
        {user_ner_tags}

        Output the result as a Python dictionary. Do not include any additional text, explanations, or markdown formatting. Only the dictionary should be present in the output.
        multiple entities belong to same category should be separated by comma in a list
        '''

        # running Gemini model
        try:
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": "you are a NLP python library that only returns that asked formatted answer but not a chatbot like response"},{"role": "user", "content": prompt_template}],
            )
            response1 = response.choices[0].message.content
            # Use a regular expression to capture everything between the first and last curly braces
            pattern = r"\{.*\}"
            match = re.search(pattern, response1, re.DOTALL)

            if match:
                dict_str = match.group(0)  # Get the matched portion
            else:
                return response1

            try:
                extracted_dict = ast.literal_eval(dict_str)  # Safely evaluate the string to a dictionary
            except (SyntaxError, ValueError) as e:
                return response1
            
            return extracted_dict

        except Exception as e:
            raise ValueError("Please provide correct AnyScale API key or try again")
        
    ############## TEXT SUMMARIZATION ##############
    def text_summarize(self, user_input: str, summary_length: int = 4) -> str:
        """
        Generate a summary of the input text.

        Example:
        >>> text_summarize("Beneath the canopy of a vast and ancient forest, where the towering trees ...", 1)
        'The moon, a luminous orb hanging in the sky like a celestial lantern, casts its gentle light upon the forest floor, illuminating the intricate tapestry of life that thrives in the shadows.'

        Args:
            user_input (str): A string representing the input text.
            summary_length (int): An integer representing the length of the summary.

        Returns:
            str: A string representing the summary of the input text.
        """
        
        # Define the prompt template
        prompt_template = f'''
        Given the input text:
        Input: {user_input}
        Produce a {summary_length} sentences length summary of the text.
        Captures the main ideas and key points of the text.
        Summary Does not include verbatim sentences from the original text.
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(summary_length, int):
            raise TypeError("summary_length must be of type int")

        try:
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": "you are a NLP python library that only returns that asked formatted answer but not a chatbot like response"},{"role": "user", "content": prompt_template}],
            )
            response1 = response.choices[0].message.content
            
            return response1

        except Exception as e:
            raise ValueError("Please provide correct AnyScale API key or try again")

    ############## TEXT QUESTION ANSWERING ##############
    def text_qna(self, user_input: str, question: str) -> str:
        """
        answer the given question based on the input text.

        Example:
        >>> text_qna("OpenAI has hosted a hackathon for developers to build AI models. The event took place on 15th October 2022. The event was a huge success with over 1000 participants from around the world.", "When did the event happen?")
        '15th October 2022'

        Args:
            user_input (str): A string representing the input text.
            question (str): A string representing the question to be answered based on the input text.

        Returns:
            str: A string representing the answer to the question based on the input text.
        """
        # Define the prompt template
        prompt_template = f'''
        Given the input text:
        Input: {user_input}
        Answer the following question: 
        {question}
        The answer should be relevant and concise, without any additional information.
        Ensure that the answer directly addresses the question.
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(question, str):
            raise TypeError("question must be of type str")

        try:
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": "you are a NLP python library that only returns that asked formatted answer but not a chatbot like response"},{"role": "user", "content": prompt_template}],
            )
            response1 = response.choices[0].message.content
            return response1

        except Exception as e:
            raise ValueError("Please provide correct AnyScale API key or try again")

    ############## TEXT INTENT DETECTION ##############
    def text_intent(self, user_input: str) -> list:
        """
        Identify the intent of the user input.

        Example:
        >>> text_intent("let's book a flight for our vacation and reserve a table at a restaurant for dinner. also going to watch football match at 8 pm.")
        ['book a flight', 'reserve a table', 'watch football match']

        Args:
            user_input (str): A string representing the input sentence.

        Returns:
            list: A list of strings representing the detected intents from the input text.
        """
            
        # Define the prompt template
        prompt_template = f'''
        Given the input sentence:
        user input: {user_input}
        Identify the intent of the text.
        If no clear intent can be determined from the input, return None.
        output must be a python list of strings
        output format: ['intent1', 'intent2', ...]
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        

        try:
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": "you are a NLP python library that only returns that asked formatted answer but not a chatbot like response"},{"role": "user", "content": prompt_template}],
            )
            response1 = response.choices[0].message.content
            # Use a regular expression to capture everything between the first and last curly braces
            pattern = r"\[.*\]"
            match = re.search(pattern, response1, re.DOTALL)

            if match:
                dict_str = match.group(0)  # Get the matched portion
            else:
                return response1

            try:
                extracted_list = ast.literal_eval(dict_str)  # Safely evaluate the string to a dictionary
            except (SyntaxError, ValueError) as e:
                return response1
            
            return extracted_list

        except Exception as e:
            raise ValueError("Please provide correct AnyScale API key or try again")

    ############## TEXT EMBEDDING ##############
    def text_embedd(self, user_input: str, embedd_model: str = "thenlper/gte-large") -> list:
        """
        Generate embeddings for the given input text.

        Example:
        >>> text_embedd("OpenAI has hosted a hackathon for developers to build AI models. The event took place on 15th October 2022. The event was a huge success with over 1000 participants from around the world.", "thenlper/gte-large")
        [-0.007, 0.003, 0.002, 0.001, 0.003, 0.002, 0.002, 0.001, 0.002, 0.002 ...]

        Args:
            user_input (str): A string representing the input text.
            embedd_model (str): A string representing the embedding model to use. If not provided, the default value is "thenlper/gte-large".

        Returns:
            list: A list representing the embeddings of the input text.
        """

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(embedd_model, str):
            raise TypeError("embedd_model must be of type str")
        
        try:
            response = self.client.embeddings.create(
            input=user_input,
            model=embedd_model,
            )
            return response.data[0].embedding
        except Exception as e:
            raise ValueError("Please provide correct AnyScale API key or try again")
        
    ############## TEXT SPAM ##############
    def detect_spam(self, user_input: str, num_classes: str = "spam, not_spam, unknown") -> dict:
        """
        Perform spam detection on the input text.

        Example:
        >>> detect_spam("Congratulations! You have won a lottery of $1,000,000!", "harmed, not_harmed, unknown")
        {'prediction': 'harmed'}

        Args:
            user_input (str): A string representing the input text.
            num_classes (str): A string representing the number of classes for spam detection.

        Returns:
            dict: A dictionary containing the prediction of the spam detection.
        """
        
        # Question to be asked
        prompt_template = f'''Given the input text, perform spam detection on it
        {user_input}
        num_classes: {num_classes}
        answer must be a dictionary with two keys class and explanation
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(num_classes, str):
            raise TypeError("num_classes must be of type str")

        try:
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": "you are a NLP python library that only returns that asked formatted answer but not a chatbot like response"},{"role": "user", "content": prompt_template}],
            )
            response1 = response.choices[0].message.content
            # Use a regular expression to capture everything between the first and last curly braces
            pattern = r"\{.*\}"
            match = re.search(pattern, response1, re.DOTALL)

            if match:
                dict_str = match.group(0)  # Get the matched portion
            else:
                return response1

            try:
                extracted_dict = ast.literal_eval(dict_str)  # Safely evaluate the string to a dictionary
            except (SyntaxError, ValueError) as e:
                return response1
            
            return extracted_dict

        except Exception as e:
            raise ValueError("Please provide correct AnyScale API key or try again")

    ############## TEXT SPELLCHECK ##############
    def text_spellcheck(self, user_input: str) -> str:
        """
        Correct the spelling of the input text.

        Example:
        >>> text_spellcheck("I am a good speler")
        'I am a good speller'

        Args:
            user_input (str): A string representing the input text.

        Returns:
            str: A string representing the corrected spelling of the input text.
        """
        
        prompt_template = f'''Given the input text:
        user input: {user_input}
        output must be in this format:
        misspelled_word:corrected_word
        ...
        output must not contain any other information than the format
        output must be in dictionary format, dont provide output text
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")

        try:
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": "you are a NLP python library that only returns that asked formatted answer but not a chatbot like response"},{"role": "user", "content": prompt_template}],
            )
            response1 = response.choices[0].message.content
            # Use a regular expression to capture everything between the first and last curly braces
            pattern = r"\{.*\}"
            match = re.search(pattern, response1, re.DOTALL)

            if match:
                dict_str = match.group(0)  # Get the matched portion
            else:
                return response1

            try:
                extracted_dict = ast.literal_eval(dict_str)  # Safely evaluate the string to a dictionary
            except (SyntaxError, ValueError) as e:
                return response1
            
            for word, replacement in extracted_dict.items():
                user_input = user_input.replace(word, replacement)
            
            return user_input

        except Exception as e:
            raise ValueError("Please provide correct AnyScale API key or try again")

    ############## TEXT SEMANTIC ROLE LABELING ##############
    def text_srl(self, user_input: str) -> dict:
        """
        Perform Semantic Role Labeling (SRL) on the input text.

        Example:
        >>> text_srl("John ate an apple")
        {'Predicate': 'ate', 'Agent': 'John', 'Theme': 'apple'}

        Args:
            user_input (str): A string representing the input sentence.

        Returns:
            dict: A dictionary containing the Predicate, Agent, and Theme of the input sentence.
        """

        prompt_template = f'''Given the input sentence:
        user input: {user_input}

        __________________________

        Perform Semantic Role Labeling (SRL) on the input sentence to identify the predicate, agent, and theme.
        - Predicate: The action or state described by the verb.
        - Agent: The entity performing the action.
        - Theme: The entity that is affected by the action.

        Ensure the output must be in dictionary format  with three keys Predicate Agent Theme
        - Predicate: [predicate]
        - Agent: [agent]
        - Theme: [theme]

        If any component is not present or cannot be identified, return None for that component.
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")

        try:
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": "you are a NLP python library that only returns that asked formatted answer but not a chatbot like response"},{"role": "user", "content": prompt_template}],
            )
            response1 = response.choices[0].message.content
            # Use a regular expression to capture everything between the first and last curly braces
            pattern = r"\{.*\}"
            match = re.search(pattern, response1, re.DOTALL)

            if match:
                dict_str = match.group(0)  # Get the matched portion
            else:
                return response1

            try:
                extracted_dict = ast.literal_eval(dict_str)  # Safely evaluate the string to a dictionary
            except (SyntaxError, ValueError) as e:
                return response1
            
            return extracted_dict

        except Exception as e:
            raise ValueError("Please provide correct AnyScale API key or try again")

    ############## TEXT SENTIMENT ##############
    def text_sentiment(self, user_input: str, num_classes: str = "positive, negative, neutral") -> dict:
        """
        Perform sentiment detection on the input text.

        Example:
        >>> text_sentiment("I love this product. It works great!", "positive, negative, neutral")
        {'prediction': 'positive'}

        Args:
            user_input (str): A string representing the input text.
            num_classes (str): A string representing the number of classes for sentiment detection.

        Returns:
            dict: A dictionary containing the prediction of the sentiment detection.
        """

        
        # Question to be asked
        prompt_template = f'''Given the input text, perform sentiment analysis on it
        {user_input}
        num_classes: {num_classes}
        answer must be a dictionary with two keys rediction and explanation
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(num_classes, str):
            raise TypeError("num_classes must be of type str")

        try:
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": "you are a NLP python library that only returns that asked formatted answer but not a chatbot like response"},{"role": "user", "content": prompt_template}],
            )
            response1 = response.choices[0].message.content
            # Use a regular expression to capture everything between the first and last curly braces
            pattern = r"\{.*\}"
            match = re.search(pattern, response1, re.DOTALL)

            if match:
                dict_str = match.group(0)  # Get the matched portion
            else:
                return response1

            try:
                extracted_dict = ast.literal_eval(dict_str)  # Safely evaluate the string to a dictionary
            except (SyntaxError, ValueError) as e:
                return response1
            
            return extracted_dict

        except Exception as e:
            raise ValueError("Please provide correct AnyScale API key or try again")

    ############## TEXT TOPIC ##############
    def text_topic(self, user_input: str) -> dict:
        """
        Perform topic detection on the input text.

        Example:
        >>> text_topic("OpenAI has hosted a hackathon for developers to build AI models. The event took place on 15th October 2022. The event was a huge success with over 1000 participants from around the world.")
        {'topic': 'technology'}

        Args:
            user_input (str): A string representing the input text.

        Returns:
            dict: A dictionary containing the detected topic from the input text.
        """
        
        # Question to be asked
        prompt_template = f'''Given the input text, perform topic detection  on it
        {user_input}
        answer must be a dictionary with two keys topic_name and explanation
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")

        try:
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": "you are a NLP python library that only returns that asked formatted answer but not a chatbot like response"},{"role": "user", "content": prompt_template}],
            )
            response1 = response.choices[0].message.content
            # Use a regular expression to capture everything between the first and last curly braces
            pattern = r"\{.*\}"
            match = re.search(pattern, response1, re.DOTALL)

            if match:
                dict_str = match.group(0)  # Get the matched portion
            else:
                return response1

            try:
                extracted_dict = ast.literal_eval(dict_str)  # Safely evaluate the string to a dictionary
            except (SyntaxError, ValueError) as e:
                return response1
            
            return extracted_dict

        except Exception as e:
            raise ValueError("Please provide correct AnyScale API key or try again")

    ############## POS DETECTION ##############
    def detect_pos(self, user_input: str, pos_tags: str = "") -> dict:
        """
        Perform Part-of-Speech (POS) detection on the input text.

        Example:
        >>> detect_pos("I love Lamborghini, but Bugatti is even better. Although, Mercedes is a class above all. and i work in Google", "noun, verb, adjective")
        {'noun': ['Lamborghini', 'Bugatti', 'Mercedes', 'Google'], 'verb': ['love', 'is', 'work'], 'adjective': ['better', 'above', 'all']}

        Args:
            user_input (str): A string representing the input sentence.
            pos_tags (str): A string of comma-separated POS tags to be detected from the input.

        Returns:
            dict: A dictionary containing the detected NER tags with their values.
        """

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(pos_tags, str):
            raise TypeError("ner_tags must be of type str")
        
        # check if parameters are not empty
        if not user_input:
            raise ValueError("user_input cannot be empty")
        
        # user ner tags
        if pos_tags != "":
            user_pos_tags = f'''POS TAGS: {pos_tags}'''
        else:
            user_pos_tags = f'''POS TAGS:noun, verb, adjective, adverb, pronoun, preposition, conjunction, interjection, determiner, cardinal, foreign, number, date, time, ordinal, money, percent, symbol, punctuation, emoticon, hashtag, email, url, mention, phone, ip, cashtag, entity,'''

        # Generate the prompt template
        prompt_template = f'''Given the input text:
        user input: {user_input}

        perform POS tagging on it.
        {user_pos_tags}

        Output the result as a Python dictionary. Do not include any additional text, explanations, or markdown formatting. Only the dictionary should be present in the output.
        multiple entities belong to same category should be separated by comma in a list
        '''

        # running Gemini model
        try:
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": "you are a NLP python library that only returns that asked formatted answer but not a chatbot like response"},{"role": "user", "content": prompt_template}],
            )
            response1 = response.choices[0].message.content
            # Use a regular expression to capture everything between the first and last curly braces
            pattern = r"\{.*\}"
            match = re.search(pattern, response1, re.DOTALL)

            if match:
                dict_str = match.group(0)  # Get the matched portion
            else:
                return response1

            try:
                extracted_dict = ast.literal_eval(dict_str)  # Safely evaluate the string to a dictionary
            except (SyntaxError, ValueError) as e:
                return response1
            
            return extracted_dict

        except Exception as e:
            raise ValueError("Please provide correct AnyScale API key or try again")
        
    ############## EMOJIS TEXT ############## 
    def text_emojis(self, user_input: str, high_cost: bool = False) -> str:
        """
        Detect the badness level of the input text.

        Example:
        >>> text_emojis("I love 🍕 and 🍔", high_cost=False)
        'I love pizza and burger'

        Args:
            user_input (str): A string representing the input text.
            high_cost (bool): A boolean value representing whether to use the high-cost model.

        Returns:
            str: A string representing the text with emojis replaced by their text representation.
        """

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(high_cost, bool):
            raise TypeError("high_cost must be of type bool")
        
        # check if parameters are not empty
        if not user_input:
            raise ValueError("user_input cannot be empty")
        

        try:
            if high_cost:
                prompt_template = f'''Given the input text:
                user input: {user_input}
                Identify the emojis in the text and replace them with their text representation.
                output must be the updated text with emojis replaced by their text representation.
                output must not contain any other information than the updated text.
                '''
                response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": "you are a NLP python library that only returns that asked formatted answer but not a chatbot like response"},{"role": "user", "content": prompt_template}],
                )
                response1 = response.choices[0].message.content

            else:
                x = []
                y = ''
                for c in user_input.split():
                    if c in emoji.EMOJI_DATA:
                        # get c and its index
                        x.append((user_input.index(c)))
                        y += c+'\n'

                if x == []:
                    return "No emojis found in the text."
                prompt_template = f'''Given the below emojis
                {y}
                Replace each emoji with its text representation in the same order
                output format must be like this:
                text respresentation
                ...
                output must mot contain any other information than the format
                '''

                response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": "you are a NLP python library that only returns that asked formatted answer but not a chatbot like response"},{"role": "user", "content": prompt_template}],
                )
                response1 = response.choices[0].message.content

                ya = response1.split('\n')

                # Convert the text string into a list of characters
                text_list = list(user_input)

                # Replace emojis at positions specified in x with corresponding items from ya
                for i in range(len(x)):
                    if x[i] < len(text_list):
                        text_list[x[i]] = ya[i]

                # Join the list of characters back into a string
                modified_text = ''.join(text_list)

                return modified_text

        except Exception as e:
            raise ValueError("Please provide a correct AnyScale API key or try again.")

    ############## TEXT IDIOMS ##############
    def text_idioms(self, user_input: str) -> list:
        """
        Identify and extract any idioms present in the given sentence.

        Example:
        >>> text_idioms("I am over the moon")
        ['over the moon']

        Args:
            user_input (str): A string representing the input sentence.

        Returns:
            list: A list of strings containing the extracted idioms from the input sentence.
        """

        prompt_template = f'''Given the input sentence:
        user input: {user_input}

        __________________________

        Identify and extract any idioms present in the sentence.
        Output must only contain the extracted idioms.
        Output must not contain any bullet points.
        If there is more than one idiom found, return both in new lines.
        If no idiom is found, return None.

        output must be a python list and must not contain any other information.
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")

        try:
            # Generate response using the provided model (assuming it's defined elsewhere)            
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": "you are a NLP python library that only returns that asked formatted answer but not a chatbot like response"},{"role": "user", "content": prompt_template}],
            )
            response1 = response.choices[0].message.content

                        # Use a regular expression to capture everything between the first and last curly braces
            pattern = r"\[.*\]"
            match = re.search(pattern, response1, re.DOTALL)

            if match:
                dict_str = match.group(0)  # Get the matched portion
            else:
                return response1

            try:
                extracted_list = ast.literal_eval(dict_str)  # Safely evaluate the string to a dictionary
            except (SyntaxError, ValueError) as e:
                return response1
            
            return extracted_list
        except Exception as e:
            raise ValueError("Please provide a correct AnyScale API key or try again.")

    ############## TEXT ANOMALY ##############
    def text_anomaly(self, user_input: str) -> list:
        """
        Detect any anomalies or outliers in the given input text.

        Example:
        >>> text_anomaly("man is walking on the road while eating a book")
        ['eating a book']

        Args:
            user_input (str): A string representing the input sentence.

        Returns:
            list: A list of strings containing the detected anomalies or outliers in the input text.
        """

        prompt_template = f'''Given the input sentence:
        user input: {user_input}
        Detect any anomalies or outliers in the text.
        output only the detected anomalies and dont provide any other information
        dont use bullet points or any other formatting
        output must be a python list
        output must not contain any other information. if no anomaly found return None
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        
        try:
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": "you are a NLP python library that only returns that asked formatted answer but not a chatbot like response"},{"role": "user", "content": prompt_template}],
            )

            response1 = response.choices[0].message.content

            # convert string of list to list
            # Use a regular expression to capture everything between the first and last curly braces
            pattern = r"\[.*\]"
            match = re.search(pattern, response1, re.DOTALL)

            if match:
                dict_str = match.group(0)  # Get the matched portion
            else:
                return response1

            try:
                extracted_list = ast.literal_eval(dict_str)  # Safely evaluate the string to a dictionary
            except (SyntaxError, ValueError) as e:
                return response1
            
            return extracted_list
        except Exception as e:
            raise ValueError("Please provide a correct AnyScale API key or try again.")

    ############## TEXT COREFERENCE ##############
    def text_coreference(self, user_input: str) -> dict:
        """
        Perform coreference resolution on the input text to identify the resolved references for each pronoun.

        Example:
        >>> text_coreference("John loves his dog. He takes care of it.")
        {'his': 'John', 'it': 'dog'}

        Args:
            user_input (str): A string representing the input paragraph.

        Returns:
            dict: A dictionary containing the resolved references for each pronoun in the input text.
        """
        
        prompt_template = f'''Given the input paragraph:
        user input: {user_input}

        __________________________

        Perform coreference resolution to identify what each pronoun in the paragraph is referring to.
        Output must not contain any bullet points.

        Output must be in python dictionary
        ...
        outpust must not contain anyother information.
        If no referent/pronoun is found for a pronoun, return None. 
        '''


        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")

        # running Gemini model
        try:
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": "you are a NLP python library that only returns that asked formatted answer but not a chatbot like response"},{"role": "user", "content": prompt_template}],
            )
            response1 = response.choices[0].message.content
            # Use a regular expression to capture everything between the first and last curly braces
            pattern = r"\{.*\}"
            match = re.search(pattern, response1, re.DOTALL)

            if match:
                dict_str = match.group(0)  # Get the matched portion
            else:
                return response1

            try:
                extracted_dict = ast.literal_eval(dict_str)  # Safely evaluate the string to a dictionary
            except (SyntaxError, ValueError) as e:
                return response1
            
            return extracted_dict

        except Exception as e:
            raise ValueError("Please provide correct AnyScale API key or try again")
 
    ############## PDF CHAT ##############
    def chat_pdf(self, user_input: str, pdf_path: str, pages: str = "all") -> str:
        """
        Answer the given question based on the information extracted from the provided PDF file.

        Example:
        >>> chat_pdf("Who is the CEO of Apple Inc.?", "apple.pdf", "1-2")
        'Tim Cook'

        Args:
            user_input (str): A string representing the question to be answered based on the information extracted from the PDF file.
            pdf_path (str): A string representing the path to the PDF file.
            pages (str): A string representing the range of pages to extract information from. If set to "all", extract information from all pages. format should be "integer-integer".
        
        Returns:
            str: A string representing the answer to the question based on the information extracted from the PDF file.
        """

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(pdf_path, str):
            raise TypeError("pdf_path must be of type str")
        if not isinstance(pages, str):
            raise TypeError("pages must be of type str")
        
        # check if parameters are not empty
        if not user_input:
            raise ValueError("user_input cannot be empty")
        if not pdf_path:
            raise ValueError("pdf_path cannot be empty")
        
        
        if pages != "all":
            docs = []
            # pages string must patch this format integer-integer
            if "-" not in pages:
                raise ValueError("Please provide pages in the format 'integer-integer'")
            pages = pages.split("-")
            pdf_text = ''
            doc = fitz.open(pdf_path)
            for page_num in range(int(pages[0]), int(pages[1])):
                doc = doc.load_page(page_num)
                pdf_text += doc.get_text()
        else:
            pdf_text = ''
            doc = fitz.open(pdf_path)
            for page in doc:
                pdf_text += page.get_text()


        
        prompt_template = f'''Given the input paragraph:
        Information: {pdf_text}

        __________________________

        Answer the below question based on the information provided. if information is not enough to answer the question, return No information found.
        {user_input}
        '''

        try:
                
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": "you are a NLP python library that only returns that asked formatted answer but not a chatbot like response"},{"role": "user", "content": prompt_template}],
            )
            response1 = response.choices[0].message.content

            return response1
        except Exception as e:
            raise ValueError("Please provide a correct PdF path/API Key or try again.")

############## OPENAI LINGUA CLASS ##############
class OpenAILingua:

    ############## INITIALIZATION ##############
    def __init__(self, api_key: str, model_name: str = 'gpt-3.5-turbo-0125', vision_model_name: str = 'gpt-4-turbo'):
        """
        Initializes the OpenAILingua class with the provided API key and model names (optional).

        Args:
            api_key (str): A string representing the API key for the Generative AI model.
            model_name (str): If not provided, the default value is 'gpt-3.5-turbo-0125'.
            vision_model_name (str): If not provided, the default value is 'gpt-4-turbo'.
        """

        # Configuring Generative AI with API key
        self.client = OpenAI(api_key=api_key)


        # text generation model
        self.model_name = model_name
        
        # vision model
        self.vision_model_name = vision_model_name


    ############## EXTRACT PATTERNS ##############
    def extract_patterns(self, user_input: str, patterns: str) -> dict:
        """
        Extracts patterns from the given input.

        Example:
        >>> extract_patterns("The phone number of fareed khan and asad are 123-67-325 and 242-176-921", "phone numbers, person names")
        {'phone numbers': ['123-67-325', '242-176-921'], 'person names': ['fareed khan', 'asad']}

        Args:
            user_input (str): A string representing the input sentence.
            patterns (str): A string of comma-separated patterns to be extracted from the input.

        Returns:
            dict: A dictionary containing the extracted patterns with their values.
        """

        # Generate the prompt template
        prompt_template = f'''
        Given the input text:
        user input: {user_input}

        extract following patterns from it: {patterns}

        output must be a python dictionary with keys as patterns and values as list of extracted patterns
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(patterns, str):
            raise TypeError("patterns must be of type str")
        

        # check if parameters are not empty
        if not user_input:
            raise ValueError("user_input cannot be empty")
        if not patterns:
            raise ValueError("patterns cannot be empty")
        
        # running Gemini model
        try:
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt_template}],
            )
            response1 = response.choices[0].message.content
            # convert x to a dictionary and values must be a list
            pattern = r"\{.*\}"
            match = re.search(pattern, response1, re.DOTALL)

            if match:
                dict_str = match.group(0)  # Get the matched portion
            else:
                return response1

            try:
                extracted_dict = ast.literal_eval(dict_str)  # Safely evaluate the string to a dictionary
            except (SyntaxError, ValueError) as e:
                return response1
            
            return extracted_dict
        except Exception as e:
            raise ValueError("Please provide correct API key or try again")
        

    ############## TRANSLATE TEXT ##############
    def text_translate(self, user_input: str, target_lang: str) -> str:
        """
        Translates the given input text into the target language.

        Example:
        >>> text_translate("Farid khan and asad are coming to my house at 5 pm", "urdu")
        'فرید خان اور اسد شام 5 بجے میرے گھر آ رہے ہیں'

        Args:
            user_input (str): A string representing the input sentence.
            target_lang (str): A string representing the target language to translate the input into.

        Returns:
            str: A string representing the translated text.
        """

        # Generate the prompt template
        prompt_template = f'''
        Given the input text:
        user input: {user_input}

        convert it into {target_lang} language
        output must contain only the translated text
        '''
        
        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(target_lang, str):
            raise TypeError("target_lang must be of type str")
        
        # check if parameters are not empty
        if not user_input:
            raise ValueError("user_input cannot be empty")
        if not target_lang:
            raise ValueError("target_lang cannot be empty")
        
        try:
            # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt_template}],
            )
            response1 = response.choices[0].message.content
            return response1
        except Exception as e:
            return "Translation failed. Only the most popular languages are supported. Actively working to add more."
        

    ############## TEXT REPLACEMENT ##############
    def text_replace(self, user_input: str, replacement_rules: str) -> str:
        """
        Replaces words in the given input text according to the replacement rules provided.
        
        Example:
        >>> text_replace("Fareed Phone number is 123-67-325", "sensitive information such as phone numbers:XXXXX")
        'Fareed Phone number is XXXXX'
        
        Args:
            user_input (str): A string representing the input sentence.
            replacement_rules (str): A string representing the replacement rules in the format "word_to_replace:replacement_word".

        Returns:
            str: A string representing the modified text with replacements.
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

        output format:
        word_to_replace: replacement_word
        '''


        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(replacement_rules, str):
            raise TypeError("replacement_rules must be of type str")

        try:
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt_template}],
            )
            response1 = response.choices[0].message.content
            # convert x to a dictionary and values must be a list
            response = response1.split('\n')

            # remove white spaces from the list
            response = [line.strip() for line in response]

            # Create a list comprehension to extract key-value pairs from each line
            result = [(line.split(":")[0], line.split(":")[1]) for line in response if line.strip()]

            # Replace the misspelled words with the corrected words
            for word, corrected_word in result:
                corrected_user_input = user_input.replace(word, corrected_word)

            return corrected_user_input
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")


    ############## NER EXTRACTION ##############
    def detect_ner(self, user_input: str, ner_tags: str = "") -> dict:
        """
        Perform Named Entity Recognition (NER) detection on the input text.

        Example:
        >>> detect_ner("I love Lamborghini, but Bugatti is even better. Although, Mercedes is a class above all. and i work in Google", "cars, date, time")
        {'cars': ['Lamborghini', 'Bugatti', 'Mercedes'], 'date': [], 'time': []}

        Args:
            user_input (str): A string representing the input sentence.
            ner_tags (str): A string of comma-separated NER tags to be detected from the input.

        Returns:
            dict: A dictionary containing the detected NER tags with their values.
        """

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(ner_tags, str):
            raise TypeError("ner_tags must be of type str")
        
        # check if parameters are not empty
        if not user_input:
            raise ValueError("user_input cannot be empty")
        
        # user ner tags
        if ner_tags != "":
            user_ner_tags = f'''NER TAGS: {ner_tags}'''
        else:
            user_ner_tags = f'''NER TAGS: FAC, CARDINAL, NUMBER, DEMONYM, QUANTITY, TITLE, PHONE_NUMBER, NATIONAL, JOB, PERSON, LOC, NORP, TIME, CITY, EMAIL, GPE, LANGUAGE, PRODUCT, ZIP_CODE, ADDRESS, MONEY, ORDINAL, DATE, EVENT, CRIMINAL_CHARGE, STATE_OR_PROVINCE, RELIGION, DURATION, URL, WORK_OF_ART, PERCENT, CAUSE_OF_DEATH, COUNTRY, ORG, LAW, NAME, COUNTRY, RELIGION, TIME'''

        # Generate the prompt template
        prompt_template = f'''Given the input text:
        user input: {user_input}
        
        perform NER detection on it.
        {user_ner_tags}
        answer must be in the format
        tag:value
        '''

        try:
            # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt_template}],
            )
            response1 = response.choices[0].message.content

            # convert x to a dictionary and values must be a list
            x = response1.split('\n')
            x = [i.split(':') for i in x if i]
            x = {i[0].strip(): [j.strip() for j in i[1].split(',')] for i in x}
            return x

        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")
        
    ############## TEXT SUMMARIZATION ##############
    def text_summarize(self, user_input: str, summary_length: int = 4) -> str:
        """
        Generate a summary of the input text.

        Example:
        >>> text_summarize("Beneath the canopy of a vast and ancient forest, where the towering trees stand as silent sentinels of time, the night unfolds with a grandeur unmatched by any other hour. The moon, a luminous orb hanging in the sky like a celestial lantern, casts its gentle light upon the forest floor, illuminating the intricate tapestry of life that thrives in the shadows. Each leaf, each blade of grass, seems to shimmer with a silvered glow, as if touched by the hand of magic.", 1)
        'The moon, a luminous orb hanging in the sky like a celestial lantern, casts its gentle light upon the forest floor, illuminating the intricate tapestry of life that thrives in the shadows.'

        Args:
            user_input (str): A string representing the input text.
            summary_length (int): An integer representing the length of the summary.

        Returns:
            str: A string representing the summary of the input text.
        """
        
        # Define the prompt template
        prompt_template = f'''
        Given the input text:
        Input: {user_input}
        Produce a {summary_length} sentences length summary of the text.
        Captures the main ideas and key points of the text.
        Summary Does not include verbatim sentences from the original text.
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(summary_length, int):
            raise TypeError("summary_length must be of type int")

        try:
            # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt_template}],
            )
            response1 = response.choices[0].message.content
            return response1
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")


    ############## TEXT QUESTION ANSWERING ##############
    def text_qna(self, user_input: str, question: str) -> str:
        """
        answer the given question based on the input text.

        Example:
        >>> text_qna("OpenAI has hosted a hackathon for developers to build AI models. The event took place on 15th October 2022. The event was a huge success with over 1000 participants from around the world.", "When did the event happen?")
        '15th October 2022'

        Args:
            user_input (str): A string representing the input text.
            question (str): A string representing the question to be answered based on the input text.

        Returns:
            str: A string representing the answer to the question based on the input text.
        """
        # Define the prompt template
        prompt_template = f'''
        Given the input text:
        Input: {user_input}
        Answer the following question: 
        {question}
        The answer should be relevant and concise, without any additional information.
        Ensure that the answer directly addresses the question.
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(question, str):
            raise TypeError("question must be of type str")

        try:
            # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt_template}],
            )
            response1 = response.choices[0].message.content
            return response1
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")

    ############## TEXT INTENT DETECTION ##############
    def text_intent(self, user_input: str) -> list:
        """
        Identify the intent of the user input.

        Example:
        >>> text_intent("let's book a flight for our vacation and reserve a table at a restaurant for dinner. also going to watch football match at 8 pm.")
        ['book a flight', 'reserve a table', 'watch football match']

        Args:
            user_input (str): A string representing the input sentence.

        Returns:
            list: A list of strings representing the detected intents from the input text.
        """
            
        # Define the prompt template
        prompt_template = f'''
        Given the input sentence:
        user input: {user_input}
        Identify the intent of the text.
        If no clear intent can be determined from the input, return None.
        If the output intent contains multiple words, separate them with comma.
        output must be in this format -> Intent: intent1, intent2, ...
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        

        try:
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt_template}],
            )

            response1 = response.choices[0].message.content
            # make those intent in a list
            response1 = response1.split("Intent: ")[1].split(", ")
            return response1
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")

    ############## TEXT EMBEDDING ##############
    def text_embedd(self, user_input: str, embedd_model: str = "text-embedding-3-large") -> list:
        """
        Generate embeddings for the given input text.

        Example:
        >>> text_embedd("OpenAI has hosted a hackathon for developers to build AI models. The event took place on 15th October 2022. The event was a huge success with over 1000 participants from around the world.", "text-embedding-3-large")
        [-0.007, 0.003, 0.002, 0.001, 0.003, 0.002, 0.002, 0.001, 0.002, 0.002 ...]

        Args:
            user_input (str): A string representing the input text.
            embedd_model (str): A string representing the embedding model to use. If not provided, the default value is "text-embedding-3-large".

        Returns:
            list: A list representing the embeddings of the input text.
        """

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(embedd_model, str):
            raise TypeError("embedd_model must be of type str")
        
        try:
            response = self.client.embeddings.create(
            input=user_input,
            model=embedd_model,
            )
            return response.data[0].embedding
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")


    ############## TEXT SPAM ##############
    def detect_spam(self, user_input: str, num_classes: str = "spam, not_spam, unknown", explanation: bool = True) -> dict:
        """
        Perform spam detection on the input text.

        Example:
        >>> detect_spam("Congratulations! You have won a lottery of $1,000,000!", "harmed, not_harmed, unknown", explanation=False)
        {'prediction': 'harmed'}

        Args:
            user_input (str): A string representing the input text.
            num_classes (str): A string representing the number of classes for spam detection.
            explanation (bool): A boolean value representing whether to include explanation in the output.

        Returns:
            dict: A dictionary containing the prediction of the spam detection.
        """
        if explanation:
            format_answer = "prediction:your_answer\nexplanation:your_answer"
        else:
            format_answer = "prediction:your_answer\n"
        
        # Question to be asked
        prompt_template = f'''Given the input text, perform spam detection on it
        {user_input}
        num_classes: {num_classes}
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
            # # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt_template}],
            )

            response1 = response.choices[0].message.content

            result = {}
            for line in response1.split("\n"):
                if line.strip():
                    key, value = line.split(": ", 1)
                    result[key.strip()] = value.strip()

            return result
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")


    ############## TEXT SPELLCHECK ##############
    def text_spellcheck(self, user_input: str) -> str:
        """
        Correct the spelling of the input text.

        Example:
        >>> text_spellcheck("I am a good speler")
        'I am a good speller'

        Args:
            user_input (str): A string representing the input text.

        Returns:
            str: A string representing the corrected spelling of the input text.
        """
        
        prompt_template = f'''Given the input text:
        user input: {user_input}
        output must be in this format:
        misspelled_word:corrected_word
        ...
        output must not contain any other information than the format
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")

        try:
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt_template}],
            )

            response1 = response.choices[0].message.content

            response1 = response1.split("\n")
            result = [(line.split(":")[0], line.split(":")[1]) for line in response1 if line.strip()]

            # replace the misspelled words with the corrected words
            for word, corrected_word in result:
                user_input = user_input.replace(word, corrected_word)

            return user_input
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")


    ############## TEXT SEMANTIC ROLE LABELING ##############
    def text_srl(self, user_input: str) -> dict:
        """
        Perform Semantic Role Labeling (SRL) on the input text.

        Example:
        >>> text_srl("John ate an apple")
        {'Predicate': 'ate', 'Agent': 'John', 'Theme': 'apple'}

        Args:
            user_input (str): A string representing the input sentence.

        Returns:
            dict: A dictionary containing the Predicate, Agent, and Theme of the input sentence.
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
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt_template}],
            )

            response1 = response.choices[0].message.content

            result = {}
            for line in response1.split("\n"):
                if line.strip():
                    key, value = line.split(": ", 1)
                    result[key.strip()] = value.strip()

            # Remove "- " from the keys
            result = {key.replace("- ", ""): value for key, value in result.items()} 

            return result
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")


    ############## TEXT SENTIMENT ##############
    def text_sentiment(self, user_input: str, num_classes: str = "positive, negative, neutral", explanation: bool = True) -> dict:
        """
        Perform sentiment detection on the input text.

        Example:
        >>> text_sentiment("I love this product. It works great!", "positive, negative, neutral", explanation=False)
        {'prediction': 'positive'}

        Args:
            user_input (str): A string representing the input text.
            num_classes (str): A string representing the number of classes for sentiment detection.
            explanation (bool): A boolean value representing whether to include explanation in the output.

        Returns:
            dict: A dictionary containing the prediction of the sentiment detection.
        """
        if explanation:
            format_answer = "prediction:your_answer\nexplanation:your_answer"
        else:
            format_answer = "prediction:your_answer\n"
        
        # Question to be asked
        prompt_template = f'''Given the input text, perform sentiment detection on it
        {user_input}
        num_classes: {num_classes}
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
            # # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt_template}],
            )

            response1 = response.choices[0].message.content


            result = {}
            for line in response1.split("\n"):
                if line.strip():
                    key, value = line.split(": ", 1)
                    result[key.strip()] = value.strip()

            return result
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")

    ############## TEXT TOPIC ##############
    def text_topic(self, user_input: str, explanation: bool = True) -> dict:
        """
        Perform topic detection on the input text.

        Example:
        >>> text_topic("OpenAI has hosted a hackathon for developers to build AI models. The event took place on 15th October 2022. The event was a huge success with over 1000 participants from around the world.", explanation=False)
        {'topic': 'technology'}

        Args:
            user_input (str): A string representing the input text.
            explanation (bool): A boolean value representing whether to include explanation in the output.

        Returns:
            dict: A dictionary containing the detected topic from the input text.
        """
        if explanation:
            format_answer = "topic:your_answer\nexplanation:your_answer"
        else:
            format_answer = "topic:your_answer\n"
        
        # Question to be asked
        prompt_template = f'''Given the input text, perform topic detection on it
        {user_input}
        You must not provide any other information than the format
        {format_answer}
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(explanation, bool):
            raise TypeError("explanation must be of type bool")

        try:
            # Generate response using the provided model (assuming it's defined elsewhere)
            # # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt_template}],
            )

            response1 = response.choices[0].message.content

            result = {}
            for line in response1.split("\n"):
                if line.strip():
                    key, value = line.split(": ", 1)
                    result[key.strip()] = value.strip()

            return result
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")


    ############## POS DETECTION ##############
    def detect_pos(self, user_input: str, pos_tags: str = "") -> dict:
        """
        Perform Part-of-Speech (POS) detection on the input text.

        Example:
        >>> detect_pos("I love Lamborghini, but Bugatti is even better. Although, Mercedes is a class above all. and i work in Google", "noun, verb, adjective")
        {'noun': ['Lamborghini', 'Bugatti', 'Mercedes', 'Google'], 'verb': ['love', 'is', 'work'], 'adjective': ['better', 'above', 'all']}

        Args:
            user_input (str): A string representing the input sentence.
            pos_tags (str): A string of comma-separated POS tags to be detected from the input.

        Returns:
            dict: A dictionary containing the detected NER tags with their values.
        """

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(pos_tags, str):
            raise TypeError("ner_tags must be of type str")
        
        # check if parameters are not empty
        if not user_input:
            raise ValueError("user_input cannot be empty")
        
        # user ner tags
        if pos_tags != "":
            user_pos_tags = f'''POS TAGS: {pos_tags}'''
        else:
            user_pos_tags = f'''POS TAGS:noun, verb, adjective, adverb, pronoun, preposition, conjunction, interjection, determiner, cardinal, foreign, number, date, time, ordinal, money, percent, symbol, punctuation, emoticon, hashtag, email, url, mention, phone, ip, cashtag, entity,'''

        # Generate the prompt template
        prompt_template = f'''Given the input text:
        user input: {user_input}
        
        perform POS detection on it.
        {user_pos_tags}
        answer must be in the format
        tag:value
        '''

        try:
            # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt_template}],
            )
            response1 = response.choices[0].message.content

            # convert x to a dictionary and values must be a list
            x = response1.split('\n')
            x = [i.split(':') for i in x if i]
            x = {i[0].strip(): [j.strip() for j in i[1].split(',')] for i in x}
            return x

        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")


    ############## TEXT BADNESS ##############
    def text_badness(self, user_input: str) -> dict:
        """
        Detect the badness level of the input text.

        Example:
        >>> text_badness("I hate you")
        {'categories': Categories(harassment=False, harassment_threatening=False, ...),
        'category_scores': CategoryScores(harassment=0.034894540905952454, harassment_threatening=0.0005514638032764196, ...),
        'flagged': True}

        Args:
            user_input (str): A string representing the input text.

        Returns:
            dict: A dictionary containing the badness level of the input text.
        """

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        
        # check if parameters are not empty
        if not user_input:
            raise ValueError("user_input cannot be empty")

        try:

            # Generate response using the provided model (assuming it's defined elsewhere)
            response = self.client.moderations.create(input=user_input)
            output = response.results[0]
            return dict(output.categories)

        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")


    ############## EMOJIS TEXT ############## 
    def text_emojis(self, user_input: str, high_cost: bool = False) -> str:
        """
        Detect the badness level of the input text.

        Example:
        >>> text_emojis("I love 🍕 and 🍔", high_cost=False)
        'I love pizza and burger'

        Args:
            user_input (str): A string representing the input text.
            high_cost (bool): A boolean value representing whether to use the high-cost model.

        Returns:
            str: A string representing the text with emojis replaced by their text representation.
        """

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(high_cost, bool):
            raise TypeError("high_cost must be of type bool")
        
        # check if parameters are not empty
        if not user_input:
            raise ValueError("user_input cannot be empty")
        

        try:
            if high_cost:
                prompt_template = f'''Given the input text:
                user input: {user_input}
                Identify the emojis in the text and replace them with their text representation.
                output must be the updated text with emojis replaced by their text representation.
                output must not contain any other information than the updated text.
                '''
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt_template}],
                    )

                response1 = response.choices[0].message.content
                return response1

            else:
                x = []
                y = ''
                for c in user_input.split():
                    if c in emoji.EMOJI_DATA:
                        # get c and its index
                        x.append((user_input.index(c)))
                        y += c+'\n'

                if x == []:
                    return "No emojis found in the text."
                prompt_template = f'''Given the below emojis
                {y}
                Replace each emoji with its text representation in the same order
                output format must be like this:
                text respresentation
                ...
                output must mot contain any other information than the format
                '''

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt_template}],
                    )

                response1 = response.choices[0].message.content

                ya = response1.split('\n')

                # Convert the text string into a list of characters
                text_list = list(user_input)

                # Replace emojis at positions specified in x with corresponding items from ya
                for i in range(len(x)):
                    if x[i] < len(text_list):
                        text_list[x[i]] = ya[i]

                # Join the list of characters back into a string
                modified_text = ''.join(text_list)

                return modified_text

        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")


   ############## TEXT IDIOMS ##############
    def text_idioms(self, user_input: str) -> list:
        """
        Identify and extract any idioms present in the given sentence.

        Example:
        >>> text_idioms("I am over the moon")
        ['over the moon']

        Args:
            user_input (str): A string representing the input sentence.

        Returns:
            list: A list of strings containing the extracted idioms from the input sentence.
        """

        prompt_template = f'''Given the input sentence:
        user input: {user_input}

        __________________________

        Identify and extract any idioms present in the sentence.
        Output must only contain the extracted idioms.
        Output must not contain any bullet points.
        If there is more than one idiom found, return both in new lines.
        If no idiom is found, return None.

        output must be in this format:
        extracted idioms
        ...
        output must not contain any other information than the extracted idioms.
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")

        try:
            # Generate response using the provided model (assuming it's defined elsewhere)            
            response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt_template}],
                    )
            response1 = response.choices[0].message.content

            return response1.split("\n")
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")


    ############## TEXT ANOMALY ##############
    def text_anomaly(self, user_input: str) -> list:
        """
        Detect any anomalies or outliers in the given input text.

        Example:
        >>> text_anomaly("man is walking on the road while eating a book")
        ['eating a book']

        Args:
            user_input (str): A string representing the input sentence.

        Returns:
            list: A list of strings containing the detected anomalies or outliers in the input text.
        """

        prompt_template = f'''Given the input sentence:
        user input: {user_input}
        Detect any anomalies or outliers in the text.
        output only the detected anomalies and dont provide any other information
        dont use bullet points or any other formatting

        output must be in this format:
        detected anomalies
        ...
        output must not contain any other information than the detected anomalies.
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        
        try:
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt_template}],
            )

            response1 = response.choices[0].message.content

            # break it into list
            information = response1.split("\n")

            return information
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")


    ############## TEXT COREFERENCE ##############
    def text_coreference(self, user_input: str) -> dict:
        """
        Perform coreference resolution on the input text to identify the resolved references for each pronoun.

        Example:
        >>> text_coreference("John loves his dog. He takes care of it.")
        {'his': 'John', 'it': 'dog'}

        Args:
            user_input (str): A string representing the input paragraph.

        Returns:
            dict: A dictionary containing the resolved references for each pronoun in the input text.
        """
        
        prompt_template = f'''Given the input paragraph:
        user input: {user_input}

        __________________________

        Perform coreference resolution to identify what each pronoun in the paragraph is referring to. 
        Output must only contain the resolved references for each pronoun, without any additional context.
        Output must not contain any bullet points.
        If no referent is found for a pronoun, return 'None'.

        Output must be in this format:
        Pronoun: Referent
        ...
        outpust must not contain anyother information.
        '''

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")

        try:
                
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt_template}],
            )

            response1 = response.choices[0].message.content

            # Extract resolved coreferences
            coreferences = response1.split("\n")
            coreferences = [coreference for coreference in coreferences if coreference]
            coreferences_dict = {item.split(': ')[0]: item.split(': ')[1] for item in coreferences}

            return coreferences_dict
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")


   ############## PDF CHAT ##############
    def chat_pdf(self, user_input: str, pdf_path: str, pages: str = "all") -> str:
        """
        Answer the given question based on the information extracted from the provided PDF file.

        Example:
        >>> chat_pdf("Who is the CEO of Apple Inc.?", "apple.pdf", "1-2")
        'Tim Cook'

        Args:
            user_input (str): A string representing the question to be answered based on the information extracted from the PDF file.
            pdf_path (str): A string representing the path to the PDF file.
            pages (str): A string representing the range of pages to extract information from. If set to "all", extract information from all pages. format should be "integer-integer".
        
        Returns:
            str: A string representing the answer to the question based on the information extracted from the PDF file.
        """

        # check if parameters are of correct type
        if not isinstance(user_input, str):
            raise TypeError("user_input must be of type str")
        if not isinstance(pdf_path, str):
            raise TypeError("pdf_path must be of type str")
        if not isinstance(pages, str):
            raise TypeError("pages must be of type str")
        
        # check if parameters are not empty
        if not user_input:
            raise ValueError("user_input cannot be empty")
        if not pdf_path:
            raise ValueError("pdf_path cannot be empty")
        
        
        if pages != "all":
            docs = []
            # pages string must patch this format integer-integer
            if "-" not in pages:
                raise ValueError("Please provide pages in the format 'integer-integer'")
            pages = pages.split("-")
            pdf_text = ''
            doc = fitz.open(pdf_path)
            for page_num in range(int(pages[0]), int(pages[1])):
                doc = doc.load_page(page_num)
                pdf_text += doc.get_text()
        else:
            pdf_text = ''
            doc = fitz.open(pdf_path)
            for page in doc:
                pdf_text += page.get_text()


        
        prompt_template = f'''Given the input paragraph:
        Information: {pdf_text}

        __________________________

        Answer the below question based on the information provided. if information is not enough to answer the question, return No information found.
        {user_input}
        '''

        try:
                
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt_template}],
            )

            response1 = response.choices[0].message.content

            return response1
        except Exception as e:
            raise ValueError("Please provide a correct PdF path/API Key or try again.")

    ############## TEXT OCR ##############
    def text_ocr(self, image_path: str, user_input: str) -> str:
        """
        Extract text from an image using Optical Character Recognition (OCR).

        Example:
        >>> text_ocr("image.jpg", "Extract text from the image.")
        'The extracted text from the image.'

        Args:
            image_path (str): A string representing the path to the image file.
            user_input (str): A string representing the prompt to extract text from the image.

        Returns:
            str: A string representing the extracted text from the image.
        """

        prompt_template = f'''{user_input}
        if no text found returns None
        '''

        # check if parameters are of correct type
        if not isinstance(image_path, str):
            raise TypeError("image_path must be of type str")
        if not isinstance(user_input, str):
            raise TypeError("prompt must be of type str")
        
        # check if parameters are not empty
        if not image_path:
            raise ValueError("image_path cannot be empty")
        if not user_input:
            raise ValueError("prompt cannot be empty")

        try:
            import base64
            # Function to encode the image
            def encode_image(image_path):
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')

            # Getting the base64 string
            try:
                base64_image = encode_image(image_path)
            except FileNotFoundError:
                raise FileNotFoundError("The image file was not found.")

            response = self.client.chat.completions.create(
            model=self.vision_model_name,
            messages=[
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_template},
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                    },
                ],
                }
            ]
            )

            # Return the extracted text
            return response.choices[0].message.content
        except Exception as e:
            raise ValueError("Please provide a correct API key or try again.")
        