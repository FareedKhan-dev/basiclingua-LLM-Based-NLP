from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

VERSION = '1.0.0'
DESCRIPTION = 'A Python library based on Google Gemini LLM to perform basic and advanced natural language processing (NLP) tasks'
LONG_DESCRIPTION = 'Basiclingua is a Gemini LLM based Python library that provides functionalities for linguistic tasks such as tokenization, stemming, lemmatization, and many others.'

# Setting up
setup(
    name="basiclingua",
    version=VERSION,
    author="Fareed Hassan Khan, Syed Asad Rizvi",
    author_email="<fareedhassankhan12@gmail.com>, <syedasad44@gmail.com>",
    description=DESCRIPTION,
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=["google-generativeai", "grpcio", "grpcio-tools"],  # Add any dependencies here
    keywords=['python', 'NLP', 'Natural Language Processing', 'Linguistics', 'Gemini LLM', 'Google Gemini LLM'],
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Microsoft :: Windows",
        "License :: OSI Approved :: MIT License",
    ]
)