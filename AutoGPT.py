from dotenv import load_dotenv
#from langchain.llms import OpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.output_parsers.openai_functions import PydanticAttrOutputFunctionsParser
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable import RunnableBranch
from typing import Literal

from collections import deque

from utility import TopicClassifier
import streamlit as st

# Load env vars
load_dotenv()

st.title('Woman Cloth Helper Wizard')
prompt = st.text_input('Tell me what woman cloth would you like me to recommend.') 



# Define the prompt
fashion_template = """You're a very knowledgeable woman cloth fashion expert \
    who provides accurate and eloquent answers to fashion-related questions and recommend cloth for customers. \
    When you don't know the answer to a question you admit that you don't know.
    
    Here is a question:
    {input}"""
fashion_prompt = PromptTemplate.from_template(fashion_template)

#classifier_template =     """You're a very knowledgeable woman cloth fashion expert \
#    who provides accurate and eloquent answers to fashion-related questions and recommend cloth for customers. \
#    if the question is related to your expertise, then reply 'related'. Otherwise, reply 'not_related'."""

classify_type = Literal["fashion", "cloth searching", "general"]
classifier_template_str = "Please classify user question {input} and reply either 'fashion', 'cloth searching' or 'other' "


classifier_prompt = PromptTemplate(
    template=classifier_template_str,
    input_variables=['input'],
    
)


# Prompt templates
language_template = PromptTemplate(
    input_variables = ['topic'], 
    template='Suggest me a programming language for {topic} and respond in a code block with the language name only'
)

book_recommendation_template = PromptTemplate(
    input_variables = ['programming_language'], 
    template='''Recommend me a book based on this programming language {programming_language}

    The book name should be in a code block and the book name should be the only text in the code block
    '''
)

llm = AzureChatOpenAI(
    temperature=0,
    deployment_name="azure-gpt-35-turbo",
    model_name="azure-text-embedding-ada-002"
)

# Define classifier chains
classifier_function = convert_pydantic_to_openai_function(TopicClassifier)
llm_classifier = llm.bind(
    functions=[classifier_function], function_call={"name": "TopicClassifier"}
)
parser = PydanticAttrOutputFunctionsParser(
    pydantic_schema=TopicClassifier, attr_name="topic"
)
classifier_chain = llm_classifier | parser

# default chain
default_chain = "I can only answer questions about fashion."

classifier_chain = LLMChain(llm=llm, prompt=classifier_prompt, verbose=True, output_key='classification_result')
language_chain = LLMChain(llm=llm, prompt=language_template, verbose=True, output_key='programming_language')
book_recommendation_chain = LLMChain(llm=llm, prompt=book_recommendation_template, verbose=True, output_key='book_name')

sequential_chain = SequentialChain(
    chains = [language_chain, book_recommendation_chain], 
    input_variables=['topic'], 
    output_variables=['programming_language', 'book_name'],
    verbose=True)

prompt_reply_queue = deque(maxlen=10)


if prompt: 
    reply = classifier_chain({'input': prompt})
    if 



    prompt_reply_queue.append(prompt)
    prompt_reply_queue.append(reply['classification_result'])

    st.text_area("Answer", value=reply['classification_result'])
        
    strings = "\n".join(prompt_reply_queue)
    with st.expander("Chat History"):
        st.write(strings)

    #with st.expander("Recommended Book"):
    #    st.info(reply['book_name'])
