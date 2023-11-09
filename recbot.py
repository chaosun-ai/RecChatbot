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

#classify_type = Literal["fashion", "cloth searching", "general"]
classifier_template_str = "Please classify user question {input} as either 'fashion', 'cloth searching' or 'other' \
    and reply either 'fashion', 'cloth searching' or 'other' according to your classification"


classifier_prompt = PromptTemplate(
    template=classifier_template_str,
    input_variables=['input'],

)


# Prompt templates
fashion_template = PromptTemplate(
    input_variables = ['input'], 
    template = """As a very knowledgeable woman cloth fashion expert, \
    please provides accurate and eloquent answers to fashion-related questions {input}."""
)

recommendation_template = PromptTemplate(
    input_variables = ['input'], 
    template='''Recommend me a woman cloth based on the user requirement {input}
    '''
)

llm = AzureChatOpenAI(
    temperature=0,
    deployment_name="azure-gpt-35-turbo",
    model_name="azure-gpt-35-turbo"
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
fashion_chain = LLMChain(llm=llm, prompt=fashion_template, verbose=True, output_key='fashion_suggestion')
recommendation_chain = LLMChain(llm=llm, prompt=recommendation_template, verbose=True, output_key='cloth_recommendation')

"""
sequential_chain = SequentialChain(
    chains = [fashion_chain, recommendation_chain], 
    input_variables=['input'], 
    output_variables=['programming_language', 'book_name'],
    verbose=True)
"""



#prompt_reply_queue = deque(maxlen=10)


if prompt: 
    class_reply = classifier_chain({'input': prompt})
    if 'fashion' in class_reply['classification_result'].lower():
        reply = fashion_chain({'input': prompt})['fashion_suggestion'] #
    elif 'cloth' in class_reply['classification_result'].lower():
        reply = recommendation_chain({'input': prompt})['cloth_recommendation'] # to do - add memory.
    else:
        reply = default_chain


    st.text_area("Answer", value=reply)

    with open('chat_history.log', 'a') as f:
        f.write(prompt + '\n' + reply + '/n')
        
    with st.expander("Chat History"):
        st.info('Chat history in chat_history.log')

    with st.expander("Print for debugging"):
        st.info(class_reply)