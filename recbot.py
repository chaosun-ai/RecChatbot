import pandas as pd
import ast
from dotenv import load_dotenv

from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain.output_parsers.openai_functions import PydanticAttrOutputFunctionsParser
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable import RunnableBranch
from typing import Literal

from collections import deque

from utility import get_embeddings, sort_df_similarity
import streamlit as st

# Load env vars
load_dotenv()

df = pd.read_csv('articles_embedding.csv')
df['embeddings'] = df['embeddings'].apply(ast.literal_eval)

st.title('Woman Cloth Helper Wizard')
prompt = st.text_input('Tell me what woman cloth would you like me to recommend.') 


classifier_template_str = "Please classify user question {input} as either 'fashion', 'cloth searching' or 'other' \
    and reply either 'fashion', 'cloth searching' or 'other' according to your classification"


classifier_prompt = PromptTemplate(
    template=classifier_template_str,
    input_variables=['input'],

)


# Prompt templates
fashion_template = PromptTemplate(
    input_variables = ['input'], 
    template = """{chat_history}
    As a very knowledgeable woman cloth fashion expert, \
    please provides accurate and eloquent answers to fashion-related questions {input}."""
)

recommendation_template = PromptTemplate(
    input_variables = ['input'], 
    template="""{chat_history}
    Recommend me a woman cloth based on the user requirement {input}"""
)

if "llm" not in st.session_state:
    st.session_state.llm = AzureChatOpenAI(
        temperature=0,
        deployment_name="azure-gpt-35-turbo",
        model_name="azure-gpt-35-turbo"
    )

if "memory" not in st.session_state:
    st.session_state.memory = ConversationSummaryMemory(llm=st.session_state.llm, memory_key="chat_history")

# default chain
default_chain = "I can only answer questions about fashion and cloth recommendation."

classifier_chain = LLMChain(llm=st.session_state.llm, prompt=classifier_prompt, verbose=True, output_key='classification_result')
fashion_chain = LLMChain(llm=st.session_state.llm, prompt=fashion_template, verbose=True, output_key='fashion_suggestion', memory=st.session_state.memory)
recommendation_chain = LLMChain(llm=st.session_state.llm, prompt=recommendation_template, verbose=True, output_key='cloth_recommendation', memory=st.session_state.memory)



#prompt_reply_queue = deque(maxlen=10)


if prompt: 
    class_reply = classifier_chain({'input': prompt + st.session_state.memory.buffer})
    if 'fashion' in class_reply['classification_result'].lower():
        reply = fashion_chain({'input': prompt})['fashion_suggestion'] #
        reply_embedding = get_embeddings(reply)
        df_sort = sort_df_similarity(df, reply_embedding, 'embeddings')
        cloth_predict = df_sort[['article_id', 'prod_name', 'detail_desc']].head(2).to_string()
        final_reply = reply + '\n' \
            + 'Here I got some clothes for you. If these are not what you need, please let me know more details about your requirement.'\
            + '\n' + cloth_predict + '\n'

    elif 'cloth' in class_reply['classification_result'].lower():
        reply = recommendation_chain({'input': prompt})['cloth_recommendation'] # 
        reply_embedding = get_embeddings(reply)
        df_sort = sort_df_similarity(df, reply_embedding, 'embeddings')
        cloth_predict = df_sort[['article_id', 'prod_name', 'detail_desc']].head(2).to_string()
        final_reply = reply + '\n' \
            + 'Here I got some clothes for you. If these are not what you need, please let me know more details about your requirement.'\
            + '\n' + cloth_predict + '\n'
    else:
        final_reply = default_chain


    st.text_area("Answer", value=final_reply)

    with open('chat_history.log', 'a') as f:
        f.write(prompt + '\n' + final_reply + '\n')
        
    with st.expander("Chat History"):
        st.info('Chat history in chat_history.log')

    with st.expander("Print for debugging"):
        st.info(st.session_state.memory)