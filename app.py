# chatting with Chatbot llm but with the GUI interface enabled by streamlit

import streamlit as st
from dotenv import load_dotenv

from langchain.embeddings import OpenAIEmbeddings

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.prompts.chat import ChatPromptTemplate

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.vectorstores import Chroma

#  api_key for openai models
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]


# function to load in vector database from local address
def get_vectorstore():
    persist_directory = "https://github.com/Reu3406/chat-with-pdfs---stored-vectorDB/tree/main/chroma_db"
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY
    )
    vectordb = Chroma(
        persist_directory=persist_directory, embedding_function=embeddings
    )
    return vectordb


# function to handle question input into the GUI textbox
def handle_userinput(user_question):
    # passing the input question to the streamlit session memory, will be referenced later by LLM
    response = st.session_state.conversation({"question": user_question})
    # getting the LLM response from streamlit session chat history
    st.session_state.chat_history = response["chat_history"]

    # chat history in streamlit session memory is ordered : user question, answer, user question, answer
    # so by enumerating and replacing the odd number entries ,which are user questions, into the user message of html template script
    # similarly replacing even number entries, bot replies, into the bot template html template
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )


# reading in system prompt instructions from text file
#prompt = open("https://github.com/Reu3406/chat-with-pdfs---stored-vectorDB/blob/main/prompt1.txt", mode="r").read()
prompt='''You are assuming the role of a clinical assistant. You are provided with an educational material. Your task is to only use the material provided to answer questions. if no answer is found within the material, reply with sorry i'm unable to answer your question, please check with your doctor who will better advise you

answer question only relying on the material provided. you can recommend likely diagnosis and treatments but always refer the patient to check with their doctor. do not give a definitive diagnosis or treatment. 
'''

# main function to converse with llm
def get_conversation_chain(vectorstore):
    # specifying LLM model and parameters
    llm = ChatOpenAI(
        model="gpt-4", streaming=True, temperature=0, openai_api_key=OPENAI_API_KEY
    )
    # injecting user question input from streamlit session state converstation memory
    user_template = "Question: {question}"

    # compiling system prompt intruction from text file and the placeholder for matching context results from the vector database
    system_template = prompt + """\n\n{context}"""

    # compiling user question and system prompt instruction and matching contex results from vector database
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(user_template),
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    # conversation memory for LLM model
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # putting it all together into the function to talk to LLM
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
    )
    return conversation_chain


# assembling all the streamlit elements and the functions above into the main function which will run once the streamlit GUI window is opened
def main():

    # setting up title for the webpage
    st.set_page_config(page_title="AI clinical information assistant")
    st.write(css, unsafe_allow_html=True)

    # initialising streamlit session state memory items
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # main header line
    st.header(
        "Hi! i'm G-AI-RY, your document assistant. Press the start button at the side and ask me about the documents you converted into the vectorDB"
    )
    # creating input text field for question as well as the header for it
    user_question = st.text_input("what would you like to know about your documents?")

    # what happens when question is entered
    if user_question:
        handle_userinput(user_question)

    # the sidebar with the  start button to initialise everything
    with st.sidebar:

        if st.button("Start Engine"):
            with st.spinner("Processing"):

                vectorstore = get_vectorstore()
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()
