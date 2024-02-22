# openAI chatbot API code with local embedding vectorstore
# By: Reuben
# Assisted by : Yap Chun Wei
# with inputs from : DSOR team

from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.vectorstores import Chroma

# purchase openAI account to obtain personal account api key, input to use openAI models
OPENAI_API_KEY = ""

# directory to the chroma database folder
persist_directory = "chroma_db"

# load embedding model to translate words into meaningful number vectors for LLM
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=OPENAI_API_KEY,
)

# loading previously stored vector database into memory
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# defining retriever needed for the llm model, specifying retrieval from vector database
retriever = vectordb.as_retriever()

# setting up the large language model parameters
llm = ChatOpenAI(
    model="gpt-4",
    # model="gpt-3.5-turbo",
    openai_api_key=OPENAI_API_KEY,
    streaming=True,  # toggle True for words appearing one by one, False for all at once
    temperature=0,  # between 0 and 1 , 0 being completely deterministic model without probabalistic fluctuations
)

# initialising memory unit for back and forth conversation with chatbot
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# loading in system prompt instructions from text file
prompt = open("C:/Users/reu34/Desktop/GAIRY_ver4/prompt1.txt", mode="r").read()

# appending prompt with placeholder for context matches from vector database
system_template = prompt + """\n\n{context}"""

# initializing user template which will contain the question in future
user_template = "Question: {question}"

# compiling system and user templates into final prompt to be fed to LLM
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(user_template),
]
qa_prompt = ChatPromptTemplate.from_messages(messages)

# setting up conversation chain function to talk to chatbot, using all the previous initialized components
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": qa_prompt},
)


##########################################################
# Using the LLM chatbot
########################################################


# input query, generate result and display result answer
query = "my skin is red and itchy, is it eczema?"
result = conversation_chain({"question": query})

result["answer"]

# input next question....
query = "what are the treatments"
result = conversation_chain({"question": query})

result["answer"]


# clear and reset to erase LLM memory
