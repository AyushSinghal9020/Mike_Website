import pandas as pd
import streamlit as st 
# from tqdm.notebook import tqdm
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def process(value) :

    value = str(value)
    value = value.replace(' ' , '')

    return value

def process_csv(csv_file_path) :

    temp = []
    data = pd.read_csv(csv_file_path)

    for column in data.columns : data[column] = data[column].apply(process)

    data.to_csv(csv_file_path)

    data = open(csv_file_path).read().split('\n')
    columns = data[0]
    data = data[1 :]

    for row in data :

        tem = []

        for value , column in zip(row.split(',') , columns.split(',')) :

            tem.append(f'{column} is {value} \n')

        temp.append(' '.join(tem))

    return temp

def create_vectorstore(chunks , batch_size , embeddings , save = False , vector_store_name = 'vectorstore') :

    vectorstore = None
    batches = [
        chunks[index : index + batch_size]
        for index
        in range(0 , len(chunks) , batch_size)
    ]

    # pbar = tqdm(total = len(batches) , desc = 'Ingesting documents')

    for batch in batches :

        if vectorstore : vectorstore.add_texts(texts = batch)
        else : vectorstore = FAISS.from_texts(texts = batch , embedding = embeddings)

        # pbar.update(1)

    if save : vectorstore.save_local(vector_store_name)

    return vectorstore

def ask_llm(question , vectorstore) :

    chat = ChatCohere(cohere_api_key = 'vJZr4T4bWAJMn0kOdkSN1pmjxzrqLlPOy1YaA3fa')

    similar_docs = vectorstore.similarity_search(question)
    context = '\n'.join([doc.page_content for doc in similar_docs])

    prompt = '''
    You are a Tabular data Specialist

    - You will be provided with data in concatenated format of (column is value)
    - Your task is to answer the user queries based on the data provided.

    Context : {}

    Query : {}
    '''

    prompt = prompt.format(context , question)

    messages = [
        HumanMessage(
            content = prompt
        )
    ]

    response = chat.invoke(messages)

    return response

files = st.file_uploader(
    'Upload your CSV' , 
    type = 'csv' , 
    accept_multiple_files = True
)

question = st.text_input('Ask your question')




if st.button('Ask') : 

    data = []
    batch_size = 16
    embeddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')

    for file in files : 
        
        with open('file.csv' , 'wb') as ufile : ufile.write(file.getbuffer())
        
        data.extend(process_csv('file.csv'))

    vectorstore = create_vectorstore(
        chunks = data , 
        batch_size = batch_size , 
        embeddings = embeddings , 
        save = True
    )

    response = ask_llm(question , vectorstore)
    st.write(response.content)
