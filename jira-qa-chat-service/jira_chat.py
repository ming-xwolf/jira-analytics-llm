import shutil
import os

from langchain_community.vectorstores import DeepLake
from langchain_community.llms import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import CSVLoader
from langchain.docstore.document import Document


base_url="http://192.168.2.52:11434"
model_name = "llama3"
ollama_embeddings = OllamaEmbeddings(base_url=base_url,model=model_name)
ollama_llm = Ollama(base_url=base_url, model=model_name)

def process_csv_files(file_paths) -> list[Document]:
    documents = []
    for file_path in file_paths:
        loader = CSVLoader(file_path)
        loaded_docs = loader.load()
        documents.extend(loaded_docs)
    return documents


def get_vectorstore(documents, embeddings, dataset_path) -> DeepLake:

    if documents:
        vectorstore = DeepLake.from_documents(documents, embeddings, dataset_path=dataset_path)
    else:
        vectorstore = DeepLake(embedding_function=embeddings, dataset_path=dataset_path)
    return vectorstore


def check_data_updated(documents, dataset_path, embeddings) -> bool:
    # Get the number of rows in the dataframe
    num_rows_doc = len(documents)
    if os.path.isdir(dataset_path):
        # Get the number of documents in the DeepLake dataset
        try:
            check_vectorstore = DeepLake(embedding_function=embeddings, dataset_path=dataset_path)
            num_docs_deep_lake = check_vectorstore.ds.num_documents
            if num_docs_deep_lake == 0:
                shutil.rmtree(dataset_path)
        except Exception as e:
            print(f"Error processing vectorstore: {e}")
            num_docs_deep_lake = 0
        print(
            f"Number of rows in the file: {num_rows_doc} Number of documents in the DeepLake dataset: {num_docs_deep_lake}")
        return num_rows_doc != num_docs_deep_lake
    else:
        return True


def update_vectorstore():
    global vectorstore
    dataset_path = './db/deeplake'
    data_csv_file_path = './sample_data/issues.csv'

    documents = process_csv_files([data_csv_file_path])
    data_updated = check_data_updated(documents, dataset_path, ollama_embeddings)


    # Compare the number of rows in the dataframe with the number of documents in the DeepLake dataset
    if data_updated:
        # Execute your code here
        try:
            if os.path.isdir(dataset_path):
                shutil.rmtree(dataset_path)
            documents = documents
        except Exception as e:
            print(f"Error processing dataframe: {e}")
            documents = None
    else:
        documents = None

    vectorstore = get_vectorstore(documents, ollama_embeddings, dataset_path)


def answer_question(query):
    update_vectorstore()

    retriever = vectorstore.as_retriever()
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["k"] = 10

    qa = ConversationalRetrievalChain.from_llm(ollama_llm, retriever=retriever,
                                               return_source_documents=True)

    # Answer query
    chat_history = []
    query = "You are Quality Assurance Manager - provider of insights on current qa process " \
            "You should always behave as QA Manager, " \
            "neither not a chatbot, nor an AI language model. " \
            "Examine the issues context and provide an answer to a question: " + query
    result = qa({"question": query, "chat_history": chat_history})

    return result
