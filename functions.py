from langchain_community.embeddings import VertexAIEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyPDFLoader
from langdetect import detect
import time, hashlib
import streamlit as st
from datetime import datetime
from utils.chains import *

def verify_quiz_qn(question):
    """
    Function to verify if the format of the question is correct

    Args:
        question (dict): Question with answer, options and explanation in dict

    Returns:
        boolean: True if correct else False
    """
    checklist = ["options", "question", "correct_answer", "explanation"]
    for check in checklist:
        if check not in question.keys():
            return False
        if check == "options":
            if len(question['options'].keys()) + len(question['options'].values()) != 8:
                return False
    return True


def delete_invalid_question(quiz):
    """
        Function to clean up invalid questions in quiz

        Args:
            question (dict): Quiz with questions

        Returns:
            quiz (dict): resulting quiz after deleting invalid questions
    """
    i=0
    while i<len(quiz):
        is_valid = verify_quiz_qn(quiz[i])
        if not is_valid:
            quiz.pop(i)
        else:
            i+=1
    return quiz


def compute_file_hash(file):
    """
    Function to compute SHA256 hash of a file

    Args:
        file (UploadedFile): Uploaded file from the user

    Returns:
        file_hash: SHA256 hash of the file
    """
    sha256_hash = hashlib.sha256()
    for byte_block in iter(lambda: file.read(4096), b""):
        sha256_hash.update(byte_block)
    file.seek(0)
    return sha256_hash.hexdigest()


def check_embeddings(file_hash, index_name):
    """
    Check if embeddings exist in the namespace of Pinecone Vectore DB
    
    Args:
        file_hash (str): SHA256 hash of the file
        index_name (str): Pinecone index name
    
    Returns:
        boolean
    """
    namespaces = PineconeVectorStore.get_pinecone_index(index_name).describe_index_stats()["namespaces"]
    if file_hash in namespaces:
        return True
    return False


def update_timestamp(file_hash, index_name):
    """
    Update timestamp of the embeddings in the namespace specified
    
    Args:
        file_hash (str): SHA256 hash of the file
        index_name (str): Pinecone index name
    """
    index = PineconeVectorStore.get_pinecone_index(index_name)
    num_dims = index.describe_index_stats()["dimension"]
    namespace_matches = index.query(namespace=file_hash, vector=[0] * num_dims, top_k=5, include_metadata=True)['matches']
    update_metadata = {}
    for match in namespace_matches:
        match["metadata"]["timestamp"] = time.time() 
        update_metadata[match["id"]] = match["metadata"]
    for ids,metadata in update_metadata.items():
        index.update(id=ids, namespace=file_hash, set_metadata=metadata)
    

def process_file(index_name, file_path, file_hash, to_reindex=False):
    """
    Process uploaded file and create FAISS index.
    
    Args:
        index_name (str): Pinecone index name
        file_path (str): Path to the uploaded file.
        file_hash (str): SHA256 hash of the file
        to_reindex (boolean): whether to force reindexing of the file
    
    Returns:
        PineconeVectorStore: A PineconeVectorStore index created from the processed file.
    """
    if file_path.split(".")[-1].lower() == "pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = UnstructuredWordDocumentLoader(file_path)
    documents = loader.load()

    docs_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = docs_splitter.split_documents(documents)
    if detect(docs[0].page_content) != "en":
        return None

    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")

    # store embeddings in vector store
    pages_chunk = 50
    texts = [doc.page_content for doc in docs]
    skip = False
    if check_embeddings(file_hash, index_name) and not to_reindex:
        st.info("This file has been indexed previously. Retrieved embeddings from previous indexing. Updating timestamp for last accessed")
        skip = True
        update_timestamp(file_hash, index_name)
        return PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace=file_hash), docs, skip
    
    st.info("First upload of this file. Indexing is required.")
    bar = st.progress(0, text="Indexing In Progress")
    pinecone_index = PineconeVectorStore.from_texts(texts[:50], embeddings, index_name=index_name, namespace=file_hash, metadatas=[{'timestamp': time.time()} for i in range(len(texts[:50]))])
    
    if len(texts)>50:
        bar.progress(pages_chunk/len(texts), text=f"Indexing In Progress  \t Completion: {pages_chunk}/{len(texts)}")
        for i in range(pages_chunk, len(texts) ,pages_chunk):
            pinecone_index = PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace=file_hash)
            bar.progress(i/len(texts), text=f"Waiting 60 seconds for quota  \t Completion: {i}/{len(texts)}")
            time.sleep(60)
            bar.progress(i/len(texts), text=f"Indexing In Progress  \t Completion: {i}/{len(texts)}")
            selected_texts = texts[i:i+pages_chunk]
            pinecone_index.add_texts(selected_texts, namespace=file_hash,  metadatas=[{'timestamp': time.time()} for i in range(len(selected_texts))])
            
    bar.empty()     
    return pinecone_index, docs, skip


def generate_topics(docs):
    """
    Generate topics from the FAISS index.
    
    Args:
        index: The FAISS index containing the document embeddings.
    
    Returns:
        list: A list of extracted topics.
    """
    with st.status(f"Generate topics from the file", expanded=True) as status:
        st.write("Retrieving Information from Index...")
        time.sleep(2)
        combined_text = format_docs(docs)
            
                # Extract topics
        st.write("LLM Generating Topics...")
        topic_chain = create_topic_extraction_chain()
        topics = topic_chain.invoke(combined_text)
        topics = topics.strip().split(", ")
        status.update(label=f"Topics are generated. Options for Quiz/Summary Generation are ready", state="complete", expanded=False)
    return topics


def generate_quiz(index, num_questions, difficulty, topic, docs):
    """
    Generate quiz questions.
    
    Args:
        index: The Pinecone index containing the document embeddings.
        num_questions (int): Number of questions to generate.
        difficulty (str): Difficulty level of the questions.
        topic (str): The topic for which to generate questions.
    
    Returns:
        list: A list of generated quiz questions.
    """
    quiz_chain = create_quiz_generation_chain(index, topic, docs)
    quiz = quiz_chain.invoke({
                "topics": topic,
                "num_questions": f"{num_questions}",
                "difficulty": difficulty
            })
    return quiz['questions']


def generate_summary(index, topic, docs, conciseness, provide_layman):
    """
    Generate summary.
    
    Args:
        index: The Pinecone index containing the document embeddings.
        topic (str): The topic for which to generate a summary.
        docs (list[Document]): Full list of LangChain documents.
    
    Returns:
        str: The generated summary.
    """
    layman_prompt = {True: "Provide additional layman explanation.", False: ""}
    conciseness_prompt = {"Brief": "brief definitions with core meaning", 
                          "Detailed": "detailed definitions includes the core meaning along with some additional context and explanation",
                          "Verbose": "verbose definitions includes extensive information, background, and mechanisms."}
    summary_chain = create_summary_generation_chain(index, topic, docs)
    summary = summary_chain.invoke({
                "conciseness": conciseness_prompt[conciseness],
                "layman": layman_prompt[provide_layman],
                "topic": topic
        })
    return summary