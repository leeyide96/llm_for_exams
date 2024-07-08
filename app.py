import streamlit as st
from langchain_community.embeddings import VertexAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain.pydantic_v1 import BaseModel, Field
from typing import List, Dict
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
import os,time, uuid
import vertexai
import tempfile

PROJECT_ID = "CHOOSE YOUR PROJECT"  # @param {type:"string"}
REGION = "us-central1"  # @param {type:"string"}

# Initialize Vertex AI SDK
vertexai.init(project=PROJECT_ID, location=REGION)

@st.cache_resource
def create_index_dict():
    """
    Create a cached resource for storing FAISS indexes.
    
    Returns:
        dict: An empty dictionary to store FAISS indexes.
    """
    return {}
faiss_indexes = create_index_dict()

class QuizQuestion(BaseModel):
    """
    Pydantic model for structuring a single quiz question.
    """
    question: str = Field(description="The text of the question")
    options: Dict[str, str] = Field(description="A dictionary of options, with keys A, B, C, D")
    correct_answer: str = Field(description="The correct answer, should be A, B, C, or D")
    explanation: str = Field(description="A brief explanation of the correct answer")

class Quiz(BaseModel):
    """
    Pydantic model for structuring a complete quiz.
    """
    questions: List[QuizQuestion] = Field(description="A list of quiz questions")

# Function to clean up old sessions
def cleanup_old_sessions(max_age=3600):  # max_age in seconds (1 hour)
    """
    Remove old session data from faiss_indexes.
    
    Args:
        max_age (int): Maximum age of a session in seconds before it's removed. Default is 1 hour.
    """
    current_time = time.time()
    to_delete = []
    for session_id, session_data in faiss_indexes.items():
        if current_time - session_data['timestamp'] > max_age:
            to_delete.append(session_id)
    for session_id in to_delete:
        del faiss_indexes[session_id]

def create_topic_extraction_chain():
    """
    Create a chain for extracting topics from text.
    
    Returns:
        Chain: A LangChain chain for topic extraction.
    """
    llm = ChatVertexAI(model_name="gemini-1.5-flash-001")
    prompt = PromptTemplate.from_template(
        "Extract the main and general topics from the following text which is delimited by triple backticks. Ensure that there is no repeating topics. Do not create too many topics. List them as comma-separated values.\n\n```{text}```"
    )
    return prompt | llm | StrOutputParser()

def create_quiz_generation_chain(vectorstore,search_query):
    """
    Create a chain for generating quiz questions.
    
    Args:
        vectorstore: The FAISS vector store containing the document embeddings.
        search_query (str): The search query to retrieve relevant documents.
    
    Returns:
        Chain: A LangChain chain for quiz generation.
    """
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    llm = ChatVertexAI(model_name="gemini-1.5-flash-001",temperature=0.1,max_tokens=8000)
    retriever = vectorstore.as_retriever()
    parser = JsonOutputParser(pydantic_object=Quiz)
    prompt = PromptTemplate(
        template="""Generate an in-depth academic quiz based on the following parameters:

        Text: {text}
        Topics: {topics}
        Number of questions: {num_questions}
        Difficulty: {difficulty}
        
        For each question:
        1. Include only multiple-choice questions with 4 options (A, B, C, D).
        2. The questions will be in {difficulty} difficulty
        3. Specify which cognitive level of Bloom's Taxonomy it addresses (Remember, Understand, Apply, Analyze, Evaluate, Create).
        4. Provide the correct answer and a brief explanation.
        5. Ensure that the concept each question tests does not overlap

        {format_instructions}

        Ensure the questions are precise, academically rigorous, and cover the main topics extracted from the text.
        Do not include ```json or ``` in your output.
        """,
        input_variables=["text", "topics", "num_questions", "difficulty"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    return RunnablePassthrough.assign(text= lambda x: format_docs(retriever.get_relevant_documents(search_query))) | prompt | llm | parser

def create_summary_generation_chain(vectorstore, search_query):
    """
    Create a chain for generating summaries.
    
    Args:
        vectorstore: The FAISS vector store containing the document embeddings.
        search_query (str): The search query to retrieve relevant documents.
    
    Returns:
        Chain: A LangChain chain for summary generation.
    """
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    llm = ChatVertexAI(model_name="gemini-1.5-flash-001",temperature=0.1,max_tokens=8000)
    retriever = vectorstore.as_retriever()
    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
            You are an expert academic tutor preparing students for exams. Given the following academic context, provide a comprehensive summary focusing on key concepts related.

            Academic Context:
            {text}

            Please provide an academic summary that:
            1. Defines and explains the main concepts related to the topic
            2. Highlights important theories, models, or frameworks
            3. Provides relevant examples, case studies, or applications

            Your summary should be structured as follows:

            I. Key Concepts and Definitions:
            [List key concepts with brief definitions]

            II. Important Theories/Models:
            [Outline relevant theories or models]

            III. Examples and Applications:
            [Provide practical examples or case studies]

            Summary:
            """
    )
    return RunnablePassthrough.assign(text= lambda x: format_docs(retriever.get_relevant_documents(search_query))) | prompt | llm | StrOutputParser()

def process_file(file_path):
    """
    Process uploaded file and create FAISS index.
    
    Args:
        file_path (str): Path to the uploaded file.
    
    Returns:
        FAISS: A FAISS index created from the processed file.
    """
    loader = UnstructuredWordDocumentLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Create new FAISS index
    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")


    # store embeddings in vector store
    faiss_index = FAISS.from_documents(texts, embeddings)
    return faiss_index

def generate_topics(index):
    """
    Generate topics from the FAISS index.
    
    Args:
        index: The FAISS index containing the document embeddings.
    
    Returns:
        list: A list of extracted topics.
    """
    docs = []
    for i in list(index.index_to_docstore_id.values()):
        docs.append(index.docstore.search(i))
    combined_text = "\n".join([doc.page_content for doc in docs])
            
            # Extract topics
    topic_chain = create_topic_extraction_chain()
    topics = topic_chain.invoke(combined_text)
    topics = topics.strip().split(", ")
    return topics


def generate_quiz(index, num_questions, difficulty, topic):
    """
    Generate quiz questions.
    
    Args:
        index: The FAISS index containing the document embeddings.
        num_questions (int): Number of questions to generate.
        difficulty (str): Difficulty level of the questions.
        topic (str): The topic for which to generate questions.
    
    Returns:
        list: A list of generated quiz questions.
    """
    quiz_chain = create_quiz_generation_chain(index, topic)
    quiz = quiz_chain.invoke({
                "topics": topic,
                "num_questions": f"{num_questions}",
                "difficulty": difficulty
            })
    return quiz['questions']

def generate_summary(index, topic):
    """
    Generate summary.
    
    Args:
        index: The FAISS index containing the document embeddings.
        topic (str): The topic for which to generate a summary.
    
    Returns:
        str: The generated summary.
    """
    summary_chain = create_summary_generation_chain(index, topic)
    summary = summary_chain.invoke({})
    return summary

def upload_page():
    """
    Streamlit page for file upload.
    """
    st.title("Academic Quiz Generator")
    uploaded_file = st.file_uploader("Choose a DOCX file", type="docx")
    # Create a unique session ID if not exists
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        print(st.session_state.session_id)

    if uploaded_file is not None and uploaded_file != st.session_state.current_file:
        st.session_state.current_file = uploaded_file
        with st.status(f"Processing {uploaded_file.name}... Please Wait", expanded=True) as status:
            st.write(f"Loading {uploaded_file.name}... ")
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                # Save the uploaded file
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                st.write(f"Indexing {uploaded_file.name}... ")
                faiss_index  = process_file(file_path)

                # Store the FAISS index in memory with a timestamp
                faiss_indexes[st.session_state.session_id] = {
                    'index': faiss_index,
                    'timestamp': time.time()
                }
            status.update(label=f"File '{uploaded_file.name}' processed and indexed.", state="complete", expanded=False)
    if faiss_indexes.get(st.session_state.session_id) is not None:
        if st.button("Create Quiz"):
            st.session_state.page = 'generate_quiz'
            st.rerun()
        if st.button("Create Summary"):
            st.session_state.page = 'generate_summary'
            st.rerun()


def generate_quiz_page():
    """
    Streamlit page for quiz generation.
    """
    st.title("Generate Quiz")
    faiss_index = faiss_indexes.get(st.session_state.session_id, {'index': None})['index']
    if faiss_index and st.session_state.quiz is None:
        topics = generate_topics(faiss_index)
        st.subheader("Quiz Generation Options")
        with st.form(key='quiz_form'):
            num_questions = st.number_input("Number of questions", min_value=1, max_value=20, value=5)
            difficulty = st.select_slider("Difficulty", options=["Easy", "Medium", "Hard"], value="Medium")
            topic  = st.selectbox("Topics", options=topics)
            submit_button = st.form_submit_button(label='Generate Quiz')

        if submit_button:
            st.info("Generating Quiz. Please Wait.")
            st.session_state.quiz = generate_quiz(faiss_index, num_questions, difficulty, topic)
            st.session_state.current_question = 0
            st.session_state.page = 'quiz'
            st.session_state.score = None
            st.success("Quiz Generated")
            st.rerun()
    elif not faiss_index:
        st.session_state.page = 'upload'
        st.info("Please upload DOCX file to start a new quiz.")
        st.rerun()
    
    if st.button("Back to Upload"):
        st.session_state.page = 'upload'
        st.rerun()

    cleanup_old_sessions()

def generate_summary_page():
    """
    Streamlit page for summary generation.
    """
    st.title("Generate Summary")
    faiss_index = faiss_indexes.get(st.session_state.session_id, {'index': None})['index']
    if faiss_index and st.session_state.quiz is None:
        topics = generate_topics(faiss_index)
        st.subheader("Summary Generation Options")
        with st.form(key='summary_form'):
            topic  = st.selectbox("Topics", options=topics)
            submit_button = st.form_submit_button(label='Generate Summary')

        if submit_button:
            st.info("Generating Summary. Please Wait.")
            st.session_state.summary = generate_summary(faiss_index, topic)
            st.session_state.page = 'summary'
            st.success("Summary Generated")
            st.rerun()
    elif not faiss_index:
        st.session_state.page = 'upload'
        st.info("Please upload DOCX file to start a new quiz.")
        st.rerun()
    
    if st.button("Back to Upload"):
        st.session_state.page = 'upload'
        st.rerun()

    cleanup_old_sessions()

def quiz_page():
    """
    Streamlit page for taking the quiz.
    """
    st.title("Take Quiz")
    if st.session_state.quiz is not None:
        if st.session_state.score is None:
            st.session_state.score = [0] * len(st.session_state.quiz)
        if st.session_state.current_question < len(st.session_state.quiz):
            question = st.session_state.quiz[st.session_state.current_question]
            st.subheader(f"Question {st.session_state.current_question + 1}")
            st.write(question['question'])
            
            user_answer = st.radio("Select your answer:", list(question['options'].items()), format_func=lambda x: f"{x[0]}) {x[1]}")
            
            if st.button("Submit Answer"):
                if user_answer[0] == question['correct_answer']:
                    st.success("Correct!")
                    st.session_state.score[st.session_state.current_question] = 1
                else:
                    st.error(f"Incorrect. The correct answer is {question['correct_answer']}: {question['options'][question['correct_answer']]}")
                
                st.write(f"Explanation: {question.get('explanation', '')}")
                
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Previous Question") and st.session_state.current_question > 0:
                    st.session_state.current_question -= 1
                    st.rerun()
            with col2:
                if st.button("Next Question"):
                    st.session_state.current_question += 1
                    st.rerun()
        else:
            st.subheader("Quiz Completed!")
            st.write(f"Your final score: {sum(st.session_state.score)} out of {len(st.session_state.quiz)}")
            if st.button("End Quiz"):
                st.write(f"Final Score: {st.session_state.score}/{len(st.session_state.quiz)}")
                st.session_state.quiz = None
                st.session_state.current_question = 0
                st.session_state.score = None
                if st.session_state.session_id in faiss_indexes:
                    del faiss_indexes[st.session_state.session_id]
                st.session_state.clear()
                st.session_state.page = 'upload'
                st.rerun()
            if st.button("Generate New Quiz"):
                st.session_state.quiz = None
                st.session_state.current_question = 0
                st.session_state.score = None
                st.session_state.page = 'generate_quiz'
                st.rerun()

def summary_page():
    """
    Streamlit page for reading and downloading summary.
    """
    st.title("Summary")
    if st.session_state.summary is not None:
        st.text(st.session_state.summary)
        st.download_button(
            label="Download Summary",
            data=st.session_state.summary,
            file_name=f"summary_{st.session_state.current_file.name.rsplit('.', 1)[0]}.txt",
            mime="text/plain"
        )
        if st.button("Generate New Summary"):
            st.session_state.summary = None
            st.session_state.page = 'generate_summary'
            st.rerun()
        if st.button("Back to Upload"):
            if st.session_state.session_id in faiss_indexes:
                del faiss_indexes[st.session_state.session_id]
            st.session_state.clear()
            st.session_state.page = 'upload'
            st.rerun()
        

def main():

    if 'page' not in st.session_state:
        st.session_state.page = 'upload'
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    if 'texts' not in st.session_state:
        st.session_state.texts = None
    if 'quiz' not in st.session_state:
        st.session_state.quiz = None
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'score' not in st.session_state:
        st.session_state.score = None
    if 'summary' not in st.session_state:
        st.session_state.summary = None


    # Page routing
    if st.session_state.page == 'upload':
        upload_page()
    elif st.session_state.page == 'generate_quiz':
        generate_quiz_page()
    elif st.session_state.page == 'generate_summary':
        print("go summary")
        generate_summary_page()
    elif st.session_state.page == 'quiz':
        quiz_page()
    elif st.session_state.page == 'summary':
        summary_page()

if __name__ == "__main__":
    main()