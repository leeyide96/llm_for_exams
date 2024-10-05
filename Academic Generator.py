import streamlit as st
import os,time, uuid
import vertexai
import tempfile
from pinecone import Pinecone, ServerlessSpec
from utils.functions import *

PROJECT_ID = os.getenv("PROJECTID")  # @param {type:"string"}
REGION = "us-central1"  # @param {type:"string"}
index_name = "academic-generator-index"

# Initialize Vertex AI SDK
vertexai.init(project=PROJECT_ID, location=REGION)

pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

if index_name not in [index['name'] for index in pinecone.list_indexes().indexes]:
    pinecone.create_index(index_name, dimension=768, metric="cosine", spec=ServerlessSpec(cloud="aws",region="us-east-1"))  # Adjust dimension based on your embedding model

@st.cache_resource
def create_index_dict():
    """
    Create a cached resource for storing pinecone indexes.
    
    Returns:
        dict: An empty dictionary to store pinecone indexes.
    """
    return {}
pinecone_indexes = create_index_dict()


def cleanup_old_sessions(max_age=3600):  # max_age in seconds (1 hour)
    """
    Remove old session data from pinecone_indexes.
    
    Args:
        max_age (int): Maximum age of a session in seconds before it's removed. Default is 1 hour.
    """
    current_time = time.time()
    to_delete = []
    for session_id, session_data in pinecone_indexes.items():
        if current_time - session_data['timestamp'] > max_age:
            to_delete.append(session_id)
    for session_id in to_delete:
        del pinecone_indexes[session_id]


def cleanup_old_namespaces(max_age=1):
    """
    Remove old namespaces from Pinecone index DB.
    
    Args:
        max_age (int): Maximum age of a session in days before it's removed. Default is 30 days.
    """
    current_time = time.time()
    index = pinecone.Index(index_name)
    namespaces = index.describe_index_stats()["namespaces"].keys()
    for namespace in namespaces:
        ids = list(index.list(namespace=namespace, limit=5))[0]
        ids_info = index.fetch(ids=ids, namespace=namespace)["vectors"].values()
        latest_timestamp = max([info["metadata"]["timestamp"] for info in ids_info])
        age = current_time - latest_timestamp
        if age > max_age * 24 * 60 * 60:
            index.delete(namespace=namespace, delete_all=True)
          

def upload_page():
    """
    Streamlit page for file upload.
    """
    cleanup_old_sessions()
    st.title("Academic Quiz/Summary Generator", anchor=False)
    st.info("Uploading a file for the first time will require indexing which will take a long period of time due to quota issues. Subsequent upload of the same file will be faster as indexing is no longer required")
    uploaded_file = st.file_uploader("Choose a DOCX/DOC/PDF file", type=["docx", "doc", "pdf"])

    if (uploaded_file is not None and uploaded_file != st.session_state.current_file) or st.session_state.reindex:
        st.session_state.current_file = uploaded_file
        with st.status(f"Processing {uploaded_file.name}... Please Wait", expanded=True) as status:
            st.write(f"Loading {uploaded_file.name}... ")
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                file_hash = compute_file_hash(uploaded_file)

                st.write(f"Indexing {uploaded_file.name}... ")
                try:
                    pinecone_index, full_docs, skip_indexing  = process_file(index_name, file_path, file_hash, st.session_state.reindex)
                    # Store the pinecone index in memory with a timestamp
                    pinecone_indexes[st.session_state.session_id] = {
                        'index': pinecone_index,
                        'namespace': file_hash,
                        'documents': full_docs,
                        'skip_indexing': skip_indexing,
                        'timestamp': time.time()
                    }
                    if pinecone_index is None:
                        status.update(label=f"The file {uploaded_file.name} has unknown sentences in it. Please reupload.", state="error", expanded=False)
                    else:
                        st.session_state.topics = None
                        status.update(label=f"File '{uploaded_file.name}' processed.", state="complete", expanded=False)
                except Exception as e:
                     status.update(label=f"As there is limited quota for running Gemini or other error {e}, please try again a minute later.", state="error", expanded=False)
                     st.error(f"{e}")
                st.session_state.reindex = False
                    
    if uploaded_file is not None and pinecone_indexes.get(st.session_state.session_id) is not None:

        st.session_state.pinecone_index = pinecone_indexes.get(st.session_state.session_id, {'index': None})['index']
        st.session_state.docs = pinecone_indexes.get(st.session_state.session_id, {'documents': None})['documents']
        placeholder = st.empty()
        with placeholder.container():
            if st.button("Create Quiz"):
                st.session_state.page = 'generate_quiz'
                st.rerun()
            if st.button("Create Summary"):
                st.session_state.page = 'generate_summary'
                st.rerun()
            if pinecone_indexes.get(st.session_state.session_id)["skip_indexing"]:
                if st.button("Reindex file"):
                   placeholder.empty()
                   st.session_state.reindex=True
                   st.rerun()

def generate_quiz_page():
    """
    Streamlit page for quiz generation.
    """

    st.title("Generate Quiz", anchor=False)
    st.session_state.update({"topic": None,
                             "topic_type": None})
    if st.session_state.pinecone_index and st.session_state.docs and st.session_state.quiz is None:
        if st.session_state.topics is None:
            st.session_state.topics = generate_topics(st.session_state.docs)
        st.subheader("Quiz Generation Options", anchor=False)
        with st.container(border=True):
            topic_type = st.radio(
                "Which one do you prefer? Your Custom Topics or Generated Topics?",
                ["Your Custom Topics", "Generated Topics"], horizontal=True)
            if topic_type == "Your Custom Topics":
                topic = st.text_input("Key in your own topics", max_chars=100,
                                      placeholder="All Topics",
                                      help="If there are multiple topics, use comma to separate them")
            else:
                topic = st.selectbox("Topics",
                                     options=["All Topics"] + st.session_state.topics)
            num_questions = st.number_input("Number of questions", min_value=1, max_value=20, value=5)
            difficulty = st.select_slider("Difficulty", options=["Easy", "Medium", "Hard"], value="Medium")
            with st.form(key='quiz_form', border=False):
                submit_button = st.form_submit_button(label='Generate Quiz')

        if submit_button:
            st.info("Generating Quiz. Please Wait.")
            try:
                st.session_state.update({"quiz": generate_quiz(st.session_state.pinecone_index, num_questions, difficulty, topic, st.session_state.docs),
                                         "current_question": 0,
                                         "score": None,
                                         "page": "quiz"})
                st.success("Quiz Generated")
                st.rerun()
            except Exception as e:
                st.error("There is some error. Please try again.")
            
    elif not st.session_state.pinecone_index and not st.session_state.docs:
        st.session_state.page = 'upload'
        st.info("Please upload DOCX file to start a new quiz.")
        st.rerun()
    
    if st.button("Back to Upload"):
        st.session_state.page = 'upload'
        st.rerun()


def generate_summary_page():
    """
    Streamlit page for summary generation.
    """
    st.title("Generate Summary", anchor=False)
    if st.session_state.pinecone_index and st.session_state.docs and st.session_state.quiz is None:
        if st.session_state.topics is None:
            st.session_state.topics = generate_topics(st.session_state.docs)
        st.subheader("Summary Generation Options", anchor=False)
        with st.container(border=True):
            topic_type = st.radio(
                "Which one do you prefer? Your Custom Topics or Generated Topics?",
                ["Your Custom Topics", "Generated Topics"], horizontal=True)
            if topic_type == "Your Custom Topics":
                topic = st.text_input("Key in your own topics", max_chars=100,
                                                       placeholder="All Topics",
                                                       help="If there are multiple topics, use comma to separate them")
            else:
                topic = st.selectbox("Topics", options=["All Topics"] + st.session_state.topics)
            conciseness = st.select_slider("Conciseness", options=["Brief", "Detailed", "Verbose"], value="Detailed")
            provide_layman = st.checkbox("Provide layman explanation")
            with st.form(key='summary_form', border=False):
                submit_button = st.form_submit_button(label='Generate Summary')

        if submit_button:
            status = st.empty()
            status.info("Generating Summary. Please Wait.")
            st.session_state.summary = generate_summary(st.session_state.pinecone_index, topic, st.session_state.docs, conciseness, provide_layman)
            if st.session_state.summary:
                st.session_state.page = 'summary'
                status.success("Summary Generated")
                st.rerun()
            else:
                status.error("Gemini Quota has reached. Please try again after awhile.")
    elif not st.session_state.pinecone_index and not st.session_state.docs:
        st.session_state.page = 'upload'
        st.info("Please upload DOCX file to start a new quiz.")
        st.rerun()
    
    if st.button("Back to Upload"):
        st.session_state.page = 'upload'
        st.rerun()


def quiz_page():
    """
    Streamlit page for taking the quiz.
    """
    st.title("Take Quiz", anchor=False)
    if st.session_state.quiz is not None:
        if st.session_state.score is None:
            st.session_state.score = [0] * len(st.session_state.quiz)
        if st.session_state.current_question < len(st.session_state.quiz):
            next_button_title = "Next Question" if st.session_state.current_question < len(st.session_state.quiz) - 1 else "Finish Quiz"
            question = st.session_state.quiz[st.session_state.current_question]
            st.subheader(f"Question {st.session_state.current_question + 1}", anchor=False)
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
                if st.session_state.current_question > 0 and st.button("Previous Question"):
                    st.session_state.current_question -= 1
                    st.rerun()
            with col2:
                if st.button(next_button_title):
                    st.session_state.current_question += 1
                    st.rerun()
        else:
            st.subheader("Quiz Completed!", anchor=False)
            st.write(f"Your final score: {sum(st.session_state.score)} out of {len(st.session_state.quiz)}")
            all_answers = ""
            for i in range(len(st.session_state.quiz)):
                all_answers += f"{i+1}. Question: {st.session_state.quiz[i]['question']} \n\n\t Answer: {st.session_state.quiz[i]['correct_answer']} - {st.session_state.quiz[i]['explanation']} \n\n"
            st.markdown(all_answers)
            if st.button("End Quiz"):
                st.write(f"Final Score: {st.session_state.score}/{len(st.session_state.quiz)}")
                st.session_state.update({"quiz": None,
                                         "current_question": 0,
                                         "score": None})
                if st.session_state.session_id in pinecone_indexes:
                    del pinecone_indexes[st.session_state.session_id]
                st.session_state.clear()
                st.session_state.page = 'upload'
                st.rerun()
            if st.button("Generate New Quiz"):
                st.session_state.update({"quiz": None,
                                         "current_question": 0,
                                         "score": None,
                                         "page": "generate_quiz"})
                st.rerun()


def summary_page():
    """
    Streamlit page for reading and downloading summary.
    """
    st.title("Summary", anchor=False)
    if st.session_state.summary is not None:
        st.markdown(st.session_state.summary)
        st.download_button(
            label="Download Summary",
            data=st.session_state.summary,
            file_name=f"summary_{st.session_state.current_file.name.rsplit('.', 1)[0]}.txt",
            mime="text/plain"
        )
        if st.button("Generate New Summary"):
            st.session_state.update({"summary": None,
                                     "page": "generate_summary"})
            st.rerun()
        if st.button("Back to Upload"):
            if st.session_state.session_id in pinecone_indexes:
                del pinecone_indexes[st.session_state.session_id]
            st.session_state.clear()
            st.session_state.page = 'upload'
            st.rerun()
        

def main():

    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'page' not in st.session_state or pinecone_indexes.get(st.session_state.session_id) is None:
        st.session_state.page = 'upload'

    states = {"current_file", "pinecone_index", "namespace", "docs", "quiz",
              "current_question", "score", "summary", "topics", "reindex"}
    existing_states = st.session_state.keys()
    new_states = states - existing_states
    for state in new_states:
        value = None
        if state == "current_question":
            value = 0
        elif state == "reindex":
            value = False
        st.session_state[state] = value

    # Page routing
    page_functions = {"upload": upload_page, "generate_quiz": generate_quiz_page,
                      "generate_summary": generate_summary_page, "quiz": quiz_page,
                      "summary": summary_page}
    to_run = page_functions[st.session_state.page]
    to_run()

if __name__ == "__main__":
    main()