from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.pydantic_v1 import BaseModel, Field
from typing import List, Dict


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


def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

def create_topic_extraction_chain():
    """
    Create a chain for extracting topics from text.
    
    Returns:
        Chain: A LangChain chain for topic extraction.
    """
    llm = ChatVertexAI(model_name="gemini-1.5-flash-001",temperature=0.1)
    prompt = PromptTemplate.from_template(
        """
        Extract the main important topics from the following text which is delimited by triple backticks.  
        Consider topics to be important if they are central to the main idea, frequently mentioned, or pivotal to the understanding of the text.
        Ensure that there is no repeating topics. Think twice before producing topics and make sure that the topics are relevant and necessary information can be extracted out from the topics.
        List them as comma-separated values.\n\n```{text}```
        """
    )
    return prompt | llm | StrOutputParser()


def create_quiz_generation_chain(vectorstore,search_query, docs):
    """
    Create a chain for generating quiz questions.
    
    Args:
        vectorstore: The Pinecone vector store containing the document embeddings.
        search_query (str): The search query to retrieve relevant documents.
        docs (list[Document]): Full list of LangChain documents.
    
    Returns:
        Chain: A LangChain chain for quiz generation.
    """
    
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
        6. If Topics are All Topics, ensure that every concept and topic is tested
        7. Ensure that the answer is right. Check it with yourself to ensure the explanation and answer is sound. If there is math involved, ensure that the math is correct too.

        {format_instructions}

        Ensure the questions are precise, academically rigorous, and cover the main topics extracted from the text. Ensure that the answer for the math and math datetime must be correct. 
        Do think twice before producing the output. Do not include ```json or ``` in your output.
        """,
        input_variables=["text", "topics", "num_questions", "difficulty"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    if search_query != "All Topics":
        docs = retriever.invoke(search_query)
        
    return RunnablePassthrough.assign(text= lambda x: format_docs(docs)) | prompt | llm | parser


def create_summary_generation_chain(vectorstore, search_query, docs):
    """
    Create a chain for generating summaries.
    
    Args:
        vectorstore: The Pinecone vector store containing the document embeddings.
        search_query (str): The search query to retrieve relevant documents.
        docs (list[Document]): Full list of LangChain documents.
    
    Returns:
        Chain: A LangChain chain for summary generation.
    """
    
    llm = ChatVertexAI(model_name="gemini-1.5-flash-001",temperature=0.1,max_tokens=8000)
    retriever = vectorstore.as_retriever()
    prompt = PromptTemplate(
        input_variables=["text", "conciseness", "layman"],
        template="""
            You are an expert academic tutor preparing students for exams. Given the following academic context, provide a comprehensive summary focusing on key concepts related.

            Academic Context:
            {text}

            Please provide an academic summary that:
            1. Defines and explains the main concepts related to the topic. If it is on All Topics, ensure that every main concepts are explained clearly.
            2. Highlights important theories, models, or frameworks
            3. Provides relevant examples, case studies, or applications
            4. It is in Github-flavored Markdown format.

            Think twice before producing the summary. Ensure that the summary is accurate. Your summary should be structured as follows:

            I. Key Concepts and Definitions:
            [List every key concepts with {conciseness} definitions. {layman}]

            II. Important Theories/Models:
            [Outline relevant theories or models]

            III. Examples and Applications:
            [Provide practical examples or case studies]

            Summary:
            """
    )
    if search_query != "All Topics":
        docs = retriever.invoke(search_query)
        
    return RunnablePassthrough.assign(text= lambda x: format_docs(docs)) | prompt | llm | StrOutputParser()
