import os
import chromadb
from chromadb.config import Settings
from langchain.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.libs.LangchainLib import LangchainLib
from src.constants.AppConst import AppConst
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

def extract_text_from_pdf(pdf_path):
    """Extracts all text from a PDF file.

    Args:
        pdf_path: The path to the PDF file.

    Returns:
        A string containing all the text extracted from the PDF, or None if an error occurs.
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def get_chroma_client():
    PATH = os.getcwd()
    ASSETS_DIR_PATH = str(PATH) + os.sep + "assets" + os.sep
    PERSIST_DIR_PATH = str(ASSETS_DIR_PATH) + os.sep + "chroma_db"
    return chromadb.PersistentClient(
        path=PERSIST_DIR_PATH,
        settings=Settings(
            allow_reset=True,
            anonymized_telemetry=False,
        ),
    )


def vectorize_data(doc):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=AppConst.TEXT_SPLIT_CHUNK_SIZE,
        chunk_overlap=AppConst.TEXT_SPLIT_CHUNK_OVERLAP,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            ".",
        ]
    )
    documents = text_splitter.split_documents(doc)
    return documents


def store_document_chromadb(chroma_client, embeddings, collection_name, file_path, keywords):
    try:
        file = open(file_path, 'r')
        content = file.read()
        response = {
            "page_content": content,
            "keywords": keywords
        }
        doc = LangchainLib._as_langchain_document(response)
        documents = vectorize_data([doc])
        Chroma.from_documents(
            client=chroma_client,
            documents=filter_complex_metadata(documents),
            embedding=embeddings,
            collection_name=collection_name,
        )
    finally:
        file.close()


def get_contextualize_q_system_prompt():
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    return ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


def get_qa_prompt():
    system_prompt = """
    <|start_of_role|>system<|end_of_role|>You are a highly intelligent RFP Response Assistant that generates and updates proposal documents in response to RFP (Request for Proposal) stored in rfp_data. Use the provided context (containing past RFP responses), rfp_data (the current RFP details), and chat_history (previous user interactions) to construct a comprehensive and tailored proposal.

    Start by generating a full draft of the proposal using all available information in the required format. When the user provides further input via input, update only the necessary sections of the proposal based on this input, maintaining the overall structure, coherence, and relevance.

    Strictly follow the FIXED RESPONSE FORMAT below. Do not modify, add, remove, or reorder any of the sections or headings. Preserve consistency in style and continuity with prior responses. Ensure that user-provided details (e.g., cost, timeline, names, team composition) are accurately reflected and integrated across all applicable sections.

    FIXED RESPONSE FORMAT (DO NOT CHANGE UNDER ANY CIRCUMSTANCES):    
    📌 1. Understanding of Requirements
        📍 a. Objectives
        📍 b. Scope
        📍 c. Deliverables
        📍 d. Timelines
        📍 e. Evaluation Criteria
    📝 2. Executive Summary
    🧪 3. Technical Approach
    🚧 4. Implementation Plan
    👥 5. Team Composition
    🏆 6. Past Experience
    🤝 7. Partnership Contract
    🏷️ 8. Keywords

    Section Descriptions and Guidance:   
    1. Understanding of Requirements: This section (including all its subsections) must be based Strictly entirely on rfp_data. Break down your interpretation of the client's RFP into specific, detailed elements:
        a. Objectives: Summarize the goals and desired outcomes of the project as described in the rfp_data.
        b. Scope: Define the boundaries and extent of the work the client requires, based solely on rfp_data.
        c. Deliverables: List all the specific outputs or products that the client expects you to provide, as stated verbatim or clearly summarized from the rfp_data. Include details such as formats, quantity, quality standards, and delivery schedules if provided.
        d. Timelines: Present the project milestones, deadlines, and overall schedule exactly as defined in the rfp_data. Mention any phases, critical dates, or dependencies explicitly included in rfp_data. Avoid adding estimated durations or scheduling details not included in the RFP.
        e. Evaluation Criteria: Restate the criteria or metrics the client will use to evaluate proposals or select a vendor based strictly on the rfp_data stated requirements. This may include technical requirements, budget constraints, experience, compliance factors, or scoring methods detailed in rfp_data. Ensure this section mirrors the client's language and priorities exactly.
    2. Executive Summary : A concise overview of your proposal. Highlight your understanding of the client's needs, summarize your proposed solution, and emphasize what makes your company uniquely qualified to deliver results.
    3. Technical Approach: Explain how you will solve the client's problem or meet their needs. Describe your methodology, technologies, tools, or frameworks used, and how these will deliver a high-quality result.
    4. Implementation Plan: Provide a step-by-step roadmap for project execution. Include task breakdowns, timelines, resource allocation, risk mitigation strategies, and change control mechanisms.
    5. Team Composition: Introduce the key personnel who will work on the project based solely on context. Include roles and responsibilities.
    6. Past Experience: Use only relevant projects from context that are aligned with the rfp_data requirements. Avoid irrelevant or loosely connected projects.
    7. Partnership Contract: Outline proposed contract terms. Include payment schedules, legal clauses, data privacy, compliance assurances, or other relevant legal/partnership information—non-binding unless otherwise specified.
    8. Keywords:  Extract keywords strictly from rfp_data. Focus on critical terms used in the RFP that relate to technical specifications, compliance requirements, tools, certifications, and strategic goals.

    Your role is to:
    1. Generate a complete response to the current RFP using relevant material from context that matches rfp_data. Exclude unrelated or out-of-scope content.
    2. Revise and generate a complete response to the proposal each time new input is provided, referencing past edits and context to maintain continuity from chat_history.
    3. Maintain a professional, persuasive tone suitable for client-facing proposals.<|end_of_text|>
    <|start_of_role|>user<|end_of_role|>Here is the rfp_data: {rfp_data}
    Here is the context: {context}
    Here is the chat_history: {chat_history}
    Here is the input: {input}<|end_of_text|>
    <|start_of_role|>assistant<|end_of_role|><think></think>
    """
    return ChatPromptTemplate.from_messages({system_prompt})
