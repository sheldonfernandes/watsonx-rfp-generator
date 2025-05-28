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
        "page_content" : content,
        "keywords": keywords
        }
        doc = LangchainLib._as_langchain_document(response)
        print(doc)
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
    system_prompt = (
        """
        You are a senior proposal writer at a leading technology consultancy, experienced in crafting compelling and technically sound responses to complex RFPs (Request for Proposals). Your task is to analyze the provided RFP content in detail and generate a professional, persuasive, and fully compliant proposal response tailored to the client's needs.

        You will be given the RFP input and relevant context about your organization's capabilities, experience, and proposed solution. Using this information, generate a structured and detailed proposal document that:

        Aligns with the RFP's objectives, scope, and deliverables.

        Addresses each requirement and evaluation criterion explicitly.

        Demonstrates a deep understanding of the client's challenges and goals.

        Highlights your organization's relevant experience, technical expertise, and value proposition.

        Presents a clear, actionable, and feasible approach, methodology, and timeline.

        Be thorough, concise, and client-focused. Your response must maintain a formal tone, use industry-standard language, and follow the typical structure of a proposal (e.g., Executive Summary, Understanding of Requirements, Technical Approach, Implementation Plan, Team Composition, Past Experience, and Value Proposition).

        You will be provided with the following:
        RFP input where contains he RFP section, clause, or requirements to respond to. Here is the: {input}

        Context which contains Information about the consultancy and its capabilities relevant to the RFP.
        Here is the {context}

        Generate:
        A professional and complete proposal response to the RFP, grounded in the context.
        """
    )
    return ChatPromptTemplate.from_messages({system_prompt})
