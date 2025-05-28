import chainlit as cl
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from src.libs.WatsonxAiLib import WatsonxAiLib
from src.utils.AppUtil import extract_text_from_pdf, get_chroma_client, get_contextualize_q_system_prompt, get_qa_prompt
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


@cl.on_chat_start
async def on_chat_start():
    WatsonxAiLib.initialize_llm_embedding()
    files = None

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a RFP file to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    rfp_data = extract_text_from_pdf(file.path)
    msg.content = f"Processing `{file.name}` done!"
    await msg.update()

    msg = cl.Message(
        content=f"Generating Response to RFP Based on Knowledge Base...")
    await msg.send()

    chroma_client = get_chroma_client()

    vectorstore = Chroma(
        client=chroma_client,
        embedding_function=WatsonxAiLib.get_embeddings(),
        collection_name="response_to_rfp_collection",
        collection_metadata={"hnsw:space": "cosine"}
    )

    retriever = vectorstore.as_retriever()
    llm = WatsonxAiLib.get_watsonx_ai_llm()

    contextualize_q_system_prompt = get_contextualize_q_system_prompt()
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_system_prompt
    )

    qa_prompt = get_qa_prompt()
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(
        history_aware_retriever, question_answer_chain)
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    response = conversational_rag_chain.invoke(
        {"input": rfp_data},
        config={
            "configurable": {"session_id": "abc123"}
        },
    )["answer"]

    msg = cl.Message(
        content=f"Generated!")
    await msg.send()

    await cl.Message(
        content=response,
    ).send()

    cl.user_session.set("conversational_rag_chain", conversational_rag_chain)

@cl.on_message
async def on_message(message: cl.Message):
    conversational_rag_chain = cl.user_session.get("conversational_rag_chain")
    response = conversational_rag_chain.invoke(
        {"input": message.content},
        config={
            "configurable": {"session_id": "abc123"}
        },
    )["answer"]
    await cl.Message(
        content=response,
    ).send()
