import os
from dotenv import load_dotenv
from src.libs.WatsonxAiLib import WatsonxAiLib
from src.utils.AppUtil import get_chroma_client, store_document_chromadb, vectorize_data

PATH = os.getcwd()
ASSETS_DIR_PATH = str(PATH) + os.sep + "assets" + os.sep + "proposals"
load_dotenv()
WatsonxAiLib.initialize_llm_embedding()

  
embeddings = WatsonxAiLib.get_embeddings()
collection_name = "response_to_rfp_collection"
chroma_client = get_chroma_client()
chroma_client.get_or_create_collection(
        name=collection_name)
store_document_chromadb(chroma_client, embeddings, collection_name,
                            ASSETS_DIR_PATH + os.sep + "full_stack.txt", ["ReactJs", "NextJs", "Vue.js", "AngularJS", "Node.js", "Next.js",
                                                                          "Javascript", "MongoDB", "PostgreSQL", "Elasticsearch"])


store_document_chromadb(chroma_client, embeddings, collection_name,
                            ASSETS_DIR_PATH + os.sep + "Mobile.txt", ["IOS", "Android", "Vue.js", "Mobile Development","Firebase"])

store_document_chromadb(chroma_client, embeddings, collection_name,
                            ASSETS_DIR_PATH + os.sep + "Salesforce_Azure.txt", ["Salesforce", "Azure"])

print("Job Completed!")


