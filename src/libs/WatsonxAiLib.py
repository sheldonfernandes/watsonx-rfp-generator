import os
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams

from src.constants.AppConst import AppConst


class WatsonxAiLib:
    @staticmethod
    def initialize_llm_embedding():
        params = {
            GenTextParamsMetaNames.DECODING_METHOD: "greedy",
            GenTextParamsMetaNames.MAX_NEW_TOKENS: 1000,
            GenTextParamsMetaNames.MIN_NEW_TOKENS: 0,
            GenTextParamsMetaNames.REPETITION_PENALTY: 1
        }

        global watsonx_ai_llm
        watsonx_ai_llm = WatsonxLLM(
            model_id="ibm/granite-3-8b-instruct",
            url=os.getenv("WATSONX_AI_API", None),
            apikey=os.getenv("WATSONX_AI_KEY", None),
            project_id=os.getenv("WATSONX_AI_PROJECT_ID", None),
            params=params,
            streaming=False,
        )
        global embeddings
        embeddings = WatsonxEmbeddings(
            model_id=AppConst.WX_EMBEDDING_MODEL_ID,
            url=os.getenv("WATSONX_AI_API", None),
            apikey=os.getenv("WATSONX_AI_KEY", None),
            project_id=os.getenv("WATSONX_AI_PROJECT_ID", None),
            params= {
                EmbedParams.TRUNCATE_INPUT_TOKENS: 128,
                EmbedParams.RETURN_OPTIONS: {
                    'input_text': True
                }
            }
        )

    def get_embeddings():
        return embeddings

    def get_watsonx_ai_llm():
        return watsonx_ai_llm
