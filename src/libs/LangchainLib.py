from langchain.docstore.document import Document
class LangchainLib:
    @classmethod
    def _as_langchain_document(cls, d: dict) -> Document:
        doc = Document(
            page_content=d.get("page_content", ""),
            metadata={
                "keywords": d.get("keywords", ""),
            },
        )

        return doc
