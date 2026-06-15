"""
LLM Executive Briefing Engine - RAG-powered document summarization
CEO-Track Portfolio Project
"""
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import logging

logger = logging.getLogger(__name__)

BRIEF_TEMPLATE = """You are an elite executive advisor.
Produce a structured board briefing from these document excerpts.
Context: {context}
Request: {question}

## EXECUTIVE SUMMARY
[2-3 sentence synthesis]

## KEY FINDINGS
[3-5 numbered findings with data]

## RISKS IDENTIFIED
[Regulatory, competitive, financial, operational risks]

## DECISIONS REQUIRED
[2-3 decisions leadership must make]

## RECOMMENDED ACTIONS
[Concrete next steps with owners and timelines]"""


class BriefingEngine:
    """Core RAG engine for executive document briefings."""

    def __init__(self, api_key: str, persist_dir: str = ".chroma"):
        self.api_key = api_key
        self.embeddings = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-large")
        self.llm = ChatOpenAI(api_key=api_key, model="gpt-4o", temperature=0.1, max_tokens=2000)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        self.vectorstore = None
        self.retriever = None
        self.persist_dir = persist_dir
        self.prompt = PromptTemplate(template=BRIEF_TEMPLATE, input_variables=["context", "question"])

    def ingest_document(self, text: str, doc_id: str, metadata: dict = None) -> int:
        """Ingest a document into the vector store."""
        logger.info(f"Ingesting document: {doc_id}")
        chunks = self.text_splitter.split_text(text)
        meta = [{"doc_id": doc_id, "chunk": i, **(metadata or {})} for i in range(len(chunks))]
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_texts(
                texts=chunks, embedding=self.embeddings,
                metadatas=meta, persist_directory=self.persist_dir
            )
        else:
            self.vectorstore.add_texts(texts=chunks, metadatas=meta)
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20}
        )
        logger.info(f"Ingested {len(chunks)} chunks from {doc_id}")
        return len(chunks)

    def generate_brief(self, query: str = None) -> dict:
        """Generate an executive brief from ingested documents."""
        if self.retriever is None:
            raise ValueError("No documents ingested. Call ingest_document() first.")
        default_q = "Provide comprehensive executive briefing: findings, risks, decisions."
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True
        )
        result = qa_chain({"query": query or default_q})
        sources = list(set([
            doc.metadata.get("doc_id", "Unknown")
            for doc in result.get("source_documents", [])
        ]))
        return {"brief": result["result"], "sources": sources,
                "source_count": len(result.get("source_documents", []))}

    def extract_risks(self) -> str:
        """Dedicated risk extraction."""
        return self.generate_brief(
            query="Identify ALL risks: regulatory, financial, competitive, operational. Include severity."
        )["brief"]

    def extract_actions(self) -> str:
        """Extract action items and next steps."""
        return self.generate_brief(
            query="Extract all action items, commitments, owners, deadlines."
        )["brief"]

    def compare_docs(self, doc_a: str, doc_b: str) -> str:
        """Compare two documents for delta briefing."""
        return self.generate_brief(
            query=f"Compare {doc_a} vs {doc_b}: key changes, improvements, deteriorations."
        )["brief"]

    def reset(self):
        """Clear vector store."""
        self.vectorstore = None
        self.retriever = None
