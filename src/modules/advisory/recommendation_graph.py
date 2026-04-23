import copy
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

from src.modules.advisory.dto import RecommendationResult
from src.modules.advisory.repository import VectorStoreService
from src.modules.advisory.prompts import RAG_SYSTEM_PROMPT, RAG_HUMAN_PROMPT, SAFETY_FALLBACK_RESPONSE
from src.shared.utils.logger import logger
from config import settings
from src.modules.system.config_service import ConfigService


class RecommendationState(TypedDict):
    disease_events: list[dict]
    rag_context: list[Document]
    recommendation: dict | None
    error: str | None


class RecommendationGraphBuilder:
    """Builds and compiles the LangGraph RAG workflow for agronomic advice."""

    def __init__(self, vector_store: VectorStoreService):
        self._vector_store = vector_store

        base_llm = ChatGroq(
            model=settings.GROQ_LLM_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=0.1,
        )

        # Structured output: LLM output is parsed directly into the
        # Pydantic model via Groq's JSON mode, eliminating manual
        # json.loads() and markdown-fence stripping entirely.
        self._structured_llm = base_llm.with_structured_output(
            RecommendationResult,
            method="json_mode",
        )

        # ChatPromptTemplate keeps system/human roles separate,
        # improving instruction-following vs a single flat string.
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", RAG_HUMAN_PROMPT),
        ])

    # ── Node: Retrieve ────────────────────────────────────────────

    async def retrieve_document_context(
        self, state: RecommendationState,
    ) -> dict:
        disease_classes = [ev.get("disease_class", "") for ev in state.get("disease_events", [])]
        
        all_documents = []
        seen_contents = set()
        
        for disease in disease_classes:
            if not disease:
                continue
                
            query = f"Rice disease treatment, management, and nutritional advice: {disease}"
            logger.info(f"[RAG] Retrieving context for: {query}")

            try:
                results = await self._vector_store.similarity_search(query)
                
                for doc, score in results:
                    src = doc.metadata.get("source", "?")
                    logger.debug(f"  [{score:.3f}] {src}: {doc.page_content[:80]}...")
                    if doc.page_content not in seen_contents:
                        seen_contents.add(doc.page_content)
                        all_documents.append(doc)
            except Exception as e:
                logger.error(f"[RAG] Vector retrieval failed for {disease}: {e}")
                return {"rag_context": [], "error": str(e)}

        return {"rag_context": all_documents}

    # ── Router ────────────────────────────────────────────────────

    def _route_after_retrieval(self, state: RecommendationState) -> str:
        if state.get("error"):
            return "safety_fallback"
            
        severity_threshold = ConfigService.get_float("SEVERITY_THRESHOLD", 50000.0)
        
        total_area = sum(
            lesion.get("mask_area_px", 0)
            for ev in state.get("disease_events", [])
            for lesion in ev.get("lesions", [])
        )

        if total_area > severity_threshold:
            state["error"] = f"Total lesion area {total_area} exceeds severity threshold {severity_threshold}."
            return "safety_fallback"

        if not state.get("rag_context"):
            return "safety_fallback"
        return "generate_grounded_advice"

    # ── Node: Generate ────────────────────────────────────────────

    async def generate_grounded_advice(
        self, state: RecommendationState,
    ) -> dict:
        rag_context = state["rag_context"]
        disease_classes = [ev.get("disease_class", "") for ev in state.get("disease_events", [])]
        confidence = max([ev.get("confidence", 0.0) for ev in state.get("disease_events", [])] or [0.0])

        context_text = "\n\n---\n\n".join(
            f"[Nguồn: {doc.metadata.get('source', 'unknown')} | "
            f"Độ tương đồng: {doc.metadata.get('similarity', 0):.3f}]\n"
            f"{doc.page_content}"
            for doc in rag_context
        )

        # Build the prompt chain and invoke with structured output
        chain = self._prompt | self._structured_llm

        try:
            result: RecommendationResult = await chain.ainvoke({
                "disease_name": ", ".join(disease_classes),
                "confidence": confidence,
                "rag_context": context_text,
            })

            recommendation = result.model_dump()
            logger.info("[RAG] Grounded recommendation generated successfully")
            return {"recommendation": recommendation}

        except Exception as e:
            logger.error(f"[RAG] LLM generation failed: {e}")
            fallback = copy.deepcopy(SAFETY_FALLBACK_RESPONSE)
            fallback["disease_name"] = ", ".join(disease_classes)
            fallback["confidence_note"] = (
                f"Lỗi khi tạo khuyến cáo từ LLM: {type(e).__name__}. "
                "Sử dụng phản hồi an toàn mặc định."
            )
            return {"recommendation": fallback, "error": str(e)}

    # ── Node: Safety Fallback ─────────────────────────────────────

    async def safety_fallback(self, state: RecommendationState) -> dict:
        fallback = copy.deepcopy(SAFETY_FALLBACK_RESPONSE)
        disease_classes = [ev.get("disease_class", "") for ev in state.get("disease_events", [])]
        fallback["disease_name"] = ", ".join(disease_classes)

        reason = state.get("error", "Không tìm thấy tài liệu phù hợp")
        logger.warning(
            f"[RAG] Safety fallback for {disease_classes}. "
            f"Reason: {reason}"
        )
        return {"recommendation": fallback}

    # ── Graph Builder ─────────────────────────────────────────────

    def build(self):
        graph = StateGraph(RecommendationState)

        graph.add_node(
            "retrieve_document_context", self.retrieve_document_context
        )
        graph.add_node(
            "generate_grounded_advice", self.generate_grounded_advice
        )
        graph.add_node("safety_fallback", self.safety_fallback)

        graph.add_edge(START, "retrieve_document_context")
        graph.add_conditional_edges(
            "retrieve_document_context",
            self._route_after_retrieval,
            {
                "generate_grounded_advice": "generate_grounded_advice",
                "safety_fallback": "safety_fallback",
            },
        )
        graph.add_edge("generate_grounded_advice", END)
        graph.add_edge("safety_fallback", END)

        compiled = graph.compile()
        logger.info("[RAG] LangGraph recommendation workflow compiled")
        return compiled
