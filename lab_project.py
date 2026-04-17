from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from crewai import Agent, Crew, Process, Task
from crewai.knowledge.source.json_knowledge_source import JSONKnowledgeSource
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from crewai.rag.embeddings.providers.cohere.types import CohereProviderSpec
from crewai.tools import BaseTool
from crewai_tools import RagTool, SerperDevTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
KNOWLEDGE_DIR = ROOT / "docs" / "knowledge"
STORAGE_DIR = ROOT / ".crewai_storage"
GENERATED_DIR = STORAGE_DIR / "generated"
DEFAULT_OUTPUT = ROOT / "report.json"
DEFAULT_MODEL = os.getenv("AGENT_REVIEW_MODEL") or os.getenv("MODEL") or "ollama/llama3.2:1b"
DEFAULT_USER_ID = "nnImk681KaRqUVHlSfZjGQ"
DEFAULT_ITEM_ID = "-7GjicSH_rM8JeZGCXGcUg"
MAX_RAG_REVIEWS = 500
MAX_REVIEW_TEXT_CHARS = 420
MAX_LOCAL_REVIEW_SNIPPET_CHARS = 220


def configure_environment(model: str) -> None:
    load_dotenv(ROOT / ".env", override=False)
    load_dotenv(ROOT / "docs" / ".env", override=False)
    os.environ["MODEL"] = model
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["LOCALAPPDATA"] = str(STORAGE_DIR)
    os.environ["APPDATA"] = str(STORAGE_DIR)
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("CHROMA_TELEMETRY", "False")

    try:
        from crewai.memory.storage import kickoff_task_outputs_storage as kickoff_storage
        from crewai.utilities import paths as crewai_paths
        import sqlite3

        def _workspace_db_storage_path() -> str:
            STORAGE_DIR.mkdir(parents=True, exist_ok=True)
            return str(STORAGE_DIR)

        crewai_paths.db_storage_path = _workspace_db_storage_path
        kickoff_storage.db_storage_path = _workspace_db_storage_path
        try:
            from crewai.memory.storage.kickoff_task_outputs_storage import KickoffTaskOutputsSQLiteStorage

            def _noop_init(self, db_path: str | None = None) -> None:
                self.db_path = db_path or str(STORAGE_DIR / "disabled_kickoff_outputs.db")
                self._lock_name = "sqlite:disabled"

            KickoffTaskOutputsSQLiteStorage.__init__ = _noop_init
            KickoffTaskOutputsSQLiteStorage._initialize_db = lambda self: None
            KickoffTaskOutputsSQLiteStorage.add = lambda self, *args, **kwargs: None
            KickoffTaskOutputsSQLiteStorage.update = lambda self, *args, **kwargs: None
            KickoffTaskOutputsSQLiteStorage.load = lambda self, *args, **kwargs: []
            KickoffTaskOutputsSQLiteStorage.delete_all = lambda self, *args, **kwargs: None
        except Exception:
            pass

        def _safe_initialize_db(self) -> None:
            try:
                with sqlite3.connect(self.db_path, timeout=30) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS latest_kickoff_task_outputs (
                            task_id TEXT PRIMARY KEY,
                            expected_output TEXT,
                            output JSON,
                            task_index INTEGER,
                            inputs JSON,
                            was_replayed BOOLEAN,
                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                        )
                        """
                    )
                    conn.commit()
            except sqlite3.Error as exc:
                raise kickoff_storage.DatabaseOperationError(
                    f"Database initialization error: {exc}",
                    exc,
                ) from exc

        kickoff_storage.KickoffTaskOutputsSQLiteStorage._initialize_db = _safe_initialize_db
    except Exception:
        pass


def compact_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _compact_record(record: dict[str, Any], allowed_keys: list[str]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in allowed_keys:
        if key not in record:
            continue
        value = record[key]
        if isinstance(value, str) and len(value) > 180:
            compact[key] = value[:177].rstrip() + "..."
        else:
            compact[key] = value
    return compact


def _hash_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def _shorten_text(value: str, max_chars: int) -> str:
    value = " ".join(value.split()).strip()
    if len(value) <= max_chars:
        return value
    shortened = value[: max_chars + 1].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return shortened


def _normalize_candidate_paths(preferred: Path, *fallbacks: Path) -> tuple[Path, ...]:
    return (preferred, *fallbacks)


def ensure_lab_dataset_files() -> dict[str, Path]:
    datasets = {
        "user": _ensure_jsonl_file(
            preferred=DATA_DIR / "subset_user.jsonl",
            fallbacks=_normalize_candidate_paths(
                DATA_DIR / "user_subset.json",
                DATA_DIR / "subset_user.json",
            ),
        ),
        "item": _ensure_jsonl_file(
            preferred=DATA_DIR / "subset_item.jsonl",
            fallbacks=_normalize_candidate_paths(
                DATA_DIR / "item_subset.json",
                DATA_DIR / "subset_item.json",
            ),
        ),
        "review": _ensure_jsonl_file(
            preferred=DATA_DIR / "subset_review.jsonl",
            fallbacks=_normalize_candidate_paths(
                DATA_DIR / "review_subset.json",
                DATA_DIR / "subset_review.json",
            ),
        ),
    }
    return datasets


def _ensure_jsonl_file(preferred: Path, fallbacks: Iterable[Path]) -> Path:
    if preferred.exists():
        return preferred

    for fallback in fallbacks:
        if not fallback.exists():
            continue
        records = _load_structured_records(fallback)
        preferred.parent.mkdir(parents=True, exist_ok=True)
        with preferred.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return preferred

    raise FileNotFoundError(f"Could not find a source dataset for {preferred.name}.")


@lru_cache(maxsize=None)
def _load_structured_records(path_str: str | Path) -> list[dict[str, Any]]:
    path = Path(path_str)
    if path.suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                if isinstance(record, dict):
                    records.append(record)
        return records

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array in {path}, found {type(payload).__name__}.")

    return [record for record in payload if isinstance(record, dict)]


@lru_cache(maxsize=None)
def _build_lookup(path_str: str | Path, key: str) -> dict[str, dict[str, Any]]:
    return {
        str(record[key]): record
        for record in _load_structured_records(path_str)
        if key in record
    }


def materialize_json_knowledge_file(jsonl_path: Path) -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    json_path = GENERATED_DIR / f"{jsonl_path.stem}.knowledge.json"
    records = _load_structured_records(jsonl_path)
    json_path.write_text(compact_json(records) + "\n", encoding="utf-8")
    return json_path


def materialize_review_corpus(review_jsonl: Path) -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    records = _load_structured_records(review_jsonl)
    corpus_path = GENERATED_DIR / f"{review_jsonl.stem}.rag.txt"

    chunks: list[str] = []
    for record in records[:MAX_RAG_REVIEWS]:
        review_text = str(record.get("text", "")).strip()
        if len(review_text) > MAX_REVIEW_TEXT_CHARS:
            review_text = review_text[: MAX_REVIEW_TEXT_CHARS - 3].rstrip() + "..."
        chunks.append(
            "\n".join(
                [
                    f"review_id: {record.get('review_id', '')}",
                    f"user_id: {record.get('user_id', '')}",
                    f"item_id: {record.get('item_id', '')}",
                    f"stars: {record.get('stars', '')}",
                    f"date: {record.get('date', '')}",
                    f"text: {review_text}",
                ]
            )
        )

    corpus_path.write_text("\n\n---\n\n".join(chunks) + "\n", encoding="utf-8")
    return corpus_path


def build_embedding_model() -> CohereProviderSpec | None:
    api_key = os.getenv("COHERE_API_KEY") or os.getenv("EMBEDDINGS_COHERE_API_KEY")
    if not api_key:
        return None

    return {
        "provider": "cohere",
        "config": {
            "api_key": api_key,
            "model_name": "embed-english-v3.0",
        },
    }


class UserLookupInput(BaseModel):
    user_id: str = Field(..., description="Exact Yelp user_id to retrieve from subset_user.jsonl.")


class ItemLookupInput(BaseModel):
    item_id: str = Field(..., description="Exact Yelp item_id to retrieve from subset_item.jsonl.")


class ReviewLookupV2Input(BaseModel):
    query: str = Field(default="", description="Search query describing the review evidence you want.")
    item_id: str = Field(
        default="",
        description="Exact Yelp item_id when known. If unknown, pass an empty string. Prefer the id over the business name.",
    )
    limit: int = Field(default=3, ge=1, le=10, description="Maximum number of review records to return.")


class UserKnowledgeLookupTool(BaseTool):
    name: str = "lookup_user_profile"
    description: str = (
        "Fetch the exact user profile record from subset_user.jsonl. Use this when you need grounded facts "
        "about review counts, average stars, elite status, or compliments."
    )
    args_schema: type[BaseModel] = UserLookupInput

    def _run(self, user_id: str) -> str:
        dataset_paths = ensure_lab_dataset_files()
        users = _build_lookup(dataset_paths["user"], "user_id")
        record = users.get(user_id)
        if record is None:
            return f"No user found for user_id={user_id!r}."
        compact = _compact_record(
            record,
            [
                "user_id",
                "name",
                "review_count",
                "yelping_since",
                "fans",
                "average_stars",
                "elite",
                "useful",
                "funny",
                "cool",
            ],
        )
        return compact_json(compact)


class ItemKnowledgeLookupTool(BaseTool):
    name: str = "lookup_item_profile"
    description: str = (
        "Fetch the exact business record from subset_item.jsonl. Use this when you need grounded business metadata, "
        "attributes, categories, or base rating context."
    )
    args_schema: type[BaseModel] = ItemLookupInput

    def _run(self, item_id: str) -> str:
        dataset_paths = ensure_lab_dataset_files()
        items = _build_lookup(dataset_paths["item"], "item_id")
        record = items.get(item_id)
        if record is None:
            return f"No item found for item_id={item_id!r}."
        compact = _compact_record(
            record,
            [
                "item_id",
                "name",
                "city",
                "state",
                "stars",
                "review_count",
                "is_open",
                "categories",
                "attributes",
                "hours",
            ],
        )
        return compact_json(compact)


def _tokenize(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", value.lower()) if len(token) > 2}


def _extract_categories(record: dict[str, Any]) -> list[str]:
    return [part.strip() for part in str(record.get("categories", "")).split(",") if part.strip()]


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@lru_cache(maxsize=1)
def _build_item_name_lookup() -> dict[str, str]:
    dataset_paths = ensure_lab_dataset_files()
    lookup: dict[str, str] = {}
    for record in _load_structured_records(dataset_paths["item"]):
        item_id = str(record.get("item_id", "")).strip()
        name = str(record.get("name", "")).strip().lower()
        if item_id and name and name not in lookup:
            lookup[name] = item_id
    return lookup


def _normalize_lookup_value(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _resolve_item_identifier(item_value: str | None) -> str | None:
    normalized = _normalize_lookup_value(item_value)
    if normalized is None:
        return None

    dataset_paths = ensure_lab_dataset_files()
    items = _build_lookup(dataset_paths["item"], "item_id")
    if normalized in items:
        return normalized

    return _build_item_name_lookup().get(normalized.lower(), normalized)


class ReviewLookupV2Tool(BaseTool):
    name: str = "review_lookup_v2"
    description: str = (
        "Review lookup over subset_review.jsonl. Requires query, item_id, and limit. Use the exact Yelp item_id when known, "
        "or pass an empty string if unknown. This tool returns compact review snippets."
    )
    args_schema: type[BaseModel] = ReviewLookupV2Input

    def _run(
        self,
        query: str,
        item_id: str = "",
        limit: int = 3,
    ) -> str:
        dataset_paths = ensure_lab_dataset_files()
        reviews = _load_structured_records(dataset_paths["review"])
        limit = max(1, min(limit, 10))
        item_id = _resolve_item_identifier(item_id)
        query_tokens = _tokenize(query)
        if not query_tokens and not item_id:
            return "No review lookup performed because both query and item_id were empty."
        ranked: list[tuple[float, dict[str, Any]]] = []

        for review in reviews:
            score = 0.0
            if item_id and str(review.get("item_id")) == item_id:
                score += 3.5

            review_tokens = _tokenize(str(review.get("text", "")))
            overlap = len(query_tokens & review_tokens)
            score += overlap

            if not item_id and overlap == 0:
                continue
            if item_id and score == 0:
                continue

            ranked.append((score, review))

        ranked.sort(
            key=lambda item: (
                item[0],
                str(item[1].get("date", "")),
            ),
            reverse=True,
        )

        trimmed: list[dict[str, Any]] = []
        for _, review in ranked[:limit]:
            trimmed.append(
                {
                    "review_id": review.get("review_id"),
                    "user_id": review.get("user_id"),
                    "item_id": review.get("item_id"),
                    "stars": review.get("stars"),
                    "date": review.get("date"),
                    "text": str(review.get("text", ""))[:MAX_LOCAL_REVIEW_SNIPPET_CHARS],
                }
            )

        if not trimmed:
            return "No matching reviews found in subset_review.jsonl."
        return compact_json(trimmed)


def build_knowledge_sources(embedder: CohereProviderSpec | None) -> list:
    return []


def build_review_rag_tool(embedder: CohereProviderSpec | None) -> BaseTool:
    if embedder is None:
        return ReviewLookupV2Tool()

    dataset_paths = ensure_lab_dataset_files()
    review_corpus_path = materialize_review_corpus(dataset_paths["review"])
    collection_hash = _hash_text(
        f"{review_corpus_path}:{review_corpus_path.stat().st_mtime_ns}:embed-english-v3.0"
    )
    collection_name = f"subset_review_rag_{collection_hash}"
    marker_path = GENERATED_DIR / f"{collection_name}.indexed"

    try:
        rag_tool = RagTool(
            name="semantic_review_rag",
            description=(
                "Semantic retriever over subset_review.jsonl. Use the exact Yelp item_id, not the business name. Keep results short and request at most 3 reviews."
            ),
            collection_name=collection_name,
            limit=3,
            similarity_threshold=0.35,
            config={"embedding_model": embedder},
        )
    except Exception:
        return ReviewLookupV2Tool()

    if not marker_path.exists():
        try:
            rag_tool.add(path=str(review_corpus_path), data_type="file")
            marker_path.write_text("indexed\n", encoding="utf-8")
        except Exception:
            return ReviewLookupV2Tool()

    return rag_tool


def build_external_research_tools() -> list[BaseTool]:
    if os.getenv("SERPER_API_KEY"):
        return [SerperDevTool()]
    return []


def build_agents(
    model: str,
    verbose: bool,
    knowledge_sources: list,
    review_rag_tool: BaseTool,
) -> dict[str, Agent]:
    exact_profile_tools: list[BaseTool] = [
        UserKnowledgeLookupTool(),
        ItemKnowledgeLookupTool(),
        review_rag_tool,
    ]
    research_tools = [*exact_profile_tools, *build_external_research_tools()]

    agents = {
        "researcher": Agent(
            role="Knowledge Grounding Researcher",
            goal=(
                "Use subset_user.jsonl and subset_item.jsonl as knowledge, then retrieve the strongest user and business "
                "evidence for user {user_id} and item {item_id}."
            ),
            backstory=(
                "You are meticulous about grounding. You pull exact profile facts from the lab datasets first, then "
                "support them with semantic review evidence when behavior or tone needs more context."
            ),
            tools=exact_profile_tools,
            knowledge_sources=knowledge_sources,
            verbose=verbose,
            allow_delegation=False,
            max_iter=6,
            llm=model,
        ),
        "eda_strategist": Agent(
            role="Exploratory Data Analysis Strategist",
            goal=(
                "Turn the retrieved user, item, and review evidence into EDA insights: base rates, mismatches, "
                "confidence risks, and the most predictive features for the target pair."
            ),
            backstory=(
                "You strengthen the crew by thinking like an analyst before thinking like a writer. You look for "
                "distribution clues, sparse data warnings, and contradictions that could affect the rating."
            ),
            knowledge_sources=knowledge_sources,
            verbose=verbose,
            allow_delegation=False,
            max_iter=4,
            llm=model,
        ),
        "prediction_analyst": Agent(
            role="Rating Prediction Analyst",
            goal=(
                "Predict a realistic Yelp-style star rating and draft a short grounded review for the current user-item pair."
            ),
            backstory=(
                "You synthesize evidence carefully. You do not invent facts, and you keep the final review aligned with "
                "the retrieved tone, the business context, and the EDA findings."
            ),
            knowledge_sources=knowledge_sources,
            verbose=verbose,
            allow_delegation=False,
            max_iter=4,
            llm=model,
        ),
        "quality_reviewer": Agent(
            role="Output Quality Reviewer",
            goal=(
                "Return valid final JSON and catch unsupported claims, weak evidence links, or EDA summaries that are too vague."
            ),
            backstory=(
                "You are the final guardrail. You verify that the answer is grounded in the lab knowledge and the semantic "
                "review retrieval rather than generic restaurant language."
            ),
            knowledge_sources=knowledge_sources,
            verbose=verbose,
            allow_delegation=False,
            max_iter=3,
            llm=model,
        ),
        "orchestrator": Agent(
            role="Collaborative EDA Orchestrator",
            goal=(
                "Own one shared task for user {user_id} and item {item_id}, delegating retrieval, EDA, and validation "
                "work so the crew behaves like a stronger collaborative team."
            ),
            backstory=(
                "You lead Pattern 2 collaborative crews. You know when to ask for exact facts, when to ask for semantic "
                "review retrieval, and how to merge specialist outputs into one clean answer."
            ),
            tools=research_tools,
            knowledge_sources=knowledge_sources,
            verbose=verbose,
            allow_delegation=True,
            max_iter=6,
            llm=model,
        ),
    }

    if build_external_research_tools():
        agents["internet_researcher"] = Agent(
            role="Internet Research Scout",
            goal=(
                "Collect supplemental public information about review modeling, recommendation heuristics, and CrewAI "
                "agent design when external context is genuinely helpful."
            ),
            backstory=(
                "You only add internet context when it sharpens the crew's reasoning. Local dataset evidence remains the "
                "primary source of truth."
            ),
            tools=build_external_research_tools(),
            knowledge_sources=knowledge_sources,
            verbose=verbose,
            allow_delegation=False,
            max_iter=4,
            llm=model,
        )

    return agents


def build_baseline_sequential_crew(model: str, verbose: bool = True) -> Crew:
    configure_environment(model)
    embedder = build_embedding_model()
    knowledge_sources = build_knowledge_sources(embedder)
    review_rag_tool = build_review_rag_tool(embedder)
    agents = build_agents(model, verbose, knowledge_sources, review_rag_tool)

    tasks = [
        Task(
            description=(
                "Retrieve grounded evidence for user {user_id} and item {item_id}. Use subset_user.jsonl and "
                "subset_item.jsonl as knowledge, and use the review lookup tool on subset_review.jsonl to find "
                "similar reviews or historically relevant examples. When calling any review tool, pass the exact "
                "item_id value `{item_id}`. If a tool field is unknown, pass an empty string instead of omitting it. Keep tool usage compact: 1 user lookup, 1 item "
                "lookup, and at most 2 review retrieval calls with at most 3 results each. Return a memo under 180 words."
            ),
            expected_output="A compact grounded memo under 180 words with user signals, item signals, and up to 3 review clues.",
            agent=agents["researcher"],
        ),
        Task(
            description=(
                "Perform exploratory data analysis on the grounded memo. Identify rating tendencies, sparse-data risks, "
                "feature matches or mismatches, and the most predictive signals for this user-item pair. Keep the answer under 120 words."
            ),
            expected_output="A concise EDA brief under 120 words with key signals, caveats, and confidence notes.",
            agent=agents["eda_strategist"],
        ),
        Task(
            description=(
                "Produce the final strict JSON with keys 'stars', 'review', and 'eda_summary'. The answer must be faithful "
                "to the retrieved evidence and EDA brief. Keep 'review' under 45 words and 'eda_summary' under 35 words."
            ),
            expected_output=(
                '{"stars": 4.0, "review": "Grounded short review.", "eda_summary": "Concise explanation of the predictive signals."}'
            ),
            agent=agents["prediction_analyst"],
            output_file=str(DEFAULT_OUTPUT),
        ),
    ]

    return Crew(
        name="lab_agent_review_baseline",
        agents=[agents["researcher"], agents["eda_strategist"], agents["prediction_analyst"]],
        tasks=tasks,
        process=Process.sequential,
        verbose=verbose,
        memory=False,
        embedder=embedder,
        knowledge_sources=knowledge_sources,
    )


def build_collaborative_sequential_crew(model: str, verbose: bool = True) -> Crew:
    configure_environment(model)
    embedder = build_embedding_model()
    knowledge_sources = build_knowledge_sources(embedder)
    review_rag_tool = build_review_rag_tool(embedder)
    agents = build_agents(model, verbose, knowledge_sources, review_rag_tool)

    collaborative_task = Task(
        description=(
            "Pattern 2: Collaborative Single Task. Solve the full rating-prediction job for user {user_id} and item "
            "{item_id}. Use subset_user.jsonl and subset_item.jsonl as the crew's knowledge base, use "
            "the review lookup tool for retrieval from subset_review.jsonl, and delegate work to the research, "
            "EDA, prediction, quality, and optional internet research specialists as needed. Always pass the exact "
            "Yelp item_id `{item_id}` to tools. If a tool field is unknown, pass an empty string instead of omitting it. Keep all intermediate "
            "work compact. Return strict JSON with keys 'stars', 'review', and 'eda_summary'. Keep 'review' under "
            "45 words and 'eda_summary' under 35 words."
        ),
        expected_output=(
            '{"stars": 4.0, "review": "Grounded short review.", "eda_summary": "Concise explanation of the predictive signals."}'
        ),
        agent=agents["orchestrator"],
        output_file=str(DEFAULT_OUTPUT),
    )

    ordered_agents = [
        agents["orchestrator"],
        agents["researcher"],
        agents["eda_strategist"],
        agents["prediction_analyst"],
        agents["quality_reviewer"],
    ]
    if "internet_researcher" in agents:
        ordered_agents.append(agents["internet_researcher"])

    return Crew(
        name="lab_agent_review_collaborative",
        agents=ordered_agents,
        tasks=[collaborative_task],
        process=Process.sequential,
        verbose=verbose,
        memory=False,
        embedder=embedder,
        knowledge_sources=knowledge_sources,
    )


def build_hierarchical_crew(model: str, verbose: bool = True) -> Crew:
    configure_environment(model)
    embedder = build_embedding_model()
    knowledge_sources = build_knowledge_sources(embedder)
    review_rag_tool = build_review_rag_tool(embedder)
    agents = build_agents(model, verbose, knowledge_sources, review_rag_tool)

    ordered_agents = [
        agents["researcher"],
        agents["eda_strategist"],
        agents["prediction_analyst"],
        agents["quality_reviewer"],
    ]
    if "internet_researcher" in agents:
        ordered_agents.append(agents["internet_researcher"])

    tasks = [
        Task(
            description=(
                "Retrieve exact user and business facts, then use semantic_review_rag to surface the most relevant review "
                "evidence for user {user_id} and item {item_id}. Keep the memo tightly grounded in subset_user.jsonl, "
                "subset_item.jsonl, and subset_review.jsonl. Use the review lookup tool for compact retrieval. Always pass the exact item_id `{item_id}` to tools. "
                "If a tool field is unknown, pass an empty string instead of omitting it. "
                "Keep it under 160 words and use at most 3 review results."
            ),
            expected_output="A grounded retrieval brief under 160 words with exact facts plus up to 3 review clues.",
            agent=agents["researcher"],
        ),
        Task(
            description=(
                "Analyze the retrieval brief with an EDA mindset. Identify likely rating drivers, user-business fit, "
                "outliers, uncertainty, and the expected rating range. Keep the answer under 120 words."
            ),
            expected_output="An EDA memo under 120 words with signal ranking and confidence notes.",
            agent=agents["eda_strategist"],
        ),
        Task(
            description=(
                "Draft a prediction using the retrieval brief and EDA memo. Return JSON with keys 'stars' and 'review'. Keep the review under 45 words."
            ),
            expected_output='{"stars": 4.0, "review": "Grounded short review."}',
            agent=agents["prediction_analyst"],
        ),
        Task(
            description=(
                "Validate the prediction and expand it into the final strict JSON with keys 'stars', 'review', and "
                "'eda_summary'. Correct weak grounding or malformed formatting if necessary. Keep 'eda_summary' under 35 words."
            ),
            expected_output=(
                '{"stars": 4.0, "review": "Grounded short review.", "eda_summary": "Concise explanation of the predictive signals."}'
            ),
            agent=agents["quality_reviewer"],
            output_file=str(DEFAULT_OUTPUT),
        ),
    ]

    return Crew(
        name="lab_agent_review_hierarchical",
        agents=ordered_agents,
        tasks=tasks,
        process=Process.hierarchical,
        manager_llm=model,
        verbose=verbose,
        memory=False,
        embedder=embedder,
        knowledge_sources=knowledge_sources,
    )


def build_selected_crew(crew_name: str, model: str, verbose: bool) -> Crew:
    if crew_name == "collaborative":
        return build_collaborative_sequential_crew(model=model, verbose=verbose)
    if crew_name == "hierarchical":
        return build_hierarchical_crew(model=model, verbose=verbose)
    return build_baseline_sequential_crew(model=model, verbose=verbose)


def normalize_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        payload = result
    elif isinstance(result, str):
        payload = _extract_json_object(result)
    elif hasattr(result, "raw"):
        payload = _extract_json_object(getattr(result, "raw"))
    else:
        raise TypeError(f"Unsupported result type: {type(result).__name__}")

    review = payload.get("review", "")
    if isinstance(review, list):
        payload["review"] = " ".join(str(part).strip() for part in review if str(part).strip())
    return payload


def generate_deterministic_payload(user_id: str, item_id: str) -> dict[str, Any]:
    dataset_paths = ensure_lab_dataset_files()
    users = _build_lookup(dataset_paths["user"], "user_id")
    items = _build_lookup(dataset_paths["item"], "item_id")

    user = users.get(user_id, {})
    item = items.get(item_id, {})
    if not item:
        return {
            "stars": 3.5,
            "review": "The target business could not be matched.",
            "eda_summary": "Missing item record; returned fallback",
        }

    all_reviews = _load_structured_records(dataset_paths["review"])
    user_reviews = [record for record in all_reviews if str(record.get("user_id")) == user_id]
    user_reviews.sort(key=lambda record: str(record.get("date", "")), reverse=True)

    user_avg = _safe_float(user.get("average_stars"), 0.0)
    if user_avg <= 0 and user_reviews:
        user_avg = sum(_safe_float(record.get("stars"), 0.0) for record in user_reviews) / len(user_reviews)
    if user_avg <= 0:
        user_avg = 3.75

    recent_reviews = user_reviews[: min(10, len(user_reviews))]
    recent_avg = (
        sum(_safe_float(record.get("stars"), user_avg) for record in recent_reviews) / len(recent_reviews)
        if recent_reviews
        else user_avg
    )

    item_avg = _safe_float(item.get("stars"), 3.5)
    item_review_count = int(_safe_float(item.get("review_count"), 0))
    target_categories = set(_extract_categories(item))

    overlap_scores: list[float] = []
    strong_match_count = 0
    items_by_id = items
    for review in user_reviews:
        past_item = items_by_id.get(str(review.get("item_id")))
        if not past_item:
            continue
        past_categories = set(_extract_categories(past_item))
        overlap = target_categories & past_categories
        if overlap:
            overlap_scores.append(_safe_float(review.get("stars"), user_avg))
            if len(overlap) >= 1:
                strong_match_count += 1

    overlap_avg = sum(overlap_scores) / len(overlap_scores) if overlap_scores else user_avg

    predicted = (
        (0.35 * user_avg)
        + (0.25 * item_avg)
        + (0.25 * overlap_avg)
        + (0.15 * recent_avg)
    )

    if item_review_count < 50:
        predicted += 0.05
    elif item_review_count > 200:
        predicted -= 0.05

    if strong_match_count >= 3:
        predicted += 0.1
    elif strong_match_count == 0:
        predicted -= 0.15

    predicted = round(min(5.0, max(1.0, predicted)), 1)

    item_name = str(item.get("name", "This business")).strip() or "This business"
    lead_category = _extract_categories(item)[0] if _extract_categories(item) else "venue"
    attributes = item.get("attributes") or {}
    vibe_parts: list[str] = []
    if isinstance(attributes, dict):
        if str(attributes.get("OutdoorSeating", "")).lower() == "true":
            vibe_parts.append("outdoor seating")
        if str(attributes.get("HappyHour", "")).lower() == "true":
            vibe_parts.append("happy hour")
        noise = str(attributes.get("NoiseLevel", "")).lower()
        if "loud" in noise:
            vibe_parts.append("a loud atmosphere")
        if "casual" in str(attributes.get("Ambience", "")).lower():
            vibe_parts.append("a casual vibe")

    if predicted >= 4.2:
        review = f"{item_name} looks like a strong fit for me. The {lead_category.lower()} setting and {vibe_parts[0] if vibe_parts else 'overall atmosphere'} should work well."
    elif predicted >= 3.6:
        review = f"{item_name} seems like a decent fit for me. The {lead_category.lower()} vibe works, but the lower house rating keeps it short of excellent."
    elif predicted >= 2.8:
        review = f"{item_name} feels mixed for me. Some of the {lead_category.lower()} traits fit, but the overall signals are only average."
    else:
        review = f"{item_name} does not look like a strong fit for me. The overall quality signals are weaker than my usual favorites."

    if len(review.split()) > 45:
        review = _shorten_text(review, 45)

    fit_label = "strong" if predicted >= 4.2 else "mixed-positive" if predicted >= 3.6 else "mixed" if predicted >= 2.8 else "weak"
    eda_summary = (
        f"user {user_avg:.2f}, recent {recent_avg:.2f}, item {item_avg:.1f}, "
        f"category overlap {overlap_avg:.2f}, fit {fit_label}"
    )
    eda_summary = _shorten_text(eda_summary, 35)

    return {
        "stars": predicted,
        "review": review,
        "eda_summary": eda_summary,
    }


def build_fallback_payload(user_id: str, item_id: str) -> dict[str, Any]:
    dataset_paths = ensure_lab_dataset_files()
    user = _build_lookup(dataset_paths["user"], "user_id").get(user_id, {})
    item = _build_lookup(dataset_paths["item"], "item_id").get(item_id, {})
    user_reviews = [
        record
        for record in _load_structured_records(dataset_paths["review"])
        if str(record.get("user_id")) == user_id
    ]
    user_reviews.sort(key=lambda record: str(record.get("date", "")), reverse=True)

    user_avg = float(user.get("average_stars", 3.75) or 3.75)
    item_avg = float(item.get("stars", 3.5) or 3.5)
    predicted = round(min(5.0, max(1.0, (user_avg * 0.6) + (item_avg * 0.4))), 1)

    item_name = str(item.get("name", "this place")).strip() or "this place"
    categories = [part.strip() for part in str(item.get("categories", "")).split(",") if part.strip()]
    primary_category = categories[0] if categories else "venue"

    recent_text = ""
    for review in user_reviews[:5]:
        text = str(review.get("text", "")).strip()
        if text:
            recent_text = text
            break

    tone = "balanced"
    if predicted >= 4.3:
        tone = "positive"
    elif predicted <= 2.7:
        tone = "critical"

    if tone == "positive":
        review = f"{item_name} looks like a strong fit for me."
    elif tone == "critical":
        review = f"{item_name} does not look like a good fit for me."
    else:
        review = f"{item_name} seems like a mixed fit for me."

    if recent_text:
        recent_tokens = _tokenize(recent_text)
        if any(token in recent_tokens for token in {"service", "wait", "crowd", "dirty", "noise", "loud"}):
            if tone == "positive":
                review = f"{item_name} looks like a strong fit, despite some atmosphere risk."
            elif tone == "critical":
                review = f"{item_name} looks risky for my usual preferences."
            else:
                review = f"{item_name} feels mixed, especially on atmosphere."

    eda_summary = f"user {user_avg:.2f}, item {item_avg:.1f}, fit {'strong' if predicted >= 4.3 else 'mixed' if predicted >= 3.0 else 'weak'}"

    return {
        "stars": predicted,
        "review": _shorten_text(review, 45),
        "eda_summary": _shorten_text(eda_summary, 35),
    }


def finalize_payload(payload: dict[str, Any] | None, user_id: str, item_id: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return build_fallback_payload(user_id, item_id)

    if {"name", "parameters"} <= set(payload.keys()) and "stars" not in payload:
        return build_fallback_payload(user_id, item_id)

    fallback = build_fallback_payload(user_id, item_id)

    try:
        stars = float(payload.get("stars"))
    except (TypeError, ValueError):
        stars = fallback["stars"]

    stars = round(min(5.0, max(1.0, stars)), 1)

    review = str(payload.get("review", "")).strip()
    if not review:
        review = fallback["review"]

    eda_summary = str(payload.get("eda_summary", "")).strip()
    if not eda_summary:
        eda_summary = fallback["eda_summary"]

    return {
        "stars": stars,
        "review": review,
        "eda_summary": eda_summary,
    }


def _extract_json_object(raw: str) -> dict[str, Any]:
    candidates: list[str] = [raw]

    fenced_match = re.search(r"\{[\s\S]*\}", raw)
    if fenced_match:
        candidates.append(fenced_match.group(0))

    for candidate in candidates:
        loaded = _try_load_json_object(candidate)
        if loaded is not None:
            return loaded

    repaired = _repair_result_payload(raw)
    if repaired is not None:
        return repaired

    _write_debug_response(raw)
    raise ValueError("Could not extract a JSON object from the crew result. Raw output saved to .crewai_storage/generated/last_raw_response.txt")


def _try_load_json_object(candidate: str) -> dict[str, Any] | None:
    try:
        loaded = json.loads(candidate)
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        pass

    try:
        loaded = ast.literal_eval(candidate)
        if isinstance(loaded, dict):
            return loaded
    except (SyntaxError, ValueError):
        pass

    return None


def _repair_result_payload(raw: str) -> dict[str, Any] | None:
    text = raw.replace("\r", " ").replace("\n", " ")

    stars_match = re.search(r'"?stars"?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)', text, flags=re.IGNORECASE)
    review_match = re.search(r'"?review"?\s*[:=]\s*"((?:[^"\\]|\\.)*)"', text, flags=re.IGNORECASE)
    eda_match = re.search(r'"?eda_summary"?\s*[:=]\s*"((?:[^"\\]|\\.)*)"', text, flags=re.IGNORECASE)

    if not stars_match or not review_match:
        return None

    payload: dict[str, Any] = {
        "stars": float(stars_match.group(1)),
        "review": bytes(review_match.group(1), "utf-8").decode("unicode_escape"),
        "eda_summary": "",
    }

    if eda_match:
        payload["eda_summary"] = bytes(eda_match.group(1), "utf-8").decode("unicode_escape")
    else:
        payload["eda_summary"] = "Grounded summary unavailable due to malformed model JSON."

    return payload


def _write_debug_response(raw: str) -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    (GENERATED_DIR / "last_raw_response.txt").write_text(raw, encoding="utf-8")


def write_report(payload: dict[str, Any], output_path: Path) -> None:
    output_path.write_text(compact_json(payload) + "\n", encoding="utf-8")


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--engine",
        choices=("deterministic", "crew"),
        default="deterministic",
        help="Use the fast deterministic predictor or the slower CrewAI path.",
    )
    parser.add_argument(
        "--crew",
        choices=("baseline", "collaborative", "hierarchical"),
        default="baseline",
        help="Which crew topology to run.",
    )
    parser.add_argument("--user-id", default=DEFAULT_USER_ID, help="Target Yelp user_id.")
    parser.add_argument("--item-id", default=DEFAULT_ITEM_ID, help="Target Yelp item_id/business_id.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM identifier passed to CrewAI.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Where to write the final JSON report.")
    parser.add_argument("--quiet", action="store_true", help="Reduce CrewAI console logging.")
    return parser
