# CrewAI Lab: Knowledge + Review RAG

This project is now aligned to the lab brief in a runtime-stable way:

- `subset_user.jsonl` and `subset_item.jsonl` are used through dedicated grounding tools
- `subset_review.jsonl` is queried through semantic RAG when available, with a local fallback retriever otherwise
- the professor's cached Chroma collections are reused automatically when `Rag_Crew_Profiler/chroma.sqlite3` is available
- the default embedding setup is `BAAI/bge-small-en-v1.5`, matching the professor-provided vector index
- the crew includes stronger EDA-focused agents
- all requested crew organizations are included:
  - baseline `Process.sequential`
  - collaborative single-task `Process.sequential`
  - `Process.hierarchical`
- bonus `CrewAI Flow` orchestration in [agent_flow.py](C:/Users/vansh/OneDrive/Desktop/AgentReview/agent_flow.py)

## Core Design

The shared implementation lives in [lab_project.py](C:/Users/vansh/OneDrive/Desktop/AgentReview/lab_project.py).

Data flow:

1. `subset_user.jsonl` and `subset_item.jsonl` are accessed through exact lookup tools for stable grounding.
2. When the professor's Chroma cache is present, professor-aligned semantic JSON search tools attach to the existing user/item/review collections instead of rebuilding them.
3. `subset_review.jsonl` is converted into a smaller review corpus and indexed by `semantic_review_rag` when embeddings are available.
4. Agents combine:
   - exact user/item lookup tools for reliable grounding
   - professor-aligned semantic collection tools when the cached index is present
   - semantic review retrieval for analogous review evidence
   - local review search as an automatic fallback if Chroma is unavailable
5. Final output is written to [report.json](C:/Users/vansh/OneDrive/Desktop/AgentReview/report.json).

## Agents

- `Knowledge Grounding Researcher`
- `Exploratory Data Analysis Strategist`
- `Rating Prediction Analyst`
- `Output Quality Reviewer`
- `Collaborative EDA Orchestrator`
- `Internet Research Scout` when `SERPER_API_KEY` is configured

Extra EDA knowledge lives in:

- [docs/knowledge/eda_playbook.txt](C:/Users/vansh/OneDrive/Desktop/AgentReview/docs/knowledge/eda_playbook.txt)
- [docs/knowledge/crewai_coding_skills.txt](C:/Users/vansh/OneDrive/Desktop/AgentReview/docs/knowledge/crewai_coding_skills.txt)
- [docs/knowledge/eda_rating_prediction_checklist.txt](C:/Users/vansh/OneDrive/Desktop/AgentReview/docs/knowledge/eda_rating_prediction_checklist.txt)

## Environment

Default local run with Ollama:

```bash
ollama pull llama3.2:1b
python main.py --quiet
```

Optional:

```bash
COHERE_API_KEY=your_cohere_key
GROQ_API_KEY=your_groq_key
NVIDIA_API_KEY=your_nvidia_key
NVIDIA_API_BASE=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL_NAME=minimaxai/minimax-m2.7
LLM_PROVIDER=ollama
SERPER_API_KEY=your_serper_key
```

Default embeddings now use the professor-aligned local sentence-transformer setup:

```python
embedding_model = {
    "provider": "sentence-transformer",
    "config": {
        "model_name": "BAAI/bge-small-en-v1.5"
    }
}
```

Set `AGENT_REVIEW_EMBEDDINGS=cohere` if you want to switch semantic review indexing back to Cohere embeddings.

Set `AGENT_REVIEW_CHROMA_DIR` if the professor's `chroma.sqlite3` lives somewhere other than `%LOCALAPPDATA%\Rag_Crew_Profiler`.

The default model in this repo is `ollama/llama3.2:1b`, and `LLM_PROVIDER=nvidia` will automatically map the run to the Nvidia Build API settings above.

You can place keys in `.env` or [docs/.env](C:/Users/vansh/OneDrive/Desktop/AgentReview/docs/.env).

This repo defaults to the tool-based grounding path because it is more reliable in local Windows setups than CrewAI knowledge indexing.

## Run

```bash
python main.py --crew baseline
python crew.py --crew collaborative
python crew.py --crew hierarchical
```

Bonus Flow run:

```bash
python agent_flow.py --user-id nnImk681KaRqUVHlSfZjGQ --item-id -7GjicSH_rM8JeZGCXGcUg
```

Example:

```bash
python main.py --crew collaborative --user-id nnImk681KaRqUVHlSfZjGQ --item-id -7GjicSH_rM8JeZGCXGcUg
```

Useful flags:

- `--user-id`
- `--item-id`
- `--model`
- `--output`
- `--quiet`

## Output

```json
{
  "stars": 4.0,
  "review": "Grounded short review.",
  "eda_summary": "Concise explanation of the predictive signals."
}
```

## References

- [CrewAI Quickstart](https://docs.crewai.com/en/quickstart)
- [CrewAI Tools Overview](https://docs.crewai.com/tools/overview)
- [CrewAI RAG Tool](https://docs.crewai.com/ar/tools/ai-ml/ragtool)

