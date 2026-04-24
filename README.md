## AgentReview Crew: Yelp Rating Prediction with CrewAI

This project is a multi-agent CrewAI system built to predict Yelp-style star ratings and generate short review text using user history, business metadata, and review evidence. It includes multiple crew execution patterns, stronger custom agents, index reuse, EDA-oriented knowledge, and CrewAI Flow integration.

The project is designed around a grounded prediction workflow using:

- `subset_user.jsonl`
- `subset_item.jsonl`
- `subset_review.jsonl`

The final result is written in structured JSON format for submission and evaluation.

## Installation

Make sure you have:

- Python `>=3.10`
- `uv` installed

Install `uv` if needed:

```bash
pip install uv
```

Then install the project dependencies:

```bash
uv sync
```

## Configuration

This project supports both local and API-based execution.

### Environment Variables

Create a local `.env` file in the project root and add the keys you want to use.

Example:

```env
LLM_PROVIDER=nvidia
NVIDIA_API_KEY=your_key_here
NVIDIA_MODEL_NAME=minimaxai/minimax-m2.7
NVIDIA_API_BASE=https://integrate.api.nvidia.com/v1
COHERE_API_KEY=your_key_here
SERPER_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
OTEL_SDK_DISABLED=true
```

Notes:

- `.env` is only for local use and should not be pushed to GitHub
- `SERPER_API_KEY` is optional and only needed for the internet research agent
- if no external provider is available, the project can still fall back to deterministic behavior

## Running the Project

### Deterministic Mode

```bash
python main.py --engine deterministic --quiet
```

### Baseline Sequential Crew

```bash
python main.py --engine crew --crew baseline --quiet
```

### Collaborative Sequential Crew

```bash
python main.py --engine crew --crew collaborative --quiet
```

### Hierarchical Crew

```bash
python main.py --engine crew --crew hierarchical --quiet
```

### Run Through Crew Entry Point

```bash
python crew.py --crew collaborative --quiet
python crew.py --crew hierarchical --quiet
```

### Run with Flow

```bash
python agent_flow.py --quiet --stages baseline,collaborative,hierarchical
```

### Example with Explicit IDs

```bash
python main.py --engine crew --crew hierarchical --user-id nnImk681KaRqUVHlSfZjGQ --item-id -7GjicSH_rM8JeZGCXGcUg --quiet
```

## Output Format

The project writes output to `report.json`.

Current submission format:

```json
[
  {
    "user_id": "nnImk681KaRqUVHlSfZjGQ",
    "item_id": "-7GjicSH_rM8JeZGCXGcUg",
    "crew_mode": "hierarchical",
    "predicted": {
      "stars": 3.8,
      "review": "Grounded review text."
    }
  }
]
```

## Project Deliverables and Requirements

This repository covers all required lab items.

### 1. Index-Reuse Mechanism Integration

Implemented in [lab_project.py](C:/Users/vansh/OneDrive/Desktop/AgentReview/lab_project.py).

The system checks for an existing local Chroma database and reuses available collections instead of rebuilding everything from scratch. This improves efficiency when cached vector data is already present.

### 2. Crew with `Process.sequential` Pattern 2: Collaborative Single Task

Implemented in [lab_project.py](C:/Users/vansh/OneDrive/Desktop/AgentReview/lab_project.py).

The collaborative configuration uses a shared-task sequential pattern where multiple agents contribute to one prediction outcome by combining:

- grounding
- EDA reasoning
- prediction
- validation

### 3. Crew with `Process.hierarchical` (Manager Agent)

Implemented in [lab_project.py](C:/Users/vansh/OneDrive/Desktop/AgentReview/lab_project.py).

The hierarchical crew uses `Process.hierarchical` and delegates sub-tasks through a manager-style workflow for:

- evidence retrieval
- analysis
- prediction drafting
- final checking

### 4. Create New Agents to Make the Crew Stronger

The following stronger agents were added:

- `Knowledge Grounding Researcher`
- `Exploratory Data Analysis Strategist`
- `Rating Prediction Analyst`
- `Output Quality Reviewer`
- `Collaborative EDA Orchestrator`
- `Internet Research Scout` when `SERPER_API_KEY` is available

These improve task specialization and make the crew more structured and reliable.

## Bonus Objectives

### Exploratory Data Analysis (EDA)

EDA knowledge was added through dedicated text knowledge files:

- `docs/knowledge/eda_playbook.txt`
- `docs/knowledge/crewai_coding_skills.txt`
- `docs/knowledge/eda_rating_prediction_checklist.txt`

This supports agents in reasoning about:

- user behavior patterns
- business fit
- predictive signals
- uncertainty
- concise EDA summaries

### CrewAI Flow Integration

Implemented in [agent_flow.py](C:/Users/vansh/OneDrive/Desktop/AgentReview/agent_flow.py).

The Flow:

- runs baseline, collaborative, and hierarchical stages
- stores intermediate outputs in `report_artifacts`
- creates a final output in `report.json`
- generates a `manifest.json` file for tracking stage execution

## Project Structure

- [lab_project.py](C:/Users/vansh/OneDrive/Desktop/AgentReview/lab_project.py)
  Core project logic including tools, agents, crew builders, payload formatting, and output writing.

- [main.py](C:/Users/vansh/OneDrive/Desktop/AgentReview/main.py)
  Main entry point for deterministic and crew-based runs.

- [crew.py](C:/Users/vansh/OneDrive/Desktop/AgentReview/crew.py)
  Crew execution entry point for collaborative and hierarchical modes.

- [agent_flow.py](C:/Users/vansh/OneDrive/Desktop/AgentReview/agent_flow.py)
  Flow orchestration for the bonus requirement.

- `data/`
  Contains the structured dataset files used for grounding and retrieval.

- `docs/knowledge/`
  Contains EDA-related and supporting knowledge documents.

- `report_artifacts/`
  Contains Flow stage outputs:
  - `baseline.json`
  - `collaborative.json`
  - `hierarchical.json`
  - `final.json`
  - `manifest.json`

- `submission_outputs/`
  Contains saved `.txt` outputs from the project runs used for submission.

## Stored Submission Outputs

The following files are included for result tracking:

- `submission_outputs/01_deterministic.txt`
- `submission_outputs/02_baseline_crew.txt`
- `submission_outputs/03_collaborative_crew.txt`
- `submission_outputs/04_hierarchical_crew.txt`
- `submission_outputs/05_flow.txt`
- `submission_outputs/06_report_json.txt`
- `submission_outputs/07_manifest.txt`

## Current Limitations

- Local or weaker models may produce shorter or less specific review text
- external provider availability can affect semantic retrieval performance
- exact rating quality depends on the available evidence for the selected `user_id` and `item_id`
- some runs may fall back to deterministic behavior when network-based services are unavailable

## Summary

This lab submission combines:

- index reuse
- exact grounding tools
- review retrieval
- sequential and hierarchical crew designs
- stronger custom agents
- EDA knowledge integration
- Flow orchestration
- structured submission outputs

Overall, the repository implements all requested lab requirements and bonus items in one integrated CrewAI project.
