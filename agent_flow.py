from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from uuid import uuid4

from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel, Field

from lab_project import (
    DEFAULT_ITEM_ID,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT,
    DEFAULT_USER_ID,
    build_fallback_payload,
    build_selected_crew,
    compact_json,
    finalize_payload,
    generate_deterministic_payload,
    normalize_result,
    write_report,
)


class AgentReviewFlowState(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str = Field(default=DEFAULT_USER_ID)
    item_id: str = Field(default=DEFAULT_ITEM_ID)
    model: str = Field(default=DEFAULT_MODEL)
    output: str = Field(default=str(DEFAULT_OUTPUT))
    quiet: bool = Field(default=False)
    artifacts_dir: str = Field(default="")
    stages: str = Field(default="baseline,collaborative,hierarchical")
    baseline_payload: dict | None = Field(default=None)
    collaborative_payload: dict | None = Field(default=None)
    hierarchical_payload: dict | None = Field(default=None)
    final_payload: dict | None = Field(default=None)


class AgentReviewFlow(Flow[AgentReviewFlowState]):
    initial_state = AgentReviewFlowState

    def _selected_stages(self) -> set[str]:
        return {
            stage.strip().lower()
            for stage in self.state.stages.split(",")
            if stage.strip()
        }

    def _should_run_stage(self, stage_name: str) -> bool:
        return stage_name.lower() in self._selected_stages()

    def _resolve_artifacts_dir(self) -> Path:
        output_path = Path(self.state.output).resolve()
        artifacts_dir = (
            Path(self.state.artifacts_dir).resolve()
            if self.state.artifacts_dir
            else output_path.parent / f"{output_path.stem}_artifacts"
        )
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.state.artifacts_dir = str(artifacts_dir)
        return artifacts_dir

    def _write_stage_output(self, stage_name: str, payload: dict) -> None:
        artifacts_dir = self._resolve_artifacts_dir()
        stage_path = artifacts_dir / f"{stage_name}.json"
        stage_path.write_text(compact_json(payload) + "\n", encoding="utf-8")

    def _write_manifest(self) -> None:
        artifacts_dir = self._resolve_artifacts_dir()
        manifest = {
            "flow_id": self.state.id,
            "user_id": self.state.user_id,
            "item_id": self.state.item_id,
            "model": self.state.model,
            "final_output_path": str(Path(self.state.output).resolve()),
            "artifacts_dir": str(artifacts_dir),
            "selected_stages": sorted(self._selected_stages()),
            "stages": {
                "baseline": "baseline.json" if self.state.baseline_payload else "skipped",
                "collaborative": "collaborative.json" if self.state.collaborative_payload else "skipped",
                "hierarchical": "hierarchical.json" if self.state.hierarchical_payload else "skipped",
                "final": "final.json" if self.state.final_payload else None,
            },
        }
        (artifacts_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _looks_generic(payload: dict | None) -> bool:
        if not isinstance(payload, dict):
            return True
        review = str(payload.get("review", "")).strip().lower()
        eda_summary = str(payload.get("eda_summary", "")).strip().lower()
        generic_review_phrases = (
            "looks like a strong fit for me",
            "does not look like a good fit for me",
            "seems like a mixed fit for me",
            "feels mixed for me",
            "could not be matched",
        )
        if any(phrase in review for phrase in generic_review_phrases):
            return True
        if eda_summary.startswith("user ") and ", item " in eda_summary and ", fit " in eda_summary:
            return True
        return len(review.split()) < 8

    def _select_best_payload(self) -> dict:
        candidates = [
            self.state.baseline_payload,
            self.state.collaborative_payload,
            self.state.hierarchical_payload,
        ]
        for payload in candidates:
            if isinstance(payload, dict) and not self._looks_generic(payload):
                return payload
        for payload in candidates:
            if isinstance(payload, dict):
                return payload
        return build_fallback_payload(self.state.user_id, self.state.item_id)

    def _run_crew(self, crew_name: str) -> dict:
        os.environ["AGENT_REVIEW_EMBEDDINGS"] = "none"
        os.environ["AGENT_REVIEW_DISABLE_KNOWLEDGE"] = "1"
        try:
            crew = build_selected_crew(
                crew_name,
                model=self.state.model,
                verbose=not self.state.quiet,
            )
            result = crew.kickoff(
                inputs={
                    "user_id": self.state.user_id,
                    "item_id": self.state.item_id,
                }
            )
            payload = normalize_result(result)
        except Exception:
            payload = generate_deterministic_payload(self.state.user_id, self.state.item_id)
        return finalize_payload(payload, self.state.user_id, self.state.item_id)

    @start()
    def run_baseline(self) -> dict:
        if not self._should_run_stage("baseline"):
            return {"skipped": True, "stage": "baseline"}
        payload = self._run_crew("baseline")
        self.state.baseline_payload = payload
        self._write_stage_output("baseline", payload)
        return payload

    @listen(run_baseline)
    def run_collaborative(self) -> dict:
        if not self._should_run_stage("collaborative"):
            return {"skipped": True, "stage": "collaborative"}
        payload = self._run_crew("collaborative")
        self.state.collaborative_payload = payload
        self._write_stage_output("collaborative", payload)
        return payload

    @listen(run_collaborative)
    def run_hierarchical(self) -> dict:
        if not self._should_run_stage("hierarchical"):
            return {"skipped": True, "stage": "hierarchical"}
        payload = self._run_crew("hierarchical")
        self.state.hierarchical_payload = payload
        self._write_stage_output("hierarchical", payload)
        return payload

    @listen(run_hierarchical)
    def finalize(self) -> dict:
        final_payload = self._select_best_payload()
        final_payload = finalize_payload(
            final_payload,
            self.state.user_id,
            self.state.item_id,
        )
        output_path = Path(self.state.output).resolve()
        write_report(final_payload, output_path)
        self.state.output = str(output_path)
        self.state.final_payload = final_payload
        self._write_stage_output("final", final_payload)
        self._write_manifest()
        return final_payload


def build_flow_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the CrewAI Yelp lab bonus Flow pipeline.")
    parser.add_argument("--user-id", default=DEFAULT_USER_ID, help="Target Yelp user_id.")
    parser.add_argument("--item-id", default=DEFAULT_ITEM_ID, help="Target Yelp item_id/business_id.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM identifier passed to CrewAI.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Where to write the final JSON report.")
    parser.add_argument("--quiet", action="store_true", help="Reduce CrewAI console logging.")
    parser.add_argument(
        "--stages",
        default="baseline,collaborative,hierarchical",
        help="Comma-separated stages to run: baseline, collaborative, hierarchical.",
    )
    return parser


def main() -> int:
    try:
        args = build_flow_parser().parse_args()
        flow = AgentReviewFlow(
            user_id=args.user_id,
            item_id=args.item_id,
            model=args.model,
            output=args.output,
            quiet=args.quiet,
            stages=args.stages,
        )
        payload = flow.kickoff()
        print(compact_json(payload))
        print(f"Saved report to: {Path(flow.state.output).resolve()}")
        return 0
    except Exception as exc:
        print(f"Startup error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
