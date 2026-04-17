from __future__ import annotations

from lab_project import build_parser, build_selected_crew, compact_json


def main() -> int:
    try:
        args = build_parser("Run any CrewAI lab topology for the Yelp rating project.").parse_args()
        crew = build_selected_crew(args.crew, model=args.model, verbose=not args.quiet)
        result = crew.kickoff(inputs={"user_id": args.user_id, "item_id": args.item_id})
        payload = getattr(result, "raw", result)
        print(payload if isinstance(payload, str) else compact_json(payload))
        return 0
    except Exception as exc:
        message = str(exc)
        if "OPENAI_API_KEY is required" in message:
            print("Startup error: the selected model requires OPENAI_API_KEY. Set it in your environment or switch to a provider like Groq with --model.")
            return 1
        if "GROQ_API_KEY" in message or "groq" in message.lower() and "api key" in message.lower():
            print("Startup error: the selected Groq model requires GROQ_API_KEY. Set it in your environment and try again.")
            return 1
        if "COHERE_API_KEY" in message:
            print("Startup error: COHERE_API_KEY is missing. Add it only if you want Cohere-backed knowledge or semantic RAG.")
            return 1
        print(f"Startup error: {message}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
