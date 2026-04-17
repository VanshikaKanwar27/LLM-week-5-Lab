from __future__ import annotations

from pathlib import Path

from lab_project import (
    build_fallback_payload,
    build_parser,
    build_selected_crew,
    compact_json,
    finalize_payload,
    generate_deterministic_payload,
    normalize_result,
    write_report,
)


def main() -> int:
    try:
        args = build_parser("Run the CrewAI Yelp lab with JSON knowledge and review RAG.").parse_args()
        if args.engine == "deterministic":
            payload = generate_deterministic_payload(args.user_id, args.item_id)
        else:
            crew = build_selected_crew(args.crew, model=args.model, verbose=not args.quiet)
            result = crew.kickoff(inputs={"user_id": args.user_id, "item_id": args.item_id})
            try:
                payload = normalize_result(result)
            except Exception:
                payload = build_fallback_payload(args.user_id, args.item_id)
            payload = finalize_payload(payload, args.user_id, args.item_id)

        output_path = Path(args.output).resolve()
        write_report(payload, output_path)

        print(compact_json(payload))
        print(f"Saved report to: {output_path}")
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
        if "last_raw_response.txt" in message:
            print(f"Startup error: {message}")
            return 1
        print(f"Startup error: {message}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
