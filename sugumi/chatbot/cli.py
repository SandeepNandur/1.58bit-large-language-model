"""Interactive command-line interface for the Sugumi mock chatbot."""

import argparse

from sugumi.inference import SugumiEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Sugumi mock chatbot.")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="run the dependency-free mock chatbot (required)",
    )
    parser.add_argument(
        "--checkpoint",
        help="model checkpoint path (real inference is not implemented)",
    )
    args = parser.parse_args()

    if not args.mock:
        parser.error("only mock mode is available; pass --mock")
    if args.checkpoint is not None:
        parser.error("checkpoint inference is not implemented; omit --checkpoint")

    engine = SugumiEngine(mock=True)
    print("Sugumi mock chatbot. Type /quit or /exit to stop.")

    try:
        while True:
            prompt = input("You: ").strip()
            if prompt.lower() in {"/quit", "/exit"}:
                break
            print(f"Sugumi: {engine.chat([{'role': 'user', 'content': prompt}])}")
    except KeyboardInterrupt:
        print("\nGoodbye.")


if __name__ == "__main__":
    main()
