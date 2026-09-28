from research_agent.model import get_model


def main():
    model = get_model()

    messages = [
        {
            "role": "user",
            "content": "What is Artificial Intelligence? Explain in 3 simple sentences."
        }
    ]

    response = model.generate(messages)

    print("\n=== Research Agent Baseline ===")
    print(response.text)
    print(f"\nLatency: {response.latency_s:.2f} seconds")
    print(f"Completion tokens: {response.completion_tokens}")


if __name__ == "__main__":
    main()