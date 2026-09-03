from langchain_core.messages import HumanMessage

from .graph import build_graph


def main():
    graph = build_graph()
    messages = []
    print("Sambalpuri saree sales agent. Type 'quit' to exit.")
    while True:
        try:
            user_input = input("you: ").strip()
        except EOFError:
            break
        if not user_input or user_input.lower() in {"quit", "exit"}:
            break

        messages.append(HumanMessage(content=user_input))
        result = graph.invoke({"messages": messages})
        messages = result["messages"]
        print(f"agent: {messages[-1].content}")


if __name__ == "__main__":
    main()
