from openai import OpenAI


def rewrite_query(
    question: str,
    conversation_history: list[dict[str, str]],
    client: OpenAI,
    model: str,
) -> str:
    """
    Rewrite a follow-up question as a standalone search query.
    """

    # No history means the question is already standalone.
    if not conversation_history:
        return question

    # Convert conversation history into readable text.
    history_text = ""

    for turn in conversation_history:
        history_text += f"User: {turn['question']}\nAssistant: {turn['answer']}\n\n"

    # Ask the model to rewrite the question for retrieval.
    response = client.responses.create(
        model=model,
        instructions=(
            "Rewrite the latest user question as a standalone "
            "search query for a knowledge base. Use the conversation "
            "history to resolve references such as 'it', 'that', "
            "and 'one'. Preserve the user's intended meaning. "
            "Return only the rewritten query, without explanations "
            "or quotation marks. Do not answer the question."
        ),
        input=(f"Conversation history:\n{history_text}\nLatest question:\n{question}"),
    )

    # Return the rewritten query.
    rewritten_query = response.output_text.strip()

    # Fall back to the original if the result is empty.
    if not rewritten_query:
        return question

    return rewritten_query
