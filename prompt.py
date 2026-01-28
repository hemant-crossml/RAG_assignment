from langchain_core.prompts import ChatPromptTemplate


def get_rag_prompt():
    return ChatPromptTemplate.from_template(
        """
You are a helpful assistant.

Rules:
- Use ONLY the provided context for factual answers
- If the answer is not in the context, say "I don't know"
- Be concise and clear

Context:
{context}

Question:
{question}
"""
    )
