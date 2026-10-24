from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

load_dotenv()

class ChatResponse(BaseModel):
    answer: str = Field(
        description="Detailed answer to the user's question"
    )

    topic: str = Field(
        description="Main topic of the conversation"
    )

    subtopics: list[str] = Field(
        description="Important subtopics related to the question"
    )

    sentiment: str = Field(
        description="Sentiment: positive, negative, or neutral"
    )

    intent: str = Field(
        description="What the user wants to achieve"
    )

    key_points: list[str] = Field(
        description="Important points from the response"
    )

    examples: list[str] = Field(
        description="Relevant practical examples"
    )

    advantages: list[str] = Field(
        description="Benefits or advantages of the topic"
    )

    disadvantages: list[str] = Field(
        description="Limitations or disadvantages of the topic"
    )

    recommendations: list[str] = Field(
        description="Actionable recommendations for the user"
    )

    difficulty_level: str = Field(
        description="Difficulty level: beginner, intermediate, or advanced"
    )

    follow_up_questions: list[str] = Field(
        description="Relevant questions the user could explore next"
    )


llm = ChatGroq(
    model="openai/gpt-oss-120b"
)

structured_llm = llm.with_structured_output(ChatResponse)


print("=" * 60)
print("       STRUCTURED AI CHATBOT")
print("=" * 60)
print("Type 'exit' to quit.\n")


while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ["exit", "quit"]:
        print("\nBot: Goodbye!")
        break

    if not user_input:
        continue

    try:
        response = structured_llm.invoke(
            f"""
            You are a helpful and knowledgeable AI assistant.

            Answer the user's question clearly and accurately.
            Provide useful examples, key points, advantages,
            disadvantages, and practical recommendations.

            Keep every field relevant to the user's question.
            Do not invent facts. If a field is not applicable,
            return an empty list for list fields or a suitable
            short explanation for string fields.

            User question: {user_input}
            """
        )

        # 5. Display the structured response
        print("\n" + "=" * 60)
        print("                 BOT RESPONSE")
        print("=" * 60)

        print(response)

    except Exception as e:
        print(f"\nError: {e}\n")
