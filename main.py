from src.deployment.chatbot import Chatbot

user_id = input("Enter your user ID (or press Enter for 'default_user'): ").strip() or "default_user"
chatbot = Chatbot(user_id=user_id)

print(f"User: {user_id}")
print("To stop a conversation, say 'pause'. If you wish to restart a conversation, enter 'convo id: ' plus your conversation ID. Your conversation ID is: " + str(chatbot.memory.get_conversation_id()))

conversation_id = chatbot.conversation()
print(f"\nConversation ended. Final conversation ID: {conversation_id}")

topic_scores = chatbot.consumer_memory.get_user_topic_scores(user_id)
if topic_scores:
    print(f"\nYour current topic interests:")
    for topic, score in sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {topic}: {score:.3f}")
else:
    print("\nNo topic data available yet.")



