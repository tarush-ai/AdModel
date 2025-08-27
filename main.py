from src.deployment.chatbot import Chatbot
from src.algorithms.basealg import BaseAlgorithm

chatbot = Chatbot()

print("To stop a conversation, say 'pause'. If you wish to restart a conversation, enter 'convo id: ' plus your conversation ID. Your conversation ID is: " + str(chatbot.memory.get_conversation_id()))

id = chatbot.conversation()



