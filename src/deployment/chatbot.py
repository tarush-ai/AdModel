import openai
import os
from dotenv import load_dotenv
from src.config import config
from src.memory.memory import ConversationMemory, ConsumerMemory
from src.algorithms.snurt import User_Topic_Relevance_NoAd_Algorithm
from datetime import datetime, timezone

load_dotenv()

client = openai.OpenAI()

class Chatbot:
    def __init__(self, user_id: str = "default_user"): 
        self.client = client
        self.memory = ConversationMemory()
        self.consumer_memory = ConsumerMemory()
        self.snurt = User_Topic_Relevance_NoAd_Algorithm()
        self.user_id = user_id
        self.rate_limit = config.RATE_LIMIT

    def now_utc(self):
        dt = datetime.now(timezone.utc)
        return dt.timestamp(), dt.isoformat()   

    def conversation(self) -> str:
        counter = 0
        while True:
            userinput = input("User: ")
            
            if userinput.strip().lower() == "pause": break

            elif counter >= self.rate_limit:
                print("Rate limit reached. Please restart conversation.")
                break

            elif "convo id: " in userinput.strip().lower(): 
                conversation_id = userinput.strip().lower().split("convo id: ")[1]
                if len(conversation_id) != 26:
                    print(f"Invalid conversation ID format. Expected 26 characters, got {len(conversation_id)}.")
                    print(f"Please provide the full conversation ID (e.g., '01K3V834676CHACDYWK7XSGZR5')")
                    continue
                
                self.memory.set_conversation_id(conversation_id)
                
                if os.path.exists(self.memory.path):
                    loaded_history = self.memory.load_conversation_memory()
                    print(f"Conversation ID set to: {self.memory.conversation_id}")
                    print(f"Loaded {len(loaded_history)} previous messages from conversation.")
                else:
                    print(f"Warning: No existing conversation found for ID {self.memory.conversation_id}")
                    print("Starting a new conversation with this ID.")
                continue

            ts_s, ts_iso = self.now_utc()
            self.memory.add_conversation_memory("User", userinput, ts_s, ts_iso)

            dwell_time = 30.0 if len(userinput) > 50 else 15.0
            session_freq = counter + 1
            self.consumer_memory.ingest_message(self.user_id, f"msg_{int(ts_s)}", userinput, ts_s, dwell_time, session_freq)

            message_history = self.memory.get_messages_for_llm()
            
            topic_scores = self.consumer_memory.get_user_topic_scores(self.user_id)
            top_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            
            context_prompt = ""
            if top_topics:
                topic_context = ", ".join([f"{name} ({score:.2f})" for name, score in top_topics])
                context_prompt = f"User's current topic interests: {topic_context}. "

            messages = message_history + [
                {
                    "role": "user",
                    "content": context_prompt + userinput
                }
            ]

            response = self.client.chat.completions.create(
                model = config.TEST_MODEL,
                messages = messages
            )

            llm_response = response.choices[0].message.content
            print("Assistant: " +llm_response)
            
            ts_s, ts_iso = self.now_utc()
            self.memory.add_conversation_memory("LLM", llm_response, ts_s, ts_iso)
            
            conversation_messages = self.memory.load_conversation_memory()
            self.consumer_memory.link_conversation_to_topics(
                self.memory.conversation_id, self.user_id, conversation_messages
            )
            
            counter += 1
        return self.memory.conversation_id
        


        
