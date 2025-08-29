#!/usr/bin/env python3

import os
import sys
sys.path.append('src')

from src.memory.memory import ConversationMemory, ConsumerMemory
from src.algorithms.snurt import User_Topic_Relevance_NoAd_Algorithm
from datetime import datetime, timezone
import json

def test_conversation_memory():
    print("=== Testing ConversationMemory ===")
    
    conv_mem = ConversationMemory()
    conv_id = conv_mem.get_conversation_id()
    print(f"New conversation ID: {conv_id}")
    
    ts_s = datetime.now(timezone.utc).timestamp()
    ts_iso = datetime.now(timezone.utc).isoformat()
    
    conv_mem.add_conversation_memory("User", "Tell me about machine learning", ts_s, ts_iso)
    conv_mem.add_conversation_memory("LLM", "Machine learning is a subset of AI...", ts_s + 1, ts_iso)
    
    messages = conv_mem.get_messages_for_llm()
    print(f"Loaded {len(messages)} messages for LLM")
    print("First message:", messages[0] if messages else "None")
    
    return conv_mem

def test_consumer_memory():
    print("\n=== Testing ConsumerMemory & Snurt ===")
    
    try:
        consumer_mem = ConsumerMemory()
        user_id = "test_user"
        
        test_messages = [
            "I want to learn about machine learning algorithms",
            "Can you explain decision trees?", 
            "How do neural networks work?",
            "What's the difference between supervised and unsupervised learning?",
            "Tell me about Python programming",
            "How do I use pandas for data analysis?"
        ]
        
        print(f"Ingesting {len(test_messages)} test messages...")
        
        for i, msg in enumerate(test_messages):
            topic_id = consumer_mem.ingest_message(user_id, f"test_msg_{i}", msg)
            print(f"  Message {i+1} -> Topic ID: {topic_id}")
        
        scores = consumer_mem.get_user_topic_scores(user_id)
        print(f"\nGenerated {len(scores)} topic scores:")
        
        for topic, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  {topic}: {score:.3f}")
            
        return consumer_mem, user_id
        
    except Exception as e:
        print(f"Error testing ConsumerMemory: {e}")
        print("This might be due to missing VOYAGE_API_KEY or network issues")
        return None, None

def test_snurt_algorithm():
    print("\n=== Testing Snurt Algorithm ===")
    
    try:
        snurt = User_Topic_Relevance_NoAd_Algorithm()
        
        test_text = "Machine learning is fascinating and I spend a lot of time studying it"
        
        external_score = snurt.external_factor_scoring(test_text, dwell_time=45.0, session_freq=5)
        internal_score = snurt.internal_factor_scoring(test_text)
        combined_score = snurt.snurt_reg(test_text, dwell_time=45.0, session_freq=5)
        
        print(f"Text: '{test_text}'")
        print(f"External score: {external_score:.3f}")
        print(f"Internal score: {internal_score:.3f}")
        print(f"Combined Snurt score: {combined_score:.3f}")
        
        decay_test = snurt.calculate_decay(1.0, 1000000, 1086400)
        print(f"Decay test (1.0 after 1 day): {decay_test:.3f}")
        
        return snurt
        
    except Exception as e:
        print(f"Error testing Snurt: {e}")
        return None

def test_integration():
    print("\n=== Testing Full Integration ===")
    
    conv_mem = test_conversation_memory()
    consumer_mem, user_id = test_consumer_memory()
    snurt = test_snurt_algorithm()
    
    if consumer_mem and user_id:
        conversation_messages = conv_mem.load_conversation_memory()
        topic_ids = consumer_mem.link_conversation_to_topics(
            conv_mem.conversation_id, user_id, conversation_messages
        )
        print(f"\nLinked conversation {conv_mem.conversation_id} to topics: {topic_ids}")
        
        relevance = consumer_mem.get_conversation_relevance(conv_mem.conversation_id)
        print(f"Conversation relevance scores: {len(relevance)} topics")

def main():
    print("AdModel Memory System Test")
    print("=" * 50)
    
    os.makedirs("data/memory/conversation", exist_ok=True)
    os.makedirs("data/memory", exist_ok=True)
    
    test_integration()
    
    print("\n" + "=" * 50)
    print("Test completed!")
    print("\nMemory Architecture Summary:")
    print("✓ ConversationMemory: JSON-based per-conversation storage")
    print("✓ ConsumerMemory: SQLite-based topic memory with Snurt scoring")
    print("✓ Snurt Algorithm: Exponential decay + event scoring")
    print("✓ Full Integration: Conversations linked to topic scores")

if __name__ == "__main__":
    main()