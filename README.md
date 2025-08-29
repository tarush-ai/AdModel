# AdModel - SNURT Topic Memory System

A sophisticated conversational AI system implementing the **SNURT** (Stored Non-Ad User Relevance Topic) memory architecture for persistent, decaying topic relevance scoring across conversations.

## Overview

This system combines real-time conversation management with long-term topic memory that naturally fades over time while being reinforced by user engagement. It follows the complete CURSOR.txt specification for event-driven topic scoring with exponential decay.

## Architecture

### Core Components

```
AdModel/
├── src/
│   ├── algorithms/           # All mathematical computations
│   │   ├── common.py        # Shared math utilities (z-score, sigmoid, decay)
│   │   ├── snurt_engine.py  # Complete CURSOR.txt implementation
│   │   ├── snurt.py         # High-level SNURT interface
│   │   └── basealg.py       # Base algorithm classes
│   ├── memory/              # Data persistence (math-free)
│   │   ├── memory.py        # Conversation + Consumer memory interfaces
│   │   └── topicmemory.py   # SQLite-based topic storage
│   ├── deployment/          # User-facing interfaces
│   │   ├── chatbot.py       # Interactive chat interface
│   │   └── ad.py            # Ad recommendation system
│   └── config.py            # System configuration
├── data/
│   └── memory/
│       ├── conversation/    # Per-conversation JSON files
│       └── consumer.db      # SQLite topic database
├── main.py                  # Entry point
└── test_memory_system.py    # System testing
```

## File Structure Explanation

### Algorithms Layer (`src/algorithms/`)

**`common.py`** - Mathematical utilities used across the system:
- `exponential_decay()`: Core SNURT decay formula `S * exp(-λ * Δt)`
- `l2_norm()`, `normalize_vector()`: Vector operations for embeddings
- `cosine_similarity()`: Topic assignment via nearest centroid
- `simple_z_scaling()`, `robust_z_scaling()`: Cross-topic normalization
- `sigmoid()`: Final score mapping to [0,1]

**`snurt_engine.py`** - Complete CURSOR.txt feature implementation:
- **Dwell scoring**: `tanh(log1p(dwell_seconds) / log1p(180))`
- **Frequency scoring**: `1/(1 + exp(-log1p(session_freq)))`
- **Relevance scoring**: Lifetime relevance with exponential decay
- **Prompt length scoring**: `1/(1 + exp(-log1p(text_length)))`
- **Click utility scoring**: Click-through rate with capping
- **Gratitude scoring**: Sentiment analysis for user satisfaction
- **Verbosity multiplier**: Message length adjustment factor
- **Event score combination**: `w_ext * external + w_int * internal`

**`snurt.py`** - High-level interface for topic scoring:
- `update_topic_score()`: Process new user messages
- `get_user_topic_scores()`: Retrieve current topic relevance
- `get_top_topics()`: Get most relevant topics for user

### Memory Layer (`src/memory/`)

**`memory.py`** - Dual memory system interface:
- `ConversationMemory`: JSON-based per-conversation storage
- `ConsumerMemory`: SQLite-based cross-conversation topic memory
- `link_conversation_to_topics()`: Connect chats to topic scores

**`topicmemory.py`** - SQLite persistence (math-free):
- Topic assignment via embedding similarity
- Centroid management with count-weighted updates
- Message ingestion with full CURSOR.txt parameter support
- Score retrieval with automatic decay application

### Deployment Layer (`src/deployment/`)

**`chatbot.py`** - Interactive conversation interface:
- Real-time topic scoring during conversation
- Context injection based on user's top topics
- Conversation resumption with memory persistence
- Multi-user support with isolated topic spaces

**`ad.py`** - Advertisement recommendation system
**`config.py`** - System configuration with CURSOR.txt parameters

## SNURT Algorithm Flow

### 1. Message Ingestion
```python
# User sends message
user_input = "I really appreciate your help with machine learning!"

# Extract engagement signals
dwell_time = 45.0  # seconds spent typing
session_freq = 3   # 3rd message in session
```

### 2. Feature Scoring
```python
# External factors (user behavior)
s_dur = tanh(log1p(45) / log1p(180))     # 0.627 - good dwell time
s_freq = 1/(1 + exp(-log1p(3)))          # 0.800 - moderate frequency  
s_rel = relevance_score(lifetime_rel)     # 0.500 - topic relevance

external = 0.50*s_dur + 0.30*s_freq + 0.20*s_rel = 0.687

# Internal factors (message content)
s_plen = 1/(1 + exp(-log1p(len)))        # 0.945 - good length
s_grateful = gratitude_score(text)        # 0.200 - contains "appreciate"
verbosity = 0.889                         # moderate verbosity

internal = 0.889 * (0.60*0.945 + 0.10*0.200) = 0.546

# Combined event score
event_score = 0.30*0.687 + 0.70*0.546 = 0.588
```

### 3. Topic Update
```python
# Apply exponential decay to previous score
λ = ln(2) / 21.0  # 21-day half-life
Δ_days = (now - last_updated) / 86400
S_decayed = S_previous * exp(-λ * Δ_days)

# Add new event contribution
S_new = S_decayed + η * event_score
S_new = 0.45 + 1.0 * 0.588 = 1.038
```

### 4. Cross-Topic Normalization
```python
# Get all user topics: [1.038, 0.234, 0.567, 0.123, 0.891]
# Z-score normalize
z_scores = [(x - μ) / σ for x in scores]  # [1.2, -0.8, 0.1, -1.1, 0.6]

# Sigmoid mapping with steepness k=1.4
final_scores = [1/(1 + exp(-k*z)) for z in z_scores]
# Result: [0.828, 0.246, 0.535, 0.192, 0.696]
```

### 5. Context Injection
```python
# Top topics for context
top_topics = [
    ("Machine Learning", 0.828),
    ("Python Programming", 0.696), 
    ("Data Science", 0.535)
]

# Inject into LLM prompt
context = "User's current topic interests: Machine Learning (0.83), Python Programming (0.70), Data Science (0.54). "
full_prompt = context + user_input
```

## Sample Complete Flow

### Scenario: User Learning Machine Learning

```
1. USER STARTS CONVERSATION
   Input: "Can you explain neural networks?"
   
   → ConversationMemory: Creates new conversation (01K3V9...)
   → TopicMemory: Creates "neural networks" topic
   → SnurtEngine: Scores based on length, no dwell/frequency yet
   → Result: Topic "neural networks" = 0.45

2. USER ASKS FOLLOW-UP (30 seconds later)
   Input: "That's really helpful! Can you explain backpropagation?"
   
   → Dwell time: 30s (good engagement)
   → Session frequency: 2nd message
   → Gratitude detected: "really helpful"
   → Event score: 0.72 (high due to gratitude + dwell)
   → Topic update: 0.45 + 0.72 = 1.17
   → Normalized: 0.85

3. USER CONTINUES CONVERSATION
   Input: "How do I implement this in Python?"
   
   → Creates new topic "Python implementation" 
   → Links to existing ML interest
   → Context injection: "User's interests: Neural Networks (0.85)"
   → LLM receives context-aware prompt

4. USER RETURNS NEXT DAY
   → All topics decay: S * exp(-ln(2) * 1/21) = S * 0.967
   → Neural networks: 0.85 * 0.967 = 0.82 (slight decay)
   → System remembers but with natural fading

5. USER ASKS ABOUT DIFFERENT TOPIC
   Input: "What's the weather like?"
   
   → Creates new topic with low score
   → ML topics remain high in background
   → Future ML questions will reactivate and boost those topics
```

## Configuration

Key parameters in `src/config.py`:

```python
HALF_LIFE_DAYS: float = 21.0      # Topic decay rate
TAU_ABS: float = 0.80             # Topic assignment threshold  
DELTA: float = 0.05               # Margin requirement
ETA: float = 1.0                  # Learning rate
K: float = 1.4                    # Sigmoid steepness
LEARNING_RATE: float = 1.0        # Event score multiplier
```

## Usage

### Basic Usage
```bash
# Start interactive chatbot
python main.py

# Enter user ID (for multi-user support)
# Chat normally - topics are automatically tracked
# Say 'pause' to end conversation
# Use 'convo id: <ID>' to resume previous conversation
```

### API Usage
```python
from src.algorithms.snurt import User_Topic_Relevance_NoAd_Algorithm

# Initialize SNURT system
snurt = User_Topic_Relevance_NoAd_Algorithm()

# Process user message with engagement signals
topic_id = snurt.update_topic_score(
    user_id="user123",
    text="I love learning about machine learning!",
    dwell_seconds=45.0,
    session_freq=3,
    clicked=1,
    provided=2
)

# Get current topic relevance scores
scores = snurt.get_user_topic_scores("user123")
# Returns: {"Machine Learning": 0.85, "Python": 0.67, ...}

# Get top topics for context
top_topics = snurt.get_top_topics("user123", limit=3)
# Returns: [("Machine Learning", 0.85), ("Python", 0.67), ...]
```

### Testing
```bash
# Run comprehensive system tests
python test_memory_system.py

# Test specific components
python -c "from src.algorithms.snurt_engine import SnurtEngine; print('✓ CURSOR.txt compliant')"
```

## Database Schema

SQLite tables following CURSOR.txt specification:

```sql
-- Core topic storage with SNURT scores
topics(
  id INTEGER PRIMARY KEY,
  user_id TEXT NOT NULL,
  name TEXT NOT NULL,           -- Human-readable topic name
  S REAL NOT NULL DEFAULT 0.0,  -- S_nurt_raw accumulator  
  t REAL NOT NULL DEFAULT 0.0   -- last_updated timestamp
)

-- Multi-prototype centroids per topic
centroids(
  id INTEGER PRIMARY KEY,
  topic_id INTEGER,
  provider_id TEXT,             -- "voyage-3.5-lite" etc
  vec TEXT,                     -- JSON embedding vector
  count INTEGER DEFAULT 1      -- Prototype weight
)

-- Message ground truth
messages(
  id TEXT PRIMARY KEY,          -- Unique message identifier
  user_id TEXT NOT NULL,
  text TEXT NOT NULL,
  ts REAL NOT NULL             -- Timestamp
)

-- Topic membership links  
topic_members(
  topic_id INTEGER,
  msg_id TEXT,
  FOREIGN KEY(topic_id) REFERENCES topics(id),
  FOREIGN KEY(msg_id) REFERENCES messages(id)
)
```

## Key Features

✅ **CURSOR.txt Compliant**: Full implementation of the specification  
✅ **Exponential Decay**: Natural topic fading over configurable half-life  
✅ **Multi-Factor Scoring**: 7 engagement signals (dwell, frequency, gratitude, etc.)  
✅ **Provider-Agnostic**: Swappable embedding providers (Voyage AI default)  
✅ **Cross-Conversation Memory**: Topics persist across chat sessions  
✅ **Real-Time Context**: LLM receives user's current topic interests  
✅ **Multi-User Support**: Isolated topic spaces per user  
✅ **Conversation Resumption**: Full chat history restoration  
✅ **Mathematical Separation**: All math in algorithms/, data layer math-free  

## Dependencies

```bash
pip install -r requirements.txt

# Key dependencies:
# - voyageai: Embedding provider
# - openai: LLM interface  
# - sqlite3: Topic persistence (built-in)
# - ulid: Unique conversation IDs
```

## Environment Variables

```bash
# Required for embeddings (optional for testing)
export VOYAGE_API_KEY="your_voyage_api_key"

# Required for LLM
export OPENAI_API_KEY="your_openai_api_key"
```

## License

Apache 2.0 License - See LICENSE file for details.