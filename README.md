# 🧠 Engram

**Engram** - A passive memory layer for AI conversations. Automatically injects relevant memories into LLM context and extracts new insights from responses.

> *An engram is the physical trace of a memory in the brain.*

## ✨ Key Features

- **Transparent Memory**: Memory is injected and extracted without LLM awareness
- **Intelligent MemMan Agent**: Background LLM agent that analyzes conversations and decides what to remember
- **Semantic Search**: FAISS-based vector storage with sentence transformer embeddings
- **Async Extraction**: Background memory extraction doesn't block conversation flow
- **Memory Reinforcement**: Similar memories are reinforced, increasing importance over time
- **Context-Aware**: Multi-factor relevance scoring (importance, recency, usage patterns)
- **Local-First**: All data stored locally, works offline after initial setup

## 📦 Installation

```bash
cd engram
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your API key:
```bash
export GEMINI_API_KEY=your-api-key
# Or place in .env file
```

## 🚀 Quick Start

### Interactive Chat

```bash
python brain.py
```

### CLI Commands

```bash
# Search memories
python brain.py --search "python patterns"

# Add a memory manually
python brain.py --add "Always use type hints in function signatures"

# Remove a memory by ID
python brain.py --remove abc123

# Show statistics
python brain.py --stats
```

### In-Chat Commands

```
/help          Show available commands
/memories      Search your memories
/recent        Show recent memories
/stats         Show session statistics
/add <text>    Manually add a memory
/quit          Exit the chat
```

### Memory Visualizer

Monitor memory state in real-time during conversations:

```bash
# In a separate terminal
python memory_visualizer.py
```

**Sort Modes:**
```bash
python memory_visualizer.py                      # Combined ranking (default)
python memory_visualizer.py --sort importance    # By importance score
python memory_visualizer.py --sort recency       # By timestamp
python memory_visualizer.py --sort access        # By access count
python memory_visualizer.py --query "topic"      # By relevance to query
```

**Keyboard Controls:**
- `q` - Quit
- `1` - Sort by importance
- `2` - Sort by recency
- `3` - Sort by access count
- `4` - Sort by combined score
- `5` - Sort by relevance

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Message                            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PassiveMemoryProxy                             │
│  ┌─────────────────────┐    ┌──────────────────────────┐   │
│  │ 1. RETRIEVE (sync)  │    │ Vector Search (~10ms)    │   │
│  │    Search memories  │───▶│ Get relevant memories    │   │
│  └─────────────────────┘    └──────────────────────────┘   │
│                                                             │
│  ┌─────────────────────┐    ┌──────────────────────────┐   │
│  │ 2. INJECT (sync)    │    │ Context Formatting       │   │
│  │    Build prompt     │───▶│ Add memories to system   │   │
│  └─────────────────────┘    └──────────────────────────┘   │
│                                                             │
│  ┌─────────────────────┐    ┌──────────────────────────┐   │
│  │ 3. CALL LLM (sync)  │    │ Gemini API              │   │
│  │    Get response     │───▶│ Memory-enhanced prompt   │   │
│  └─────────────────────┘    └──────────────────────────┘   │
│                                                             │
│  ┌─────────────────────┐    ┌──────────────────────────┐   │
│  │ 4. EXTRACT (async)  │    │ MemMan Agent (LLM)       │   │
│  │    Queue extraction │───▶│ Analyze & store memories │   │
│  └─────────────────────┘    └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Assistant Response                        │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| `PassiveMemoryProxy` | `memory_proxy.py` | Transparent LLM proxy with memory injection |
| `MemoryExtractor` | `memory_extractor.py` | **MemMan Agent** - async LLM-powered memory management |
| `VectorMemory` | `engram_pkg/core.py` | FAISS vector storage and semantic search |
| `MemoryContextIntegrator` | `memory_context.py` | Context-aware retrieval and scoring |
| `ContextWindowManager` | `context_window_manager.py` | Token budget management |
| `MemoryVisualizer` | `memory_visualizer.py` | Real-time CLI memory visualization |

### 🤖 MemMan Agent

The **MemMan (Memory Manager) Agent** is a background LLM-powered worker that intelligently manages memory extraction:

- **Async Processing**: Runs in a separate thread, never blocking the main conversation
- **LLM Intelligence**: Uses a fast/cheap model (Gemini Flash Lite) to analyze each exchange
- **Smart Filtering**: Decides what's actually worth remembering vs. transient chatter
- **Memory Types**: Extracts preferences, facts, decisions, and insights
- **Confidence Scoring**: Assigns importance and confidence to each memory
- **Memory Reinforcement**: When similar memories are detected, reinforces them (increases importance and access count)
- **Graceful Fallback**: Falls back to heuristic extraction if LLM is unavailable

MemMan output appears in real-time during chat:
```
MemMan: 💾 New (imp:0.70): User prefers dark mode in applications...
MemMan: 🔄 Reinforced (acc:3, imp:0.65→0.72): Working on ProjectX...
```

## 📊 Memory Schema

```python
@dataclass
class MemoryEntry:
    id: str                    # Unique identifier
    content: str               # Memory content
    timestamp: datetime        # Creation time
    importance: float          # 0.0 to 1.0
    tags: List[str]           # Categorization tags
    context: Dict[str, Any]   # Additional metadata
    access_count: int         # Usage tracking
    last_accessed: datetime   # Last retrieval time
    embedding: List[float]    # Vector representation
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | Required |

### ProxyConfig Options

```python
from memory_proxy import PassiveMemoryProxy, ProxyConfig

config = ProxyConfig(
    memory_path="vector_memory",        # Storage location
    max_memories_to_inject=5,           # Memories per query
    min_memory_importance=0.2,          # Minimum importance threshold
    model="gemini-2.0-flash",           # Main LLM model
    extraction_model="gemini-2.0-flash-lite",  # Extraction model
    memory_token_budget=1000,           # Max tokens for memory context
    extraction_enabled=True,            # Enable async extraction
    verbose=False                       # Debug logging
)

proxy = PassiveMemoryProxy(config=config)
```

## 📁 Project Structure

```
engram/
├── brain.py              # CLI chat interface
├── memory_proxy.py       # Main proxy (use this!)
├── memory_extractor.py   # MemMan Agent - async LLM memory management
├── memory_integration.py # Memory integration layer
├── memory_context.py     # Context-aware retrieval
├── memory_visualizer.py  # Real-time memory TUI
├── context_window_manager.py # Token management
├── engram_pkg/           # Core package
│   ├── __init__.py
│   ├── core.py           # VectorMemory class
│   ├── context.py
│   ├── integration.py
│   └── cli.py
├── vector_memory/        # Data storage
│   ├── metadata.pkl
│   └── faiss_index.bin
└── requirements.txt
```

## 📝 License

MIT License
