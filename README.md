# 🧠 Engram

**Engram** - A passive memory layer for AI conversations. Automatically injects relevant memories into LLM context and extracts new insights from responses.

> *An engram is the physical trace of a memory in the brain.*

## ✨ Key Features

- **Transparent Memory**: Memory is injected and extracted without LLM awareness
- **Semantic Search**: FAISS-based vector storage with sentence transformer embeddings
- **Async Extraction**: Background memory extraction doesn't block conversation flow
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
│  │ 4. EXTRACT (async)  │    │ Background Worker        │   │
│  │    Queue extraction │───▶│ Extract & store memories │   │
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
| `MemoryExtractor` | `memory_extractor.py` | Async extraction of memories from responses |
| `VectorMemory` | `engram_pkg/core.py` | FAISS vector storage and semantic search |
| `MemoryContextIntegrator` | `memory_context.py` | Context-aware retrieval and scoring |
| `ContextWindowManager` | `context_window_manager.py` | Token budget management |

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
├── brain.py              # CLI interface
├── memory_proxy.py       # Main proxy (use this!)
├── memory_extractor.py   # Async memory extraction
├── memory_integration.py # Memory integration layer
├── memory_context.py     # Context-aware retrieval
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
