# 🧠 Persistent Memory System

A self-improving AI memory system that automatically tracks conversations and builds persistent knowledge across Cursor IDE chat sessions.

## 🚀 Quick Setup for Cursor IDE

### Automatic Integration (Recommended)

1. **Add to Cursor IDE Settings:**
   ```json
   // Add to your Cursor IDE settings.json or .cursor/.cursorrules
   {
     "memoryIntegration.enabled": true,
     "memoryIntegration.autoTrack": true,
     "memoryIntegration.bootstrapCommand": "cd /home/rob/Dev/persistent_memory_project && python .cursor/activate_memory_integration.py"
   }
   ```

2. **Conversation Start Hook:**
   Add this to your Cursor IDE startup:
   ```bash
   python /home/rob/Dev/persistent_memory_project/.cursor/activate_memory_integration.py
   ```

3. **Message Tracking:**
   For automatic message tracking, Cursor IDE should call:
   ```bash
   python /home/rob/Dev/persistent_memory_project/cursor_memory_hook.py {role} "{message_content}"
   ```

### Manual Integration

If automatic integration isn't working:

1. **Bootstrap Memory System:**
   ```bash
   cd /home/rob/Dev/persistent_memory_project
   source .venv/bin/activate
   python auto_bootstrap.py
   ```

2. **Track Messages Manually:**
   ```bash
   python cursor_memory_hook.py user "Your message here"
   python cursor_memory_hook.py assistant "AI response here"
   ```

## 🎯 Features

- ✅ **Automatic Conversation Tracking** - Every message is automatically remembered
- ✅ **Self-Improving AI** - Memory system learns and optimizes itself
- ✅ **Intelligent Memory Management** - Smart consolidation and deduplication
- ✅ **High Performance** - 45+ queries per second with FAISS vector search
- ✅ **Health Monitoring** - Automatic quality assessment and maintenance
- ✅ **Cursor IDE Integration** - Seamless integration with chat conversations
- ✅ **Improved CLI Interface** - Professional argparse-based command line with comprehensive help
- ✅ **Enhanced Error Handling** - User-friendly error messages with actionable suggestions
- ✅ **Comprehensive Testing** - Full test coverage for critical components
- ✅ **Performance Optimizations** - Lazy loading and shared instances for better performance

## 📊 System Status

```bash
# Check memory system status
python quick_memory.py stats

# Query memories with improved CLI
python quick_memory.py query "How does the memory system work?" --max-memories 5

# Add new memories
python quick_memory.py add "New insight about AI" "ai,insights" 0.8

# Delete memories
python quick_memory.py delete "memory_id"

# Get health report
python persistent_memory.py health

# Run all CLI tests
python test/test_quick_memory_cli.py
```

## 🏗️ Architecture

- **Vector Database**: FAISS with sentence transformer embeddings
- **Memory Storage**: Persistent vector storage with automatic consolidation
- **Insight Detection**: Advanced pattern recognition for automatic memory capture
- **Context Management**: Intelligent priority-based memory selection
- **Performance**: CPU-optimized with AVX2 vector instructions

## 🔧 Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python test/run_tests.py

# Development server
python memory_integration.py
```

## 🛠️ Troubleshooting

### Common Issues

**Memory system not initializing:**
```bash
# Bootstrap the memory system
python quick_memory.py tool

# Or manually run bootstrap
python auto_bootstrap.py
```

**Import errors or missing dependencies:**
```bash
# Install/update dependencies
pip install -r requirements.txt

# Activate virtual environment
source .venv/bin/activate
```

**CLI command errors:**
```bash
# Get help for any command
python quick_memory.py --help
python quick_memory.py query --help

# Run tests to verify system health
python test_quick_memory_cli.py
```

**Performance issues:**
```bash
# Check system performance
python quick_memory.py stats

# The system automatically optimizes performance
# If issues persist, restart the bootstrap
python quick_memory.py tool
```

**Permission errors:**
```bash
# Check file permissions in project directory
ls -la

# Ensure virtual environment is properly activated
source .venv/bin/activate
```

### Recent Improvements

- **CLI Interface**: Upgraded to professional argparse with comprehensive error handling
- **Performance**: Lazy loading and shared instances improve query performance by ~50%
- **Testing**: Added comprehensive CLI tests and improved test coverage
- **Error Handling**: User-friendly error messages with actionable suggestions
- **Organization**: Moved configuration files to `.cursor/` directory

### Getting Help

If you encounter issues not covered here:
1. Run the diagnostic tests: `python test/test_quick_memory_cli.py`
2. Check system status: `python quick_memory.py stats`
3. Restart with bootstrap: `python quick_memory.py tool`

## 📝 Recent Changes

### v1.1.0 - Self-Improvement Update
- **CLI Enhancement**: Upgraded to professional argparse interface with comprehensive help
- **Performance Boost**: Lazy loading and shared MemoryIntegration instances (50%+ performance improvement)
- **Error Handling**: Specific, user-friendly error messages with actionable suggestions
- **Testing**: Added comprehensive test suite with organized `test/` directory
- **Organization**: Moved `.cursorrules` to `.cursor/` directory and cleaned up deprecated files
- **Code Cleanup**: Removed old demo files and database remnants (~48KB freed)
- **Documentation**: Enhanced README with troubleshooting guide and improved examples

---

*This system ensures continuous AI improvement through persistent memory across all conversations.*