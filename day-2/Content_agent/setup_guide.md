# Content Idea Generator & Planner Agent - Complete Setup Guide

## 🚀 Quick Start (5 minutes)

### Step 1: Install Ollama (the free, local LLM engine)
```bash
# Go to https://ollama.ai and download for your OS:
# • macOS: https://ollama.ai/download/mac
# • Windows: https://ollama.ai/download/windows  
# • Linux: https://ollama.ai/download/linux

# After installation, verify:
ollama --version
```

### Step 2: Download a Model
```bash
# Pull the Mistral model (recommended - lightweight, fast, creative)
ollama pull mistral

# Or other great options:
ollama pull neural-chat       # Excellent for conversations
ollama pull llama2            # More powerful, needs more RAM
ollama pull qwen              # Multilingual, creative
ollama pull openchat          # Fast, good quality
```

### Step 3: Start Ollama Server
```bash
# In a new terminal window, run:
ollama serve

# You should see output like:
# "Listening on 127.0.0.1:11434"
# Keep this terminal open!
```

### Step 4: Install Python Dependencies
```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install langchain langchain-community langgraph pydantic
```

### Step 5: Run the Agent
```bash
# In the main terminal (with venv activated):
python content_agent.py

# You should see the interactive prompt!
```

---

## 📦 Dependencies Explained

| Package | Why We Need It | Free? |
|---------|---------------|-------|
| **Ollama** | Local LLM engine (no API keys needed) | ✅ Yes, completely free |
| **LangChain** | Framework for building agents | ✅ Yes, open-source |
| **LangGraph** | Orchestrates agent logic and tools | ✅ Yes, open-source |
| **Pydantic** | Data validation | ✅ Yes, open-source |

**Total cost: $0** - Everything is completely free and open-source!

---

## 🎯 Usage Examples

### Example 1: Generate Content Ideas
```
You: I'm a productivity coach. Generate 5 blog post ideas for busy professionals 
     who want to work smarter.

Agent: [Generates creative, targeted ideas with hooks and angles]
```

### Example 2: Create Full Content Plan
```
You: Create a detailed outline for a YouTube video on "How to Learn Machine Learning 
     in 3 Months" targeting complete beginners. Then suggest a content calendar for 
     promoting this across social media.

Agent: [Creates outline with timestamps, suggests distribution schedule, recommends 
        complementary content]
```

### Example 3: Personalize Your Strategy
```
You: My brand voice is friendly and humorous. I'm a software engineer. 
     Personalize these technical content ideas to match my style.

Agent: [Adapts suggestions with humor, casual tone, real examples from engineering]
```

### Example 4: Research & Optimize
```
You: Research SEO keywords for content about "sustainable technology for Gen Z".
     Then suggest different content formats that would work best.

Agent: [Provides keyword analysis and format recommendations with platform-specific tips]
```

---

## 🔧 Troubleshooting

### ❌ "Connection refused on localhost:11434"
**Solution:**
```bash
# Make sure Ollama is running in a separate terminal
ollama serve
```

### ❌ "Model not found" or "Invalid model"
**Solution:**
```bash
# List available models
ollama list

# If empty, pull a model
ollama pull mistral

# Check what models you have
ollama list
```

### ❌ "Slow responses" or "High memory usage"
**Solution:**
- Use smaller models: `neural-chat` instead of `llama2`
- Increase your system's available memory
- Close other applications
- Try: `ollama pull neural-chat`

### ❌ On Windows: "command not found"
**Solution:**
- Make sure Ollama is installed
- Restart your terminal
- Run from Python directly: `python content_agent.py`

### ❌ "ModuleNotFoundError: No module named 'langchain'"
**Solution:**
```bash
# Make sure you're in the virtual environment
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Then install again
pip install langchain langchain-community langgraph pydantic
```

---

## 🎓 For Your Workshop

### Teaching Progression (16-hour workshop)

**Hours 0-2: Setup & Fundamentals**
- Install Ollama and dependencies
- Understand agent architecture
- Explore `content_agent.py` code structure
- Run first agent conversation

**Hours 2-4: Tools & Tool Calling**
- Understand how tools work
- Add custom tools to the agent
- Modify tool parameters
- Test different tool combinations

**Hours 4-8: Agent Logic & Reasoning**
- Study the system prompt
- Modify agent behavior and personality
- Implement multi-step reasoning
- Debug agent decisions

**Hours 8-12: Advanced Features**
- Add persistent memory
- Implement content storage
- Create feedback loops
- Build evaluation metrics

**Hours 12-16: Production & Deployment**
- Error handling and edge cases
- Performance optimization
- Testing frameworks
- Deployment options (Docker, API, etc.)

---

## 🚀 Advanced Customizations

### Option 1: Use a More Powerful Model
```python
# In content_agent.py, change the model:
llm = Ollama(
    model="llama2",  # Larger, more capable
    base_url="http://localhost:11434",
    temperature=0.7,
)
```

### Option 2: Add Custom Tools
```python
@tool
def my_custom_tool(input: str) -> str:
    """My custom tool for content creation"""
    # Your implementation here
    return result

# Add to tools list:
tools = [
    brainstorm_ideas,
    create_outline,
    my_custom_tool,  # Add here
    # ... other tools
]
```

### Option 3: Adjust Creativity/Consistency
```python
# More creative (unpredictable):
temperature=0.9

# More consistent (predictable):
temperature=0.3

# Balanced:
temperature=0.7
```

### Option 4: Save Conversation History
```python
# Add to track conversations
from datetime import datetime

conversation_history = {
    "timestamp": datetime.now().isoformat(),
    "messages": [],
    "ideas_generated": []
}
```

---

## 📊 Model Comparison

| Model | Size | Speed | Quality | RAM Needed | Best For |
|-------|------|-------|---------|-----------|----------|
| **Neural-Chat** | 7B | ⚡⚡⚡ | ⭐⭐⭐⭐ | 4GB | Fast, good quality |
| **Mistral** | 7B | ⚡⚡ | ⭐⭐⭐⭐⭐ | 8GB | **Recommended** |
| **Llama2** | 7B-70B | ⚡ | ⭐⭐⭐⭐ | 8-64GB | Very capable |
| **Qwen** | 7B-72B | ⚡⚡ | ⭐⭐⭐⭐⭐ | 8-64GB | Creative, multilingual |
| **OpenChat** | 3.5B-8B | ⚡⚡⚡ | ⭐⭐⭐ | 4-8GB | Lightweight |

---

## 🔐 Privacy & Security

✅ **Completely Private** - Everything runs locally on your machine
✅ **No Internet Required** - After initial model download
✅ **No Data Collection** - Your conversations stay on your computer
✅ **No API Keys** - No external accounts or billing
✅ **Open Source** - Full code transparency

---

## 💡 Extension Ideas for Workshop

1. **Persist to Database**: Save generated ideas to SQLite
2. **Web Interface**: Build a Flask/Django UI
3. **Multi-Agent**: Combine with other specialized agents
4. **Analytics**: Track performance of generated content ideas
5. **Integration**: Connect to platforms (Twitter, Medium, YouTube)
6. **Fine-tuning**: Train on your specific content domain
7. **Streaming**: Real-time response streaming for UX
8. **Evaluation**: Implement idea quality metrics

---

## 📚 Resources for Learning More

**Ollama Documentation**: https://github.com/ollama/ollama
**LangChain Documentation**: https://python.langchain.com
**LangGraph Documentation**: https://langchain-ai.github.io/langgraph
**Open Source LLMs**: https://huggingface.co/models

---

## 🎉 You're Ready!

You now have a completely free, local AI agent that can:
- Generate creative content ideas
- Create detailed outlines
- Develop compelling hooks
- Research keywords
- Plan content strategies
- Personalize recommendations

**No subscriptions. No API costs. No data sharing.**

Happy creating! 🚀
