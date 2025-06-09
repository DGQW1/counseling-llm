# LLM API Setup Instructions

This guide helps you set up API access for GPT-4o, Gemini, Claude, and Llama.

## 📦 Required Packages

Install the necessary packages for each LLM provider:

```bash
# OpenAI (for GPT-4o)
pip install openai

# Google (for Gemini)
pip install google-generativeai

# Anthropic (for Claude)
pip install anthropic

# For Llama via API providers
pip install requests

# Optional: For local Llama via Ollama
# Install Ollama from https://ollama.ai/
```

## 🔑 API Key Setup

### Option 1: Environment Variables (Recommended)
Create a `.env` file in your project root:

```bash
# .env file
OPENAI_API_KEY=your-openai-api-key-here
GOOGLE_API_KEY=your-google-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
TOGETHER_API_KEY=your-together-api-key-here
```

Then load them in your script:
```python
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
```

## 🎯 Model Provider Configuration

In `process_questions.py`, change the `MODEL_PROVIDER` variable:

```python
# Choose your model provider:
MODEL_PROVIDER = "gpt4o"     # OpenAI GPT-4o
# MODEL_PROVIDER = "gemini"  # Google Gemini
# MODEL_PROVIDER = "claude"  # Anthropic Claude
# MODEL_PROVIDER = "llama"   # Llama via Ollama (local)
# MODEL_PROVIDER = "llama-api" # Llama via API provider
# MODEL_PROVIDER = "mock"    # Mock for testing
```

## 🚀 Getting API Keys

### GPT-4o (OpenAI)
1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create account and add payment method
3. Generate API key

### Gemini (Google)
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create Google account
3. Generate API key


### Claude (Anthropic)
1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Create account and add payment method
3. Generate API key


### Llama (Local via Ollama)
1. Install [Ollama](https://ollama.ai/)
2. Run: `ollama pull llama3.1`
3. Start server: `ollama serve`


### Llama (API via Together AI)
1. Go to [Together AI](https://api.together.xyz/)
2. Create account and add payment method
3. Generate API key

## ⚠️ Rate Limiting & Costs

- **Batch Size**: Adjust `batch_size` in the script (default: 10)
- **Delay**: Adjust `delay_seconds` (default: 1.0s between calls)
- **Estimated Costs for 3,801 questions**:
  - GPT-4o: ~$20-30
  - Gemini: ~$5-10
  - Claude: ~$15-25
  - Llama (local): Free
  - Llama (API): ~$2-5

## 🔄 Running with Different Models

To process your data with different models:

```bash
# Run with GPT-4o
python process_questions.py  # (set MODEL_PROVIDER = "gpt4o")

# Run with Gemini
python process_questions.py  # (set MODEL_PROVIDER = "gemini")

# Run with Claude
python process_questions.py  # (set MODEL_PROVIDER = "claude")

# Run with Llama
python process_questions.py  # (set MODEL_PROVIDER = "llama")
```

Each run will create a separate output file:
- `llm_generated_questions_gpt4o.json`
- `llm_generated_questions_gemini.json`
- `llm_generated_questions_claude.json`
- `llm_generated_questions_llama.json`

## 🛠️ Testing Setup

Start with mock mode to test your setup:
```python
MODEL_PROVIDER = "mock"  # Test without API costs
```

This will generate mock responses to verify your pipeline works before using real APIs. 