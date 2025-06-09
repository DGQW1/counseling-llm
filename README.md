## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd counseling-llm
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up API Keys
Create a `.env` file in the project root:
```bash
# Create .env file and add your API keys
nano .env

# Add the following to your .env file:
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
TOGETHER_API_KEY=your_together_api_key_here
```

## 📁 Project Structure

```
counseling-llm/
├── data/
│   ├── raw/                          # Original ESConv dataset
│   │   └── ESConv.json              # Raw counseling conversations 
│   └── processed/                   # Processed conversation data
│       ├── supporter_questions_with_feedback.json  # Questions with context 
│       └── individual_questions.json              # Flattened questions 
├── scripts/
│   ├── data_processing/             # Data extraction and preparation
│   │   ├── extract_questions.py    # Extract supporter questions from ESConv
│   │   └── input_context.py        # Flatten questions for processing
│   └── llm_tasks/                   # LLM processing and analysis
│       ├── process_questions.py    # Multi-model question generation
│       ├── json_to_csv.py          # Convert results to CSV format
│       └── analyze_csv.py          # Comprehensive analysis pipeline
├── models/
│   └── prompt.md                    # Professional counseling prompt template
├── results/                         # Organized analysis outputs
│   ├── raw_outputs/                 # Original JSON files from LLMs
│   ├── csv_data/                    # Processed CSV files for analysis
│   ├── extracted_questions/         # Clean question text files
│   ├── visualizations/              # Analysis charts and graphs
│   ├── analysis_reports/            # Summary reports
│   └── README.md                    # Results documentation
└── requirements.txt                 # Project dependencies
```

## 📊 Data Flow

```
ESConv.json (8.6MB)
    ↓ extract_questions.py
supporter_questions_with_feedback.json (2.4MB)
    ↓ input_context.py  
individual_questions.json (1.6MB)
    ↓ process_questions.py
Multi-Model Generated Questions (JSON + CSV)
    ↓ analyze_csv.py
Comprehensive Analysis + Visualizations
```

## 🚀 Quick Start

### 1. Data Processing (Pre-completed)
The ESConv dataset has been processed and is ready for LLM analysis:
- ✅ 3,801 supporter questions extracted
- ✅ 1,249 conversations processed
- ✅ 98.2% have feedback ratings
- ✅ Context and situation included

### 2. Generate Questions with Multiple LLMs
```bash
cd scripts/llm_tasks/
python process_questions.py
```

**Supported Models:**
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`
- **Google**: `gemini-2.0-flash-lite`
- **Anthropic**: `claude-3-5-sonnet-20241022`
- **Meta**: `meta-llama/Llama-3.3-70B-Instruct-Turbo-Free`

### 3. Convert to CSV (Optional)
```bash
python json_to_csv.py results/llm_generated_questions_[model].json
```

### 4. Run Comprehensive Analysis
```bash
python analyze_csv.py
```

**Analysis Features:**
- 📊 Processing time comparison across models
- 📏 Question length analysis (words + characters)
- 🎯 Question starter pattern analysis (first two words)
- 📝 Clean question extraction
- 📋 Professional summary reports

## 🛠️ Key Scripts

### **`extract_questions.py`** 
Extracts supporter questions from raw ESConv data with context and feedback ratings.

### **`process_questions.py`** 
Main LLM processing pipeline with:
- Multi-provider support (OpenAI, Google, Anthropic, Meta)
- Batch processing with rate limiting
- Incremental saving for safety
- Automatic CSV generation

### **`analyze_csv.py`** 
Advanced analysis pipeline featuring:
- Processing time comparison
- Question length analysis  
- Starter pattern analysis
- Professional visualizations
- Organized output management
- Claude-specific cleaning (removes verbose analysis)

### **`json_to_csv.py`** 
Converts JSON outputs to analysis-ready CSV format with statistics.

## 📦 Dependencies

```
# Core LLM APIs
openai>=1.0.0, google-generativeai>=0.8.0, anthropic>=0.25.0, together>=1.0.0

# Data Processing
pandas>=2.0.0, python-dotenv>=1.0.0, requests>=2.31.0

# Visualization & Analysis  
matplotlib>=3.5.0, seaborn>=0.11.0, numpy>=1.21.0, wordcloud>=1.9.0
```