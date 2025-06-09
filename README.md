## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd counseling-llm-ESConv
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

### 4. Run the Project
```bash
# Test the setup
cd scripts/llm_tasks
python process_questions.py
```

##  Project Structure

```
counseling-llm/
├── data/
│   ├── raw/                    # Original, unprocessed data
│   │   └── ESConv.json        # Raw ESConv dataset
│   ├── processed/             # Processed conversation data
│   │   ├── supporter_questions_with_feedback.json  # Questions grouped by conversation
│   │   └── individual_questions.json              # Flattened individual questions
├── scripts/
│   ├── data_processing/       # Scripts for data extraction and processing
│   │   ├── extract_questions.py     # Extract questions from ESConv.json
│   │   └── prepare_llm_data.py      # Prepare data for LLM training
│   └── llm_tasks/            # Scripts for LLM training and evaluation
│       ├── generate_questions.py    # Generate questions using LLM
│       ├── evaluate_questions.py    # Evaluate generated questions
│       └── train_model.py           # Fine-tune LLM (optional)
├── models/                   # Trained/fine-tuned models
├── results/                  # Evaluation results and outputs
├── configs/                  # Configuration files
│   └── model_config.json
└── README.md
```

## 📊 Data Flow

1. **Raw Data** (`ESConv.json`) → **Extract Questions** → **Processed Data**
2. **Processed Data** → **Prepare LLM Data** → **LLM-Ready Format**
3. **LLM-Ready Data** → **Train/Evaluate LLM** → **Results**

## 🚀 Quick Start

### 1. Data Processing (Already Done)
```bash
cd scripts/data_processing/
python extract_questions.py  # Creates supporter_questions_with_feedback.json
```

### 2. Prepare Data for LLM
```bash
python prepare_llm_data.py   # Creates training/validation/test splits with prompts
```

### 3. Generate Questions with LLM
```bash
cd ../llm_tasks/
python generate_questions.py  # Use LLM to generate questions
```

### 4. Evaluate Results
```bash
python evaluate_questions.py  # Compare generated vs. actual questions
```

## 📝 Data Format for LLM

Each training example follows this format:
```json
{
  "prompt": "Given the situation and conversation context, what would be an appropriate counseling question to ask?\n\nSituation: [situation]\nContext: [dialogue_history]\n\nQuestion:",
  "completion": "[actual_supporter_question]",
  "situation": "[situation]",
  "context": "[dialogue_history]"
}
```

## 🛠️ Key Scripts

- **`extract_questions.py`**: Extract supporter questions from raw data
- **`prepare_llm_data.py`**: Format data for LLM training with proper prompts
- **`generate_questions.py`**: Use LLM to generate counseling questions
- **`evaluate_questions.py`**: Evaluate quality of generated questions
