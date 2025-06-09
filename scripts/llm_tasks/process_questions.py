import json
import os
import time
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv("../../.env")  # Load from project root
except ImportError:
    print("Warning: python-dotenv not installed. Using environment variables directly.")

# Import CSV conversion function from existing script
import sys
sys.path.append('.')
from json_to_csv import convert_to_basic_csv

def load_prompt_template(prompt_path: str = "../../models/prompt.md") -> str:
    """
    Load the prompt template from the markdown file.
    """
    with open(prompt_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

def load_individual_questions(data_path: str = "../../data/processed/individual_questions.json") -> List[Dict]:
    """
    Load the individual questions data.
    """
    with open(data_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def format_prompt(template: str, situation: str, context: str) -> str:
    """
    Format the prompt template with situation and context.
    Assumes template uses {situation} and {context} placeholders.
    """
    # Handle empty context
    if not context.strip():
        context = "[This is the start of the conversation]"
    
    return template.format(situation=situation, context=context)

def call_llm(prompt: str, model_provider: str = "gpt4o-mini", model_config: Dict = None) -> str:
    """
    Call your chosen LLM API.
    
    Args:
        prompt: The formatted prompt to send to the LLM
        model_provider: Which LLM to use ("gpt4o", "gpt4o-mini", "gemini", "claude", "llama")
        model_config: Optional configuration dictionary
    
    Returns:
        Generated response from the LLM
    """
    
    # GPT-4o and GPT-4o-mini (OpenAI)
    def call_openai(prompt: str, model_name: str = "gpt-4o-mini") -> str:
        try:
            import openai
            
            # Load API key from environment
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return "ERROR: OPENAI_API_KEY not found in environment variables"
            
            client = openai.OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7,
                top_p=0.9
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"ERROR: OpenAI call failed - {str(e)}"
    
    # Gemini (Google)
    def call_gemini(prompt: str) -> str:
        try:
            import google.generativeai as genai
            
            # Load API key from environment
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                return "ERROR: GOOGLE_API_KEY not found in environment variables"
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash-lite')  
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=150,
                    temperature=0.7,
                )
            )
            return response.text.strip()
            
        except Exception as e:
            return f"ERROR: Gemini call failed - {str(e)}"
    
    # Claude (Anthropic)
    def call_claude(prompt: str) -> str:
        try:
            import anthropic
            
            # Load API key from environment
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                return "ERROR: ANTHROPIC_API_KEY not found in environment variables"
            
            client = anthropic.Anthropic(api_key=api_key)
            
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",  # or "claude-3-haiku-20240307" for cheaper
                max_tokens=150,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text.strip()
            
        except Exception as e:
            return f"ERROR: Claude call failed - {str(e)}"
    
    # Llama (via Ollama local or API)
    def call_llama_ollama(prompt: str) -> str:
        try:
            import requests
            
            response = requests.post('http://localhost:11434/api/generate',
                json={
                    'model': 'llama3.1',  # or 'llama3.1:70b', 'llama3.1:8b'
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.7,
                        'num_predict': 150
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()['response'].strip()
            else:
                return f"ERROR: Ollama returned status {response.status_code}"
                
        except Exception as e:
            return f"ERROR: Llama (Ollama) call failed - {str(e)}"
    
    # Llama (via Together AI or other API providers)
    def call_llama_api(prompt: str) -> str:
        try:
            import requests
            
            # Load API key from environment
            api_key = os.getenv("TOGETHER_API_KEY")
            if not api_key:
                return "ERROR: TOGETHER_API_KEY not found in environment variables"
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "meta-llama/Llama-3.1-70B-Instruct-Turbo",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150,
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content'].strip()
            else:
                return f"ERROR: Together API returned status {response.status_code}"
                
        except Exception as e:
            return f"ERROR: Llama (API) call failed - {str(e)}"
    
    # Mock for testing
    def call_mock(prompt: str) -> str:
        """Mock LLM response for testing"""
        if "anxiety" in prompt.lower():
            return "Can you tell me more about when you first started experiencing this anxiety?"
        elif "job" in prompt.lower() or "work" in prompt.lower():
            return "How do you feel about your current work situation?"
        elif "friends" in prompt.lower():
            return "What would you like to see change in your friendships?"
        else:
            return "What brings you here today, and how can we work together on this?"
    
    # Route to the appropriate LLM
    if model_provider.lower() == "gpt4o":
        return call_openai(prompt, "gpt-4o")
    elif model_provider.lower() == "gpt4o-mini":
        return call_openai(prompt, "gpt-4o-mini")
    elif model_provider.lower() == "gemini":
        return call_gemini(prompt)
    elif model_provider.lower() == "claude":
        return call_claude(prompt)
    elif model_provider.lower() == "llama":
        return call_llama_ollama(prompt)  # Use Ollama by default
    elif model_provider.lower() == "llama-api":
        return call_llama_api(prompt)  # Use API provider
    elif model_provider.lower() == "mock":
        return call_mock(prompt)
    else:
        return f"ERROR: Unknown model provider '{model_provider}'"

def process_single_question(question_obj: Dict, prompt_template: str, question_id: int, total: int, model_provider: str = "gpt4o-mini") -> Dict:
    """
    Process a single question object through the LLM.
    """
    print(f"Processing question {question_id + 1}/{total} with {model_provider}", end="\r")
    
    situation = question_obj['situation']
    context = question_obj['context']
    
    # Format the prompt
    formatted_prompt = format_prompt(prompt_template, situation, context)
    
    # Call LLM
    start_time = time.time()
    generated_question = call_llm(formatted_prompt, model_provider)
    processing_time = time.time() - start_time
    
    # Create result object
    result = {
        "id": question_id,
        "model_provider": model_provider,
        "input": {
            "situation": situation,
            "context": context
        },
        "prompt": formatted_prompt,
        "generated_question": generated_question,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "processing_time_seconds": round(processing_time, 2),
            "success": not generated_question.startswith("ERROR:")
        }
    }
    
    return result

def save_results_incrementally(results: List[Dict], output_path: str):
    """
    Save results to file after each batch (for safety).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

def process_all_questions(
    questions_data: List[Dict], 
    prompt_template: str, 
    model_provider: str,
    batch_size: int,
    delay_seconds: float,
    output_path: str = "../../results/llm_generated_questions.json",
    max_questions: int = None
) -> List[Dict]:
    """
    Process all questions through the LLM with batching and rate limiting.
    """
    # Limit number of questions if specified (useful for testing)
    if max_questions:
        questions_data = questions_data[:max_questions]
        print(f"Limited to first {max_questions} questions for testing")
    
    results = []
    total_questions = len(questions_data)
    
    print(f"Starting to process {total_questions} questions with {model_provider}...")
    print(f"Batch size: {batch_size}, Delay: {delay_seconds}s between requests")
    
    for i, question_obj in enumerate(questions_data):
        try:
            # Process single question
            result = process_single_question(question_obj, prompt_template, i, total_questions, model_provider)
            results.append(result)
            
            # Add delay to avoid rate limiting
            if delay_seconds > 0:
                time.sleep(delay_seconds)
            
            # Save incrementally every batch
            if (i + 1) % batch_size == 0:
                save_results_incrementally(results, output_path)
                print(f"\nSaved batch at question {i + 1}")
        
        except KeyboardInterrupt:
            print(f"\nProcessing interrupted at question {i + 1}")
            break
        except Exception as e:
            print(f"\nError processing question {i + 1}: {e}")
            # Continue with next question
            continue
    
    # Final save
    save_results_incrementally(results, output_path)
    
    print(f"\n✅ Completed! Processed {len(results)}/{total_questions} questions")
    return results

def print_summary(results: List[Dict]):
    """
    Print summary statistics of the processing.
    """
    total = len(results)
    successful = sum(1 for r in results if r['metadata']['success'])
    failed = total - successful
    
    if total > 0:
        avg_time = sum(r['metadata']['processing_time_seconds'] for r in results) / total
        total_time = sum(r['metadata']['processing_time_seconds'] for r in results)
        
        print(f"\n=== Processing Summary ===")
        print(f"Total questions: {total}")
        print(f"Successful: {successful} ({successful/total*100:.1f}%)")
        print(f"Failed: {failed}")
        print(f"Average processing time: {avg_time:.2f}s per question")
        print(f"Total processing time: {total_time:.2f}s ({total_time/60:.1f} minutes)")

def print_sample_results(results: List[Dict], num_samples: int = 3):
    """
    Print sample results for verification.
    """
    print(f"\n=== Sample Results ===")
    
    for i, result in enumerate(results[:num_samples]):
        print(f"\nSample {i + 1} ({result['model_provider']}):")
        print(f"Situation: {result['input']['situation'][:80]}...")
        print(f"Context: {result['input']['context'][:80]}..." if result['input']['context'] else "Context: [Start of conversation]")
        print(f"Generated Question: {result['generated_question']}")
        print(f"Success: {result['metadata']['success']}")
        print("-" * 80)

def main():
    """
    Main function to process all individual questions through LLM.
    """
    print("🤖 Starting LLM Question Generation Pipeline...")
    
    # Configuration
    MODEL_PROVIDER = "gemini"  # Options: "gpt4o", "gpt4o-mini", "gemini", "claude", "llama", "llama-api", "mock"
    MAX_QUESTIONS = 100
    output_path = f"../../results/llm_generated_questions_{MODEL_PROVIDER}.json"
    batch_size = 20
    delay_seconds = 0.5 
    
    # Load prompt template
    try:
        prompt_template = load_prompt_template()
        print("✅ Loaded prompt template")
    except FileNotFoundError:
        print("❌ Prompt template not found at models/prompt.md")
        print("Please create the file with your prompt template.")
        return
    
    # Load questions data
    try:
        questions_data = load_individual_questions()
        print(f"✅ Loaded {len(questions_data)} questions")
    except FileNotFoundError:
        print("❌ individual_questions.json not found")
        return
    
    print(f"🎯 Using model provider: {MODEL_PROVIDER}")
    if MAX_QUESTIONS:
        print(f"🧪 Testing mode: Processing only {MAX_QUESTIONS} questions")
    
    # Check API key
    if MODEL_PROVIDER.startswith("gpt"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY not found in environment variables")
            print("Please check your .env file")
            return
        else:
            print("✅ OpenAI API key found")
    elif MODEL_PROVIDER == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("❌ GOOGLE_API_KEY not found in environment variables")
            print("Please set your Google API key in environment variables")
            return
        else:
            print("✅ Google API key found")
    elif MODEL_PROVIDER == "claude":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("❌ ANTHROPIC_API_KEY not found in environment variables")
            print("Please check your .env file")
            return
        else:
            print("✅ Anthropic API key found")
    elif MODEL_PROVIDER == "llama-api":
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            print("❌ TOGETHER_API_KEY not found in environment variables")
            print("Please check your .env file")
            return
        else:
            print("✅ Together AI API key found")
    
    # Process all questions
    results = process_all_questions(
        questions_data, 
        prompt_template,
        model_provider=MODEL_PROVIDER,
        batch_size=batch_size,
        delay_seconds=delay_seconds,
        output_path=output_path,
        max_questions=MAX_QUESTIONS
    )
    
    # Print summary and samples
    print_summary(results)
    print_sample_results(results)
    
    # Automatically generate basic CSV
    csv_output_path = output_path.replace('.json', '_basic.csv')
    convert_to_basic_csv(results, csv_output_path)
    
    print(f"\n📁 Results saved to: {output_path}")
    print("🎉 Processing complete!")

if __name__ == "__main__":
    main() 