#!/usr/bin/env python3
import json
import csv
import argparse
from pathlib import Path
import sys
from datetime import datetime

def load_json_data(input_file):
    """Load JSON data from file."""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ Loaded {len(data)} records from {input_file}")
        return data
    except FileNotFoundError:
        print(f"❌ Error: File {input_file} not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in {input_file}: {e}")
        sys.exit(1)

def convert_to_basic_csv(data, output_file):
    """Convert JSON data to basic CSV format."""
    fieldnames = [
        'id',
        'model_provider', 
        'situation',
        'context',
        'generated_question',
        'processing_time_seconds',
        'timestamp',
        'success'
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for record in data:
            row = {
                'id': record.get('id', ''),
                'model_provider': record.get('model_provider', ''),
                'situation': record.get('input', {}).get('situation', ''),
                'context': record.get('input', {}).get('context', ''),
                'generated_question': record.get('generated_question', ''),
                'processing_time_seconds': record.get('metadata', {}).get('processing_time_seconds', ''),
                'timestamp': record.get('metadata', {}).get('timestamp', ''),
                'success': record.get('metadata', {}).get('success', '')
            }
            writer.writerow(row)
    
    print(f"✓ Basic CSV saved to {output_file}")

def convert_to_analysis_csv(data, output_file):
    """Convert JSON data to analysis-friendly CSV format with additional columns."""
    fieldnames = [
        'id',
        'model_provider',
        'situation',
        'context',
        'context_length',
        'context_messages',
        'generated_question',
        'question_length',
        'question_word_count',
        'has_empathy_words',
        'has_question_mark',
        'processing_time_seconds',
        'timestamp',
        'success'
    ]
    
    # Common empathy words to check for
    empathy_words = ['feel', 'sounds', 'understand', 'hear', 'difficult', 'challenging', 
                    'overwhelming', 'tough', 'hard', 'painful', 'sorry', 'empathy']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for record in data:
            situation = record.get('input', {}).get('situation', '')
            context = record.get('input', {}).get('context', '')
            question = record.get('generated_question', '')
            
            # Count context messages
            context_messages = len([msg for msg in context.split('|') if msg.strip()]) if context else 0
            
            # Check for empathy words in question
            question_lower = question.lower()
            has_empathy = any(word in question_lower for word in empathy_words)
            
            row = {
                'id': record.get('id', ''),
                'model_provider': record.get('model_provider', ''),
                'situation': situation,
                'context': context,
                'context_length': len(context),
                'context_messages': context_messages,
                'generated_question': question,
                'question_length': len(question),
                'question_word_count': len(question.split()) if question else 0,
                'has_empathy_words': has_empathy,
                'has_question_mark': '?' in question,
                'processing_time_seconds': record.get('metadata', {}).get('processing_time_seconds', ''),
                'timestamp': record.get('metadata', {}).get('timestamp', ''),
                'success': record.get('metadata', {}).get('success', '')
            }
            writer.writerow(row)
    
    print(f"✓ Analysis CSV saved to {output_file}")

def print_summary_stats(data):
    """Print summary statistics about the data."""
    if not data:
        print("No data to analyze")
        return
    
    total_records = len(data)
    successful_records = sum(1 for record in data if record.get('metadata', {}).get('success', False))
    
    # Processing time stats
    processing_times = [record.get('metadata', {}).get('processing_time_seconds', 0) 
                       for record in data if record.get('metadata', {}).get('processing_time_seconds')]
    
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        min_time = min(processing_times)
        max_time = max(processing_times)
    else:
        avg_time = min_time = max_time = 0
    
    # Question length stats
    questions = [record.get('generated_question', '') for record in data]
    question_lengths = [len(q.split()) for q in questions if q]
    
    if question_lengths:
        avg_question_length = sum(question_lengths) / len(question_lengths)
        min_question_length = min(question_lengths)
        max_question_length = max(question_lengths)
    else:
        avg_question_length = min_question_length = max_question_length = 0
    
    print("\n" + "="*50)
    print("📊 SUMMARY STATISTICS")
    print("="*50)
    print(f"Total Records: {total_records}")
    print(f"Successful Records: {successful_records} ({successful_records/total_records*100:.1f}%)")
    print(f"Failed Records: {total_records - successful_records}")
    print()
    print("⏱️  Processing Time:")
    print(f"  Average: {avg_time:.2f} seconds")
    print(f"  Min: {min_time:.2f} seconds") 
    print(f"  Max: {max_time:.2f} seconds")
    print()
    print("📝 Question Analysis:")
    print(f"  Average length: {avg_question_length:.1f} words")
    print(f"  Shortest: {min_question_length} words")
    print(f"  Longest: {max_question_length} words")
    
    # Model provider breakdown
    providers = {}
    for record in data:
        provider = record.get('model_provider', 'unknown')
        providers[provider] = providers.get(provider, 0) + 1
    
    print()
    print("🤖 Model Provider Breakdown:")
    for provider, count in providers.items():
        print(f"  {provider}: {count} records ({count/total_records*100:.1f}%)")
    
    print("="*50)

def main():
    parser = argparse.ArgumentParser(
        description="Convert LLM-generated JSON to CSV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python json_to_csv.py results/llm_generated_questions_gpt4o-mini.json
  python json_to_csv.py input.json -o output.csv --format analysis
  python json_to_csv.py input.json --stats-only
        """
    )
    
    parser.add_argument('input_file', help='Input JSON file path')
    parser.add_argument('-o', '--output', help='Output CSV file path (auto-generated if not specified)')
    parser.add_argument('-f', '--format', choices=['basic', 'analysis'], default='basic',
                       help='Output format: basic (default) or analysis (with extra columns)')
    parser.add_argument('--stats-only', action='store_true', 
                       help='Only print statistics, do not create CSV file')
    parser.add_argument('--no-stats', action='store_true',
                       help='Skip printing statistics')
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ Error: Input file {args.input_file} does not exist")
        sys.exit(1)
    
    # Load data
    data = load_json_data(args.input_file)
    
    # Print statistics
    if not args.no_stats:
        print_summary_stats(data)
    
    # Convert to CSV if not stats-only
    if not args.stats_only:
        # Generate output filename if not provided
        if args.output:
            output_file = args.output
        else:
            stem = input_path.stem
            suffix = '_analysis' if args.format == 'analysis' else '_basic'
            output_file = input_path.parent / f"{stem}{suffix}.csv"
        
        # Convert based on format
        if args.format == 'analysis':
            convert_to_analysis_csv(data, output_file)
        else:
            convert_to_basic_csv(data, output_file)
        
        print(f"\n✅ Conversion complete! Check {output_file}")

if __name__ == "__main__":
    main() 