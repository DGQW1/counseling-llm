#!/usr/bin/env python3
"""
Comprehensive analysis script for LLM-generated counseling questions.
Analyzes processing times, creates visualizations, and extracts questions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import glob
import sys
import re
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class LLMAnalyzer:
    """
    A comprehensive analyzer for LLM processing results.
    """
    
    def __init__(self, results_dir="../../results"):
        """
        Initialize the analyzer with results directory.
        
        Args:
            results_dir: Path to directory containing CSV files
        """
        self.results_dir = Path(results_dir)
        self.data = {}
        self.combined_data = None
        
        # Create organized subdirectories if they don't exist
        self.create_organized_structure()
        
    def create_organized_structure(self):
        """
        Create organized subdirectories in results folder.
        """
        subdirs = [
            "raw_outputs",
            "csv_data", 
            "extracted_questions",
            "visualizations",
            "analysis_reports"
        ]
        
        for subdir in subdirs:
            (self.results_dir / subdir).mkdir(exist_ok=True)

    def load_csv_files(self):
        """
        Load all CSV files from the csv_data subdirectory.
        """
        csv_dir = self.results_dir / "csv_data"
        csv_files = list(csv_dir.glob("*_basic.csv"))
        
        # Fallback to root directory if no files in csv_data
        if not csv_files:
            csv_files = list(self.results_dir.glob("*_basic.csv"))
        
        if not csv_files:
            print(f"❌ No CSV files found in {csv_dir} or {self.results_dir}")
            return False
            
        print(f"📁 Found {len(csv_files)} CSV files:")
        
        for csv_file in csv_files:
            # Extract model name from filename
            model_name = csv_file.stem.replace('llm_generated_questions_', '').replace('_basic', '')
            
            try:
                # Use proper CSV reading parameters to handle commas in quoted fields
                df = pd.read_csv(csv_file, quotechar='"', escapechar='\\', on_bad_lines='skip')
                self.data[model_name] = df
                print(f"   ✅ {model_name}: {len(df)} records")
            except Exception as e:
                print(f"   ❌ Error loading {csv_file}: {e}")
                # Try alternative CSV reading approach
                try:
                    df = pd.read_csv(csv_file, sep=',', quotechar='"', doublequote=True, 
                                   skipinitialspace=True, on_bad_lines='skip')
                    self.data[model_name] = df
                    print(f"   ✅ {model_name}: {len(df)} records (recovered)")
                except Exception as e2:
                    print(f"   ❌ Failed to recover {csv_file}: {e2}")
                
        # Combine all data for cross-model analysis
        if self.data:
            all_dfs = []
            for model, df in self.data.items():
                df['model'] = model
                all_dfs.append(df)
            self.combined_data = pd.concat(all_dfs, ignore_index=True)
            
        return len(self.data) > 0
    
    def analyze_processing_times(self):
        """
        Analyze and compare processing times across models.
        """
        if self.combined_data is None:
            print("❌ No data loaded")
            return
            
        print("\n" + "="*60)
        print("⏱️  PROCESSING TIME ANALYSIS")
        print("="*60)
        
        # Filter successful records only
        successful_data = self.combined_data[self.combined_data['success'] == True].copy()
        
        if len(successful_data) == 0:
            print("❌ No successful records found")
            return
            
        # Statistics by model
        time_stats = successful_data.groupby('model')['processing_time_seconds'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(3)
        
        print(f"\n📊 Processing Time Statistics (successful records only):")
        print(time_stats.to_string())
        
        # Success rates
        print(f"\n✅ Success Rates:")
        success_rates = self.combined_data.groupby('model')['success'].agg(['count', 'sum']).round(3)
        success_rates['success_rate'] = (success_rates['sum'] / success_rates['count'] * 100).round(1)
        
        for model in success_rates.index:
            total = success_rates.loc[model, 'count']
            successful = success_rates.loc[model, 'sum']
            rate = success_rates.loc[model, 'success_rate']
            print(f"   {model}: {successful}/{total} ({rate}%)")
            
        return time_stats, successful_data
    
    def create_visualizations(self, successful_data):
        """
        Create visualizations for processing time analysis.
        """
        if successful_data is None or len(successful_data) == 0:
            print("❌ No successful data to visualize")
            return
            
        print(f"\n📈 Creating visualizations...")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('LLM Processing Time Analysis', fontsize=16, fontweight='bold')
        
        # 1. Bar plot - Average processing time
        ax1 = axes[0, 0]
        avg_times = successful_data.groupby('model')['processing_time_seconds'].mean().sort_values()
        bars = ax1.bar(avg_times.index, avg_times.values, alpha=0.7)
        ax1.set_title('Average Processing Time by Model', fontweight='bold')
        ax1.set_ylabel('Average Time (seconds)')
        ax1.set_xlabel('Model')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, avg_times.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{value:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        # 2. Box plot - Distribution of processing times
        ax2 = axes[0, 1]
        successful_data.boxplot(column='processing_time_seconds', by='model', ax=ax2)
        ax2.set_title('Processing Time Distribution by Model', fontweight='bold')
        ax2.set_ylabel('Processing Time (seconds)')
        ax2.set_xlabel('Model')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # 3. Histogram - Overall time distribution
        ax3 = axes[1, 0]
        for model in successful_data['model'].unique():
            model_data = successful_data[successful_data['model'] == model]['processing_time_seconds']
            ax3.hist(model_data, alpha=0.6, label=model, bins=20)
        ax3.set_title('Processing Time Distribution (All Models)', fontweight='bold')
        ax3.set_xlabel('Processing Time (seconds)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        
        # 4. Model comparison stats
        ax4 = axes[1, 1]
        stats_data = successful_data.groupby('model')['processing_time_seconds'].agg(['mean', 'median', 'std'])
        x_pos = np.arange(len(stats_data))
        width = 0.25
        
        ax4.bar(x_pos - width, stats_data['mean'], width, label='Mean', alpha=0.8)
        ax4.bar(x_pos, stats_data['median'], width, label='Median', alpha=0.8)
        ax4.bar(x_pos + width, stats_data['std'], width, label='Std Dev', alpha=0.8)
        
        ax4.set_title('Statistical Comparison by Model', fontweight='bold')
        ax4.set_ylabel('Time (seconds)')
        ax4.set_xlabel('Model')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(stats_data.index, rotation=45, ha='right')
        ax4.legend()
        
        plt.tight_layout()
        
        # Save the plot
        output_file = self.results_dir / "visualizations" / "processing_time_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   📊 Visualization saved to: {output_file}")
        
        plt.show()
    
    def clean_claude_question(self, question_text):
        """
        Clean up Claude questions to extract only the actual question,
        removing analysis and explanations.
        """
        # Remove leading/trailing quotes and whitespace
        question = question_text.strip().strip('"\'')
        
        # Look for patterns that indicate where the actual question ends
        # Claude often puts questions in quotes, followed by analysis
        
        # Method 1: Extract text within quotes (most common)
        quote_match = re.match(r'^["\']([^"\']+)["\']', question)
        if quote_match:
            return quote_match.group(1).strip()
        
        # Method 2: Look for text before "This question:" or similar analysis markers
        analysis_markers = [
            "This question:",
            "This question is appropriate because",
            "Rationale for this question:",
            "This open-ended question:",
            "Alternative follow-up",
            "The question is",
            "Would you like me to"
        ]
        
        for marker in analysis_markers:
            if marker in question:
                question = question.split(marker)[0].strip()
                break
        
        # Method 3: Remove numbered list explanations (- bullet points)
        if "\n-" in question:
            question = question.split("\n-")[0].strip()
        
        # Method 4: Look for the pattern "?" followed by analysis
        if "?" in question:
            # Find the first question mark and keep everything up to it
            question_parts = question.split("?")
            if len(question_parts) > 1:
                potential_question = question_parts[0] + "?"
                # Check if what follows looks like analysis (contains certain keywords)
                remainder = "?".join(question_parts[1:])
                analysis_keywords = ["This", "The question", "Rationale", "Alternative", "because", "appropriate"]
                if any(keyword in remainder for keyword in analysis_keywords):
                    question = potential_question
        
        # Final cleanup
        question = question.strip().strip('"\'')
        
        # Remove any remaining newlines and extra spaces
        question = re.sub(r'\s+', ' ', question).strip()
        
        return question

    def extract_questions_by_model(self):
        """
        Extract generated questions for each model and save to separate files.
        """
        if not self.data:
            print("❌ No data loaded")
            return
            
        print(f"\n📝 Extracting questions by model...")
        
        extraction_stats = {}
        
        for model_name, df in self.data.items():
            # Filter successful records
            successful_records = df[df['success'] == True]
            
            if len(successful_records) == 0:
                print(f"   ⚠️  {model_name}: No successful records found")
                continue
                
            # Extract questions
            questions = successful_records['generated_question'].tolist()
            
            # Clean up questions based on model
            if model_name.lower() == 'claude':
                questions = [self.clean_claude_question(q) for q in questions]
                print(f"   🧹 Applied Claude-specific cleaning")
            
            # Save to text file
            output_file = self.results_dir / "extracted_questions" / f"questions_{model_name}.txt"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Generated Questions from {model_name.upper()}\n")
                f.write("="*50 + "\n")
                f.write(f"Total questions: {len(questions)}\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for i, question in enumerate(questions, 1):
                    # Clean up the question (remove quotes if present)
                    clean_question = question.strip().strip('"\'')
                    f.write(f"{i:3d}. {clean_question}\n\n")
            
            extraction_stats[model_name] = len(questions)
            print(f"   ✅ {model_name}: {len(questions)} questions → {output_file}")
        
        return extraction_stats
    
    def generate_summary_report(self, time_stats, extraction_stats):
        """
        Generate a comprehensive summary report.
        """
        print(f"\n📋 Generating summary report...")
        
        report_file = self.results_dir / "analysis_reports" / "analysis_summary.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("LLM COUNSELING QUESTIONS ANALYSIS REPORT\n")
            f.write("="*50 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("MODELS ANALYZED\n")
            f.write("-"*20 + "\n")
            for model in self.data.keys():
                total_records = len(self.data[model])
                successful = len(self.data[model][self.data[model]['success'] == True])
                success_rate = (successful / total_records * 100) if total_records > 0 else 0
                f.write(f"• {model}: {successful}/{total_records} records ({success_rate:.1f}% success)\n")
            
            f.write(f"\nPROCESSING TIME STATISTICS\n")
            f.write("-"*30 + "\n")
            if time_stats is not None:
                f.write(time_stats.to_string())
            
            f.write(f"\n\nQUESTION EXTRACTION SUMMARY\n")
            f.write("-"*30 + "\n")
            if extraction_stats:
                for model, count in extraction_stats.items():
                    f.write(f"• {model}: {count} questions extracted\n")
            
            f.write(f"\nFILES GENERATED\n")
            f.write("-"*15 + "\n")
            f.write(f"📈 visualizations/\n")
            f.write(f"  • processing_time_analysis.png - Speed comparison charts\n")
            f.write(f"  • question_length_analysis.png - Word/character count analysis\n")
            f.write(f"  • question_starters_analysis.png - First two words analysis\n")
            f.write(f"📝 extracted_questions/\n")
            for model in extraction_stats.keys() if extraction_stats else []:
                f.write(f"  • questions_{model}.txt - Clean extracted questions\n")
            f.write(f"📋 analysis_reports/\n")
            f.write(f"  • analysis_summary.txt - This comprehensive report\n")
            f.write(f"📊 csv_data/\n")
            f.write(f"  • *_basic.csv files - Source data for analysis\n")
            f.write(f"🗂️ raw_outputs/\n")
            f.write(f"  • *.json files - Original LLM processing outputs\n")
        
        print(f"   📄 Summary report saved to: {report_file}")
    
    def analyze_question_lengths(self):
        """
        Analyze and compare question lengths across models.
        """
        if not self.data:
            print("❌ No data loaded")
            return
            
        print(f"\n📏 Analyzing question lengths...")
        
        # Collect length data for all models
        length_data = []
        
        for model_name, df in self.data.items():
            successful_records = df[df['success'] == True]
            
            for question in successful_records['generated_question']:
                # Clean the question first
                if model_name.lower() == 'claude':
                    clean_question = self.clean_claude_question(question)
                else:
                    clean_question = question.strip().strip('"\'')
                
                # Calculate different length metrics
                word_count = len(clean_question.split())
                char_count = len(clean_question)
                
                length_data.append({
                    'model': model_name,
                    'word_count': word_count,
                    'char_count': char_count,
                    'question': clean_question
                })
        
        length_df = pd.DataFrame(length_data)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Question Length Analysis Across Models', fontsize=16, fontweight='bold')
        
        # 1. Word count comparison - Box plot
        ax1 = axes[0, 0]
        length_df.boxplot(column='word_count', by='model', ax=ax1)
        ax1.set_title('Word Count Distribution by Model', fontweight='bold')
        ax1.set_ylabel('Word Count')
        ax1.set_xlabel('Model')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # 2. Average word count - Bar chart
        ax2 = axes[0, 1]
        avg_words = length_df.groupby('model')['word_count'].mean().sort_values()
        bars = ax2.bar(avg_words.index, avg_words.values, alpha=0.7, color='skyblue')
        ax2.set_title('Average Word Count by Model', fontweight='bold')
        ax2.set_ylabel('Average Word Count')
        ax2.set_xlabel('Model')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, avg_words.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Character count comparison - Box plot
        ax3 = axes[1, 0]
        length_df.boxplot(column='char_count', by='model', ax=ax3)
        ax3.set_title('Character Count Distribution by Model', fontweight='bold')
        ax3.set_ylabel('Character Count')
        ax3.set_xlabel('Model')
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # 4. Length statistics comparison
        ax4 = axes[1, 1]
        stats_data = length_df.groupby('model')['word_count'].agg(['mean', 'median', 'std']).round(1)
        x_pos = np.arange(len(stats_data))
        width = 0.25
        
        ax4.bar(x_pos - width, stats_data['mean'], width, label='Mean', alpha=0.8)
        ax4.bar(x_pos, stats_data['median'], width, label='Median', alpha=0.8)
        ax4.bar(x_pos + width, stats_data['std'], width, label='Std Dev', alpha=0.8)
        
        ax4.set_title('Word Count Statistics by Model', fontweight='bold')
        ax4.set_ylabel('Word Count')
        ax4.set_xlabel('Model')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(stats_data.index, rotation=45, ha='right')
        ax4.legend()
        
        plt.tight_layout()
        
        # Save the plot
        output_file = self.results_dir / "visualizations" / "question_length_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   📊 Length analysis saved to: {output_file}")
        
        # Print statistics
        print(f"\n📊 Question Length Statistics:")
        stats_summary = length_df.groupby('model')[['word_count', 'char_count']].agg(['mean', 'median', 'std']).round(2)
        print(stats_summary.to_string())
        
        plt.show()
        
        return length_df
    
    def analyze_question_starters(self):
        """
        Analyze the first two words of questions for each model.
        """
        if not self.data:
            print("❌ No data loaded")
            return
            
        print(f"\n🎯 Analyzing question starters (first two words)...")
        
        # Collect starter data for all models
        starter_data = {}
        
        for model_name, df in self.data.items():
            successful_records = df[df['success'] == True]
            starters = []
            
            for question in successful_records['generated_question']:
                # Clean the question first
                if model_name.lower() == 'claude':
                    clean_question = self.clean_claude_question(question)
                else:
                    clean_question = question.strip().strip('"\'')
                
                # Extract first two words
                words = clean_question.split()
                if len(words) >= 2:
                    first_two = f"{words[0]} {words[1]}"
                    starters.append(first_two)
            
            # Count frequency of starters
            starter_counts = Counter(starters)
            starter_data[model_name] = starter_counts
        
        # Create visualizations for each model
        n_models = len(starter_data)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Question Starters Analysis: First Two Words by Model', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()  # Make it easier to iterate
        
        for i, (model_name, starter_counts) in enumerate(starter_data.items()):
            if i >= 4:  # Only show first 4 models
                break
                
            ax = axes[i]
            
            # Get top 10 most common starters
            top_starters = starter_counts.most_common(10)
            
            if top_starters:
                starters, counts = zip(*top_starters)
                
                # Create horizontal bar chart for better readability
                bars = ax.barh(range(len(starters)), counts, alpha=0.7)
                ax.set_yticks(range(len(starters)))
                ax.set_yticklabels(starters)
                ax.set_xlabel('Frequency')
                ax.set_title(f'{model_name.upper()}\nTop Question Starters', fontweight='bold')
                
                # Add value labels
                for bar, count in zip(bars, counts):
                    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                           str(count), ha='left', va='center', fontweight='bold')
                
                # Invert y-axis to show most common at top
                ax.invert_yaxis()
        
        # Hide unused subplots if less than 4 models
        for i in range(len(starter_data), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save the plot
        output_file = self.results_dir / "visualizations" / "question_starters_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   📊 Question starters analysis saved to: {output_file}")
        
        # Print detailed statistics
        print(f"\n📊 Question Starters Summary:")
        for model_name, starter_counts in starter_data.items():
            total_questions = sum(starter_counts.values())
            unique_starters = len(starter_counts)
            top_3 = starter_counts.most_common(3)
            
            print(f"\n{model_name.upper()}:")
            print(f"  Total questions: {total_questions}")
            print(f"  Unique starters: {unique_starters}")
            print(f"  Top 3 starters:")
            for starter, count in top_3:
                percentage = (count / total_questions) * 100
                print(f"    '{starter}': {count} times ({percentage:.1f}%)")
        
        plt.show()
        
        return starter_data

    def run_full_analysis(self):
        """
        Run the complete analysis pipeline.
        """
        print("🤖 Starting LLM CSV Analysis Pipeline...")
        print(f"📁 Results directory: {self.results_dir}")
        
        # Step 1: Load data
        if not self.load_csv_files():
            print("❌ Failed to load CSV files. Exiting.")
            return
        
        # Step 2: Analyze processing times
        time_stats, successful_data = self.analyze_processing_times()
        
        # Step 3: Create visualizations
        self.create_visualizations(successful_data)
        
        # Step 4: Analyze question lengths
        length_data = self.analyze_question_lengths()
        
        # Step 5: Analyze question starters
        starter_data = self.analyze_question_starters()
        
        # Step 6: Extract questions
        extraction_stats = self.extract_questions_by_model()
        
        # Step 7: Generate summary report
        self.generate_summary_report(time_stats, extraction_stats)
        
        print(f"\n🎉 Analysis complete!")
        print(f"📊 Check the results directory for outputs: {self.results_dir}")

def main():
    """
    Main function to run the analysis.
    """
    # Initialize analyzer
    analyzer = LLMAnalyzer()
    
    # Run full analysis
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main() 