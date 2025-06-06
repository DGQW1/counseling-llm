import json

def flatten_questions_to_individual_objects(input_file, output_file):
    """
    Read the conversation-grouped JSON file and create individual question objects
    with only context and situation fields.
    
    Args:
        input_file: Path to the input JSON file with conversation structure
        output_file: Path to the output JSON file with flattened question objects
    """
    
    # Read the conversation-grouped data
    with open(input_file, 'r', encoding='utf-8') as file:
        conversations_data = json.load(file)
    
    # Create list to store individual question objects
    individual_questions = []
    
    # Process each conversation
    for conversation_id, conversation_data in conversations_data.items():
        # Extract questions from this conversation
        for question in conversation_data['questions']:
            # Create individual question object with only context and situation
            question_obj = {
                'context': question['context'],
                'situation': question['situation']
            }
            individual_questions.append(question_obj)
    
    # Save the flattened questions to new JSON file
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(individual_questions, file, indent=2, ensure_ascii=False)
    
    return individual_questions

def print_statistics(questions_list):
    """
    Print statistics about the flattened questions.
    """
    print(f"Total individual question objects created: {len(questions_list)}")
    
    # Count questions with non-empty context
    questions_with_context = sum(1 for q in questions_list if q['context'].strip())
    questions_without_context = len(questions_list) - questions_with_context
    
    print(f"Questions with dialogue context: {questions_with_context}")
    print(f"Questions without dialogue context (conversation starters): {questions_without_context}")
    
    # Count questions with situation
    questions_with_situation = sum(1 for q in questions_list if q['situation'].strip())
    print(f"Questions with situation: {questions_with_situation}")

def print_examples(questions_list, num_examples=5):
    """
    Print example question objects.
    """
    print(f"\nFirst {num_examples} question examples:")
    
    for i, question in enumerate(questions_list[:num_examples]):
        print(f"\n{i+1}. Question Object:")
        print(f"   Situation: {question['situation'][:80]}{'...' if len(question['situation']) > 80 else ''}")
        print(f"   Context: {question['context'][:100]}{'...' if len(question['context']) > 100 else ''}")

def main():
    """
    Main function to flatten questions from conversation structure to individual objects.
    """
    input_file = 'supporter_questions_with_feedback.json'
    output_file = 'individual_questions.json'
    
    print("Flattening questions from conversation structure to individual objects...")
    
    # Flatten the questions
    questions_list = flatten_questions_to_individual_objects(input_file, output_file)
    
    # Print statistics
    print_statistics(questions_list)
    
    # Save confirmation
    print(f"\nSaved {len(questions_list)} individual question objects to {output_file}")
    
    # Print examples
    print_examples(questions_list)

if __name__ == "__main__":
    main() 