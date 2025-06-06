import json
import csv

def extract_supporter_questions_with_feedback(json_file_path):
    """
    Extract supporter questions from ESConv.json and assign the next feedback value to each question.
    Only uses feedback that comes AFTER the question is asked.
    Also includes the 5 preceding utterances as context for each question.
    Returns data grouped by conversation.
    """
    
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    conversations_with_questions = {}
    
    # Process each conversation in the list
    for conversation_index, conversation_obj in enumerate(data):
        if 'dialog' not in conversation_obj:
            continue
            
        conversation_messages = conversation_obj['dialog']
        # Extract the situation for this conversation
        situation = conversation_obj.get('situation', '').strip()
        conversation_questions = []
        
        # Find all supporter messages with Question strategy
        for i, message in enumerate(conversation_messages):
            # Check if this is a supporter message with Question strategy
            if (message.get('speaker') == 'supporter' and 
                message.get('annotation', {}).get('strategy') == 'Question'):
                
                question_text = message.get('content', '').strip()
                
                # Find the next feedback value that comes AFTER this question
                feedback_value = None
                for j in range(i + 1, len(conversation_messages)):
                    msg_feedback = conversation_messages[j].get('annotation', {}).get('feedback', None)
                    if msg_feedback is not None:
                        feedback_value = msg_feedback
                        break
                
                # Extract context: 5 preceding utterances (without situation)
                context = extract_context(conversation_messages, i, context_length=5)
                
                # Add question to this conversation's questions
                conversation_questions.append({
                    'question_text': question_text,
                    'feedback': feedback_value,
                    'message_index': i,
                    'context': context,
                    'situation': situation
                })
        
        # Only add conversations that have questions
        if conversation_questions:
            conversations_with_questions[f"conversation_{conversation_index}"] = {
                'total_questions': len(conversation_questions),
                'situation': situation,
                'questions': conversation_questions
            }
    
    return conversations_with_questions

def extract_context(conversation_messages, question_index, context_length=5):
    """
    Extract the preceding utterances as context for a question.
    
    Args:
        conversation_messages: List of all messages in the conversation
        question_index: Index of the current question
        context_length: Number of preceding messages to include as context
    
    Returns:
        String representation of the dialogue context
    """
    # Get the preceding messages (up to context_length)
    start_index = max(0, question_index - context_length)
    context_messages = conversation_messages[start_index:question_index]
    
    # Format the dialogue context
    dialogue_lines = []
    for msg in context_messages:
        speaker = msg.get('speaker', 'unknown')
        content = msg.get('content', '').strip()
        # Clean up content - remove excessive whitespace and newlines
        content = ' '.join(content.split())
        dialogue_lines.append(f"{speaker}: {content}")
    
    return " | ".join(dialogue_lines)

def save_to_json(conversations_data, output_file='supporter_questions_with_feedback.json'):
    """
    Save the extracted questions to a JSON file.
    """
    with open(output_file, 'w', encoding='utf-8') as jsonfile:
        json.dump(conversations_data, jsonfile, indent=2, ensure_ascii=False)
    
    print(f"Saved data from {len(conversations_data)} conversations to {output_file}")

def print_statistics(conversations_data):
    """
    Print statistics about the extracted questions and feedback distribution.
    """
    # Calculate total questions across all conversations
    total_questions = 0
    all_feedbacks = []
    
    for conv_data in conversations_data.values():
        total_questions += conv_data['total_questions']
        for question in conv_data['questions']:
            all_feedbacks.append(question['feedback'])
    
    print(f"Total conversations with questions: {len(conversations_data)}")
    print(f"Total questions extracted: {total_questions}")
    
    # Count questions by feedback score
    feedback_counts = {}
    questions_with_feedback = 0
    
    for feedback in all_feedbacks:
        if feedback is not None:
            questions_with_feedback += 1
            feedback_counts[feedback] = feedback_counts.get(feedback, 0) + 1
        else:
            feedback_counts['None'] = feedback_counts.get('None', 0) + 1
    
    print(f"Questions with feedback: {questions_with_feedback}")
    print(f"Questions without feedback: {total_questions - questions_with_feedback}")
    
    print("\nFeedback distribution:")
    for feedback, count in sorted(feedback_counts.items()):
        percentage = (count / total_questions) * 100
        print(f"  Feedback {feedback}: {count} questions ({percentage:.1f}%)")

def print_examples(conversations_data, num_examples=3):
    """
    Print example conversations with their questions and context.
    """
    print(f"\nFirst {num_examples} conversation examples:")
    
    conv_keys = list(conversations_data.keys())[:num_examples]
    
    for i, conv_key in enumerate(conv_keys):
        conv_data = conversations_data[conv_key]
        print(f"\n{i+1}. {conv_key} - {conv_data['total_questions']} questions:")
        print(f"   Situation: {conv_data['situation'][:100]}{'...' if len(conv_data['situation']) > 100 else ''}")
        
        # Show first 2 questions from this conversation
        for j, question in enumerate(conv_data['questions'][:2]):
            print(f"   Question {j+1} (Index {question['message_index']}):")
            print(f"     Context: {question['context'][:120]}{'...' if len(question['context']) > 120 else ''}")
            print(f"     Question: {question['question_text']}")
            print(f"     Feedback: {question['feedback']}")
        
        if len(conv_data['questions']) > 2:
            print(f"     ... and {len(conv_data['questions']) - 2} more questions")

def main():
    """
    Main function to extract supporter questions with feedback and context, then save to JSON.
    """
    print("Extracting supporter questions with context from ESConv.json...")
    
    # Extract questions with feedback and context, grouped by conversation
    conversations_data = extract_supporter_questions_with_feedback('ESConv.json')
    
    # Display statistics
    print_statistics(conversations_data)
    
    # Save to JSON
    save_to_json(conversations_data)
    
    # Display examples
    print_examples(conversations_data)

if __name__ == "__main__":
    main() 