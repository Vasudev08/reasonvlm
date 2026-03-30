import os
import matplotlib.pyplot as plt
from PIL import Image
from datasets import load_dataset

def download_dynamath(split='sample_variant1'):
    """
    Downloads the DynaMath_Sample dataset from HuggingFace.
    Default split is 'sample_variant1'.
    """
    print(f"Downloading DynaMath/DynaMath_Sample (split: {split})...")
    dataset = load_dataset("DynaMath/DynaMath_Sample", split=split)
    print("Download complete.")
    return dataset

def filter_by_subject(dataset, subject_name='analytic geometry'):
    """
    Filters the dataset to only include items of a specific subject.
    """
    filtered = dataset.filter(lambda x: x['subject'] == subject_name)
    print(f"Filtered dataset to {len(filtered)} items with subject '{subject_name}'.")
    return filtered

def get_item_data(item):
    """
    Extracts the question, image data, answer type, and correct answer from a dataset item.
    Returns:
        tuple: (question_text, decoded_image, answer_type, correct_answer)
    """
    question = item.get('question', 'No question text')
    decoded_img = item.get('decoded_image', item.get('image', None))
    # answer_type is usually 'float' or 'multiple choice' in DynaMath
    answer_type = item.get('answer_type', 'free-form')
    correct_answer = item.get('ground_truth', 'N/A')
    return question, decoded_img, answer_type, correct_answer

def display_item(item):
    """
    Displays the image and question using matplotlib.
    """
    question, img, answer_type, correct_answer = get_item_data(item)
    
    if img is None:
        print("No image found for this item.")
        print(f"Question: {question}")
        return

    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.title('\n'.join([question[i:i+80] for i in range(0, len(question), 80)]), fontsize=10)
    plt.axis('off')
    plt.show()

    # Print extra context if available
    subject = item.get('subject', 'Unknown')
    category = item.get('category', 'Unknown')
    print(f"Subject: {subject} | Category: {category}")
