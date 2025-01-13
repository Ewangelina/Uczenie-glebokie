import os

def annotate_from_folders(base_dir, output_file):
    """
    Generates annotations based on folder structure.

    Args:
        base_dir (str): Path to the directory containing `male` and `female` folders.
        output_file (str): Path to save the generated annotations CSV file.
    """
    categories = ['male', 'female']
    annotations = []

    for category in categories:
        category_dir = os.path.join(base_dir, category)
        if not os.path.isdir(category_dir):
            print(f"Warning: Directory {category_dir} does not exist. Skipping.")
            continue

        for image_name in os.listdir(category_dir):
            if image_name.endswith('.jpg'):  # Adjust if your images use a different extension
                annotations.append((image_name, category.capitalize()))  # Capitalize labels for consistency

    # Save annotations to a CSV file
    with open(output_file, 'w') as file:
        file.write("image_name,label\n")
        for image_name, label in annotations:
            file.write(f"{image_name},{label}\n")

    print(f"Annotations saved to {output_file}")

# Example usage
base_dir = './data/WIDER/selected_faces'  # Directory containing `male` and `female` folders
output_file = './data/annotations.csv'  # Path to save the annotations
annotate_from_folders(base_dir, output_file)
