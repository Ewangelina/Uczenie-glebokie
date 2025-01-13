import os
import csv

def rename_files_to_numeric(image_dir, csv_file):
    """
    Renames all image files in the directory to numeric names and updates the CSV with new names.
    """
    new_filenames = {}
    counter = 1

    # Rename files to numeric names
    for filename in os.listdir(image_dir):
        old_path = os.path.join(image_dir, filename)
        if os.path.isfile(old_path):
            new_filename = f"{counter}.jpg"
            new_path = os.path.join(image_dir, new_filename)
            os.rename(old_path, new_path)
            new_filenames[filename] = new_filename
            counter += 1

    # Update the CSV with new filenames
    updated_rows = []
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            original_filename = row[0]
            new_filename = new_filenames.get(original_filename)
            if new_filename:
                row[0] = new_filename
                updated_rows.append(row)
            else:
                print(f"Warning: {original_filename} not found in the renamed files.")
    
    # Write the updated rows back to the CSV
    with open(csv_file, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(updated_rows)
    
    print(f"Renamed {len(new_filenames)} files and updated {csv_file}.")

# Example usage
image_dir = "./data/WIDER/selected_faces/male"  # Adjust the path to your image directory
csv_file = "./data/annotations.csv"  # Adjust the path to your CSV file
rename_files_to_numeric(image_dir, csv_file)
