import os
import cv2

def filter_and_save_faces(annotations_file, image_dir, output_dir, min_faces=3, max_faces=5, margin=0.2):
    """
    Filters images with mostly good-quality, unhidden faces and saves individual faces with extra background.

    Args:
        annotations_file (str): Path to the WIDERFace annotations file.
        image_dir (str): Directory containing WIDERFace images.
        output_dir (str): Directory to save cropped face images.
        min_faces (int): Minimum number of faces required in an image.
        max_faces (int): Maximum number of faces required in an image.
        margin (float): Margin to add around bounding boxes as a percentage of box dimensions.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    selected_images = []
    with open(annotations_file, 'r') as file:
        lines = file.readlines()
        idx = 0
        while idx < len(lines):
            # Parse the image path
            image_name = lines[idx].strip()
            idx += 1

            # Validate the number of faces
            try:
                num_faces = int(lines[idx].strip())
            except ValueError:
                print(f"Skipping malformed entry for image: {image_name}")
                continue
            idx += 1

            face_annotations = lines[idx: idx + num_faces]
            idx += num_faces

            # Process face annotations
            valid_faces = []
            for annotation in face_annotations:
                try:
                    # Parse bounding box and attributes
                    x, y, w, h, blur, expression, occlusion = map(int, annotation.split()[:7])
                    if blur == 0 and occlusion == 0 and expression == 0:
                        valid_faces.append((x, y, w, h))
                except ValueError:
                    print(f"Skipping malformed face annotation in image: {image_name}")
                    continue

            # Add images with valid face counts
            if min_faces <= len(valid_faces) <= max_faces:
                selected_images.append((image_name, valid_faces))

    # Process and save face crops
    for image_name, faces in selected_images:
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Warning: Could not read {image_path}")
            continue

        for i, (x, y, w, h) in enumerate(faces):
            # Add margin to the bounding box
            x_margin = int(w * margin)
            y_margin = int(h * margin)

            x_start = max(x - x_margin, 0)
            y_start = max(y - y_margin, 0)
            x_end = min(x + w + x_margin, image.shape[1])
            y_end = min(y + h + y_margin, image.shape[0])

            # Crop the face
            face_crop = image[y_start:y_end, x_start:x_end]

            # Save the cropped face
            face_filename = f"{os.path.splitext(image_name)[0].replace('/', '_')}_face{i + 1}.jpg"
            face_output_path = os.path.join(output_dir, face_filename)
            cv2.imwrite(face_output_path, face_crop)

    print(f"Processed and saved faces to {output_dir}")


annotations_file = './data/WIDER/wider_face_split/wider_face_train_bbx_gt.txt'
image_dir = './data/WIDER/WIDER_train/images'
output_dir = './data/WIDER/cropped_faces'

filter_and_save_faces(
    annotations_file=annotations_file,
    image_dir=image_dir,
    output_dir=output_dir,
    min_faces=3,
    max_faces=5,
    margin=0.2
)

