import os

def split_file(file_path, chunk_size=24 * 1024 * 1024):  # 24MB chunk size
    file_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f:
        part_num = 1
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            part_file = f"{file_path}.part{part_num}"
            with open(part_file, 'wb') as chunk_file:
                chunk_file.write(chunk)
            print(f"Created: {part_file}")
            part_num += 1

file_path = "image_classifier_model.h5"  # Change this to your file name
split_file(file_path)
