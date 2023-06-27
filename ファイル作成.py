import os
import shutil

def merge_folders(Datasets_kaiki_20230224, Datasets_kaiki_20230224_2tika, Datasets_kaiki):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for root, dirs, files in os.walk(source_folder1):
        for file in files:
            if file in os.listdir(source_folder2):
                src_file1 = os.path.join(root, file)
                src_file2 = os.path.join(source_folder2, file)
                dst_file = os.path.join(output_folder, file)
                # Check if the file already exists in the output folder
                if os.path.exists(dst_file):
                    print(f"File {file} already exists in the output folder.")
                else:
                    # Copy the file to the output folder
                    shutil.copy2(src_file1, dst_file)
                    # Append the contents of the second file to the first file
                    with open(dst_file, 'ab') as f:
                        with open(src_file2, 'rb') as g:
                            shutil.copyfileobj(g, f)
                            
    # Copy any remaining files in the second source folder to the output folder
    for root, dirs, files in os.walk(source_folder2):
        for file in files:
            if file not in os.listdir(output_folder):
                src_file = os.path.join(root, file)
                dst_file = os.path.join(output_folder, file)
                shutil.copy2(src_file, dst_file)
