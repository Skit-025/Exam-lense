import os

# Path to my folder
folder_path = r"C:\Users\codes\Desktop\Exam lense\examlense\uploads\Dataset"

# Listing all files in the folder
files = os.listdir(folder_path)

# # Sorting files to ensure consistent order
# files.sort()

# Looping through files and renaming them
for i, filename in enumerate(files, start=1):
    # Extract file extension
    file_ext = os.path.splitext(filename)[1]
    print(file_ext)
    # Create new filename like scan1.jpg, scan2.png, etc.
    new_name = f"scan{i}{file_ext}"
    
    # Full paths
    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, new_name)
    
    # Rename file
    os.rename(old_path, new_path)

print("Renaming complete!")