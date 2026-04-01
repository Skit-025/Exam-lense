import os

def rename_files(folder_path, prefix="image"):
    # Get all files (ignore directories)
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Sort files to ensure consistent ordering
    files.sort()

    for i, filename in enumerate(files, start=1):
        old_path = os.path.join(folder_path, filename)

        # Extract file extension
        _, ext = os.path.splitext(filename)

        new_name = f"{prefix}{i}{ext}"
        new_path = os.path.join(folder_path, new_name)

        os.rename(old_path, new_path)

    print("Renaming completed.")

# Usage
if __name__ == "__main__":
    folder = r"C:\Users\codes\Desktop\Exam lense\examlense\final_croped"
    rename_files(folder)
