import os
import sys

def add_path_comment_to_python_files(folder_path):
    folder_path = os.path.abspath(folder_path)

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.abspath(os.path.join(root, file))

                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    # Skip if file is empty
                    if not lines:
                        continue

                    first_line_comment = f"# {full_path}\n"

                    # Avoid duplicating the comment
                    if lines[0].strip() == f"# {full_path}":
                        continue

                    # Insert comment at the top
                    lines.insert(0, first_line_comment)

                    with open(full_path, "w", encoding="utf-8") as f:
                        f.writelines(lines)

                    print(f"Updated: {full_path}")

                except Exception as e:
                    print(f"Error processing {full_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python add_path_comment.py /path/to/folder")
        sys.exit(1)

    folder = sys.argv[1]

    if not os.path.isdir(folder):
        print("Invalid folder path.")
        sys.exit(1)

    add_path_comment_to_python_files(folder)