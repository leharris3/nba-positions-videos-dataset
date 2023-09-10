import os
import shutil


def split_files_into_subdirectories(source_folder):

    assert os.path.exists(source_file_path)
    total_files = len(os.listdir(source_folder))

    # break up files into batches of 500 or less
    i, j = 1, 0  # change
    files_remaining = total_files
    subdir_index = 1  # change
    max_files_per_subdirectory = 50
    subdirectory_count = 0
    current_subdirectory = None

    while files_remaining > 0:
        if files_remaining > 500:
            j = i + 500
        else:
            j = i + files_remaining
        files = os.listdir[i: j]

        for file_name in files:
            # Create a new subdirectory if needed
            if subdirectory_count == 0 or subdirectory_count >= max_files_per_subdirectory:
                subdirectory_count = 0
                new_folder = os.path.join(
                    source_folder, f"subdir_{subdir_index}")
                os.makedirs(new_folder, exist_ok=False)
                subdir_index += 1

            source_file_path = os.path.join(source_folder, file_name)
            destination_file_path = os.path.join(new_folder, file_name)

            shutil.move(source_file_path, destination_file_path)
            assert os.path.exists(destination_file_path)
            assert not os.path.exists(source_file_path)

            print(f"Moved {file_name} to {current_subdirectory}")
            subdirectory_count += 1
            files_remaining -= 1
