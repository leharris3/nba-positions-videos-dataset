import os

def group_files(folder_path):
    
    assert os.path.exists(folder_path)
    files = os.listdir(folder_path)

    for file_name in files:

        file_title = file_name[:-13]
        print(file_title)
        exit()

        new_folder_path = f"{folder_path}/"
        os.mkdir()