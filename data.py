import os
import json
import py7zr
import shutil


class Data:
    """Data object containing player positions for an NBA basketball game."""

    def __init__(self, path: str) -> None:
        if not os.path.exists(path):
            print(f"Error creating data: invalid path at {path}.")
            raise Exception
        if path[-3:] != ".7z" and path[-5:] != ".json":
            print(f"Error creating data: file extension should be .7z or .json.")
            raise Exception
        self.path = os.path.abspath(path)
        self.raw_data = None
        self.is_zipped = False
        if path[-3:] == ".7z":
            self.is_zipped = True
        else:
            self.raw_data = Data.load_json(path)

    def unzip(self, to: str) -> str:
        if not os.path.exists(to):
            print(f"Error: invalid path at {to}.")
            raise Exception
        # extract all .7z files to path
        try:
            temp_zip_path = f"{to}.7z"
            # old data path -> new folder
            shutil.copy(self.path, temp_zip_path)
            with py7zr.SevenZipFile(temp_zip_path, mode='r') as archive:
                archive.extractall(to)
            new_path = ""
            for root, _, files in os.walk(to):
                for file in files:
                    if file.endswith(".json"):
                        new_path = os.path.join(root, file)
            if new_path == "":
                print(
                    f"Error: unzipped .json path not found while unzipping to: {to}.")
            os.remove(temp_zip_path)  # remove temp zip file
            return new_path  # return path of unzipped data
        except:
            print(f"Error extracting .7z files from {self.path} to {to}.")
            raise Exception
        self.raw_data = self.load_json(self.path)

    @classmethod
    def load_json(cls, path: str):
        if not os.path.exists(path):
            print(f"Error: invalid path at {path}.")
            raise Exception
        try:
            with open(path) as file:
                return json.load(file)
        except:
            print(f"Error: could not read in .json data from {path}.")
            raise Exception
