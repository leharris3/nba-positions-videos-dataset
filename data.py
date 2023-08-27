from ast import Dict
import os
import json
import py7zr
import shutil

from typing import List


class Data:
    """Data object containing player positions for an NBA basketball game."""

    def __init__(self, path: str) -> None:
        assert os.path.exists(
            path), f"Error creating data: invalid path at {path}."
        if path[-3:] != ".7z" and path[-5:] != ".json":
            raise Exception(
                f"Error creating data: file extension should be .7z or .json.")
        self.path = os.path.abspath(path)
        self.raw_data = None
        self.is_zipped = True if path[-3:] == ".7z" else False
        if not self.is_zipped:
            self.raw_data = self.load_json(path)

    def get_moments(self):
        """Return all moments from a SportVU data file. """

        assert self.raw_data, f"Error: data file at {self.path} is None."
        return [moment for event in self.raw_data["events"] for moment in event["moments"]]

    def get_frames_moments_mapped(self):
        """Returns moment at each frame in a video."""

        moments = [moment for moment in self.get_moments() if len(
            moment) > 6 and moment[6] != -1]
        assert moments, f"Error: no timestamps information for data at {self.path}."
        return {str(moment[6]): moment for moment in moments}

    def get_data(self):
        """Data getter."""

        assert self.raw_data
        return self.raw_data

    def unzip(self, to: str):
        """Unzip a .7z file and return path to .json if found."""

        if not os.path.exists(to):
            raise Exception(f"Error: invalid path at {to}.")
        try:
            temp_zip_path = f"{to}/zip.7z"
            shutil.copy(self.path, temp_zip_path)
            with py7zr.SevenZipFile(temp_zip_path, 'r') as archive:
                archive.extractall(to)

            new_path = ""
            files = os.listdir(to)
            for file in files:
                if file.endswith(".json"):
                    new_path = os.path.join(to, file)
                    break
            os.remove(temp_zip_path)
        except:
            raise Exception(
                f"Error extracting .7z files from {self.path} to {to}.")
        self.raw_data = self.load_json(new_path)
        return new_path if new_path else Exception(f"Error: unzipped .json path not found while unzipping to: {to}.")

    @classmethod
    def load_json(cls, path: str):
        assert os.path.exists(path), f"Error: invalid path at {path}."
        try:
            with open(path) as file:
                return json.load(file)
        except:
            raise Exception(
                f"Error: could not read in .json data from {path}.")
