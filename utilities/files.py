import os
import json


class File:
    """Some file related helper methods."""

    @classmethod
    def replace_path(cls, replace: str, with_file: str) -> None:
        """Replace file at old_path with file at new_path. Rename new_path to old_path."""

        old_path, new_path = replace, with_file
        try:
            os.remove(old_path)
            os.rename(new_path, old_path)
        except:
            print(f"Error attemping to replace {old_path} with {new_path}.")
        assert os.path.exists(old_path)

    @classmethod
    def load_json(cls, path: str):
        """Load a .json file from a path as dictionary."""

        assert os.path.exists(path), f"Error: invalid path at {path}."
        try:
            with open(path) as file:
                return json.load(file)
        except:
            raise Exception(
                f"Error: could not read in .json data from {path}.")

    @classmethod
    def save_json(cls, data, to: str):
        """Save a dictionary to a path as a .json file."""

        try:
            out_file = open(to, "w")
            json.dump(data, out_file, indent=4)
            assert os.path.exists(to)
        except:
            raise Exception(
                f"Error: could not save modified data file to: {to}.")
