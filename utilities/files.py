import os


class File:

    @classmethod
    def replace_path(cls, old_path: str, new_path: str) -> None:
        """Replace the file at old_path with file at new path. Rename new_path to old_path."""
        try:
            os.remove(old_path)
            os.rename(new_path, old_path)
        except:
            print(f"Error attemping to replace {old_path} with {new_path}.")
        assert os.path.exists(old_path)
