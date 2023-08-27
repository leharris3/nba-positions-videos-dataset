import sys


def print_progress(progress):
    progress_bar = "[" + "#" * \
        int(progress * 20) + " " * (20 - int(progress * 20)) + "]"
    sys.stdout.write("\r{} {:.2f}%".format(progress_bar, progress * 100))
    sys.stdout.flush()
