import argparse
import subprocess
import sys


def run_script(script):
    try:
        subprocess.run(["./" + script], check=True)
        print(f"Script {script} executed successfully.")
    except subprocess.CalledProcessError:
        print(f"Error executing {script}.")


def add_execution_permissions(scripts):
    for script in scripts:
        try:
            subprocess.run(["chmod", "+x", script], check=True)
            print(f"File {script} is now executable.")
        except subprocess.CalledProcessError:
            print(f"Error changing permissions for {script}.")


def main():
    scripts = [
        "setup/install_dependencies.sh", "setup/download_statvu_dataset.sh", "setup/download_videos.sh"
    ]
    add_execution_permissions(scripts)

    parser = argparse.ArgumentParser(description="Script Runner")
    parser.add_argument(
        "--run",
        choices=["all", "vids", "data", "deps", "help"],
        default="all",
        help="Specify the type of script(s) to run."
    )

    args, _ = parser.parse_known_args()
    if args.run == "all":
        for script in scripts:
            run_script(script)
    elif args.run == "vids":
        run_script("setup/download_videos.sh")
    elif args.run == "data":
        run_script("setup/download_statvu_dataset.sh")
    elif args.run == "deps":
        run_script("setup/install_dependencies.sh")
    elif args.run == "help":
        print(parser.format_help())
    else:
        print("Invalid argument. Use --help to see available options.")

    print("Setup complete.")


if __name__ == "__main__":
    main()
