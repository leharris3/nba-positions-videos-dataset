import argparse
import subprocess
import sys


def run_script(script):
    try:
        subprocess.run(["./" + script], check=True)
        print(f"Script {script} executed successfully.")
    except subprocess.CalledProcessError:
        print(f"Error executing {script}.")


def main():
    scripts_to_execute = [
        "setup/install_dependencies.sh", "setup/download_statvu_dataset.sh", "setup/download_videos.sh"
    ]

    parser = argparse.ArgumentParser(description="Script Runner")
    parser.add_argument(
        "--run",
        choices=["ALL", "VIDEOS", "DATA", "DEP", "help"],
        default="ALL",
        help="Specify the type of script(s) to run."
    )

    args, _ = parser.parse_known_args()

    if args.run == "ALL":
        for script in scripts_to_execute:
            run_script(script)
    elif args.run == "VIDEOS":
        run_script("setup/download_videos.sh")
    elif args.run == "DATA":
        run_script("setup/download_statvu_dataset.sh")
    elif args.run == "DEP":
        run_script("setup/install_dependencies.sh")
    elif args.run == "help":
        print(parser.format_help())
    else:
        print("Invalid argument. Use --help to see available options.")

    print("Setup complete.")


if __name__ == "__main__":
    main()
