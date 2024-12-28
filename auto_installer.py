import subprocess
import sys
import os

def check_and_install():
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("No requirements.txt found.")
        return

    try:
        import pkg_resources
        # Read packages from requirements.txt
        with open("requirements.txt", "r") as file:
            required_packages = {line.strip().lower() for line in file if line.strip()}

        # Get installed packages
        installed_packages = {pkg.key for pkg in pkg_resources.working_set}

        # Determine missing packages
        missing_packages = required_packages - installed_packages

    except ImportError:
        print("pkg_resources not installed. Installing all packages from requirements.txt.")
        missing_packages = required_packages

    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("All required packages installed.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install packages: {e}")
    else:
        print("All packages are already installed.")