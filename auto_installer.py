import subprocess
import sys
import pkg_resources

def is_package_installed(package):
    try:
        # Use pkg_resources to check if the package is installed
        pkg_resources.get_distribution(package)
        return True
    except pkg_resources.DistributionNotFound:
        return False

def install_requirements(requirements_file='requirements.txt'):
    try:
        # Read the requirements.txt file
        with open(requirements_file, 'r') as file:
            lines = file.readlines()

        for line in lines:
            # Skip empty lines or comments
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Initialize variables
            package = line
            extra_index_url = None
            index_url = None

            # Check if the line contains --extra-index-url or --index-url
            if '--extra-index-url' in line:
                parts = line.split(' --extra-index-url ')
                package = parts[0]
                extra_index_url = parts[1]
            elif '--index-url' in line:
                parts = line.split(' --index-url ')
                package = parts[0]
                index_url = parts[1]

            # Skip installation if the package is already installed
            if is_package_installed(package):
                continue

            # Install the package with the respective index URL if present
            install_command = [sys.executable, '-m', 'pip', 'install', package]
            if extra_index_url:
                install_command.extend(['--extra-index-url', extra_index_url])
            elif index_url:
                install_command.extend(['--index-url', index_url])

            # Run the pip install command and suppress output
            subprocess.check_call(install_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Successfully installed {package}")

        print("All required packages are installed or already present.")

    except Exception as e:
        print(f"An error occurred: {e}")
