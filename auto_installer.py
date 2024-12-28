import subprocess
import sys

def install_requirements(requirements_file='requirements.txt'):
    try:
        # Read the requirements.txt file
        with open(requirements_file, 'r') as file:
            lines = file.readlines()

        # Loop through each line in the requirements.txt file
        for line in lines:
            # Skip empty lines or comments
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Check if the line contains `--extra-index-url`
            if '--extra-index-url' in line:
                # Extract the package and the extra index URL
                parts = line.split(' --extra-index-url ')
                package = parts[0]
                extra_index_url = parts[1]
                # Install the package with the extra index URL
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '--extra-index-url', extra_index_url])
            else:
                # Install the package normally
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', line])
        
        print("All required packages installed successfully!")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the function to install requirements
install_requirements()
