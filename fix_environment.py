#!/usr/bin/env python3
import os
import sys
import subprocess
import pkg_resources

def print_status(message, success=True):
    icon = "✓" if success else "✗"
    print(f"{icon} {message}")

def run_command(command, display=True):
    if display:
        print(f"\nRunning: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if display:
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.stderr.strip():
            print(f"Errors: {result.stderr.strip()}")
    return result

print("=== Environment Diagnostics and Repair ===")

# Check Python version
print(f"\nPython version: {sys.version}")

# Check pip version
run_command("pip --version")

# Check current packages
print("\nChecking problematic packages:")
try:
    numpy_version = pkg_resources.get_distribution("numpy").version
    print_status(f"NumPy version: {numpy_version}", "2." not in numpy_version)
except pkg_resources.DistributionNotFound:
    print_status("NumPy is not installed", False)

try:
    torch_version = pkg_resources.get_distribution("torch").version
    print_status(f"PyTorch version: {torch_version}")
except pkg_resources.DistributionNotFound:
    print_status("PyTorch is not installed", False)

try:
    ultralytics_version = pkg_resources.get_distribution("ultralytics").version
    print_status(f"Ultralytics version: {ultralytics_version}")
except pkg_resources.DistributionNotFound:
    print_status("Ultralytics is not installed", False)

try:
    transformers_version = pkg_resources.get_distribution("transformers").version
    print_status(f"Transformers version: {transformers_version}")
except pkg_resources.DistributionNotFound:
    print_status("Transformers is not installed", False)

# Fix NumPy issue
print("\n=== Fixing NumPy Issue ===")
if "2." in numpy_version:
    print("Detected NumPy 2.x which is causing compatibility issues.")
    choice = input("Do you want to downgrade NumPy to 1.24.3? (y/n): ").strip().lower()
    if choice == 'y':
        run_command("pip uninstall -y numpy")
        run_command("pip install numpy==1.24.3")
        print_status("NumPy downgraded to 1.24.3")
    else:
        print_status("NumPy downgrade skipped", False)
else:
    print_status("NumPy version is already compatible")

# Check Hugging Face token
print("\n=== Checking Hugging Face Token ===")
token = os.environ.get("HF_TOKEN")
if token:
    masked_token = token[:4] + "..." + token[-4:]
    print_status(f"HF_TOKEN is set: {masked_token}")
else:
    print_status("HF_TOKEN is not set", False)
    env_file = ".env"
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            content = f.read()
        if "HF_TOKEN" in content:
            print_status("Found HF_TOKEN in .env file but it's not loaded in environment")
            from dotenv import load_dotenv
            load_dotenv()
            token = os.environ.get("HF_TOKEN")
            if token:
                masked_token = token[:4] + "..." + token[-4:]
                print_status(f"Successfully loaded HF_TOKEN: {masked_token}")
            else:
                print_status("Failed to load HF_TOKEN from .env file", False)
    else:
        print_status(".env file not found", False)
        create_env = input("Do you want to create a .env file with a Hugging Face token? (y/n): ").strip().lower()
        if create_env == 'y':
            token = input("Enter your Hugging Face token: ").strip()
            with open(env_file, 'w') as f:
                f.write(f"HF_TOKEN={token}")
            print_status(f".env file created with HF_TOKEN")

print("\n=== Environment Check Complete ===")
print("\nRun the debug_models.py script next to test if the models work correctly.") 