#!/usr/bin/env python3
import subprocess
import sys
import os

def run_command(command):
    print(f"Running: {command}")
    subprocess.run(command, shell=True)

print("=== Model Fix Script ===")

# Install compatible versions of key packages
print("\nInstalling compatible versions of key packages...")
run_command("pip install numpy==1.26.0")
run_command("pip install ultralytics==8.0.0")
run_command("pip install transformers==4.38.0")
run_command("pip install python-dotenv==1.0.0")

# Check if the YOLO model file exists
print("\nChecking for YOLO model file...")
model_file = "yolo11n.pt"
if os.path.exists(model_file):
    print(f"✓ YOLO model file found: {model_file} ({os.path.getsize(model_file) / (1024*1024):.1f} MB)")
else:
    print(f"✗ YOLO model file NOT found: {model_file}")
    print("Please make sure the YOLO model file is in the project directory.")

# Check if the .env file exists with HF_TOKEN
print("\nChecking for Hugging Face token...")
env_file = ".env"
if os.path.exists(env_file):
    with open(env_file, 'r') as f:
        content = f.read()
    if "HF_TOKEN" in content:
        print("✓ HF_TOKEN found in .env file")
    else:
        print("✗ HF_TOKEN not found in .env file")
        token = input("Enter your Hugging Face token: ").strip()
        with open(env_file, 'w') as f:
            f.write(f"HF_TOKEN={token}")
        print("✓ HF_TOKEN added to .env file")
else:
    print("✗ .env file not found")
    token = input("Enter your Hugging Face token: ").strip()
    with open(env_file, 'w') as f:
        f.write(f"HF_TOKEN={token}")
    print("✓ .env file created with HF_TOKEN")

print("\n=== Fix Complete ===")
print("You can now try running your application again with: python run.py") 