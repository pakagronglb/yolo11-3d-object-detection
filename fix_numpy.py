#!/usr/bin/env python3
import subprocess
import sys

def run_command(command):
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True)

print("=== NumPy Fix Script ===")

# Check current NumPy version
try:
    import numpy as np
    print(f"Current NumPy version: {np.__version__}")
    
    if np.__version__.startswith('2.'):
        print("Detected NumPy 2.x which is causing compatibility issues.")
        print("Downgrading to NumPy 1.24.3...")
        
        # Uninstall current NumPy
        run_command("pip uninstall -y numpy")
        
        # Install the compatible version
        run_command("pip install numpy==1.24.3")
        
        print("NumPy downgraded successfully to 1.24.3")
    else:
        print("NumPy version is already compatible. No action needed.")
        
except ImportError:
    print("NumPy is not currently installed. Installing NumPy 1.24.3...")
    run_command("pip install numpy==1.24.3")
    print("NumPy 1.24.3 installed successfully")

print("\n=== NumPy Fix Complete ===")
print("You can now try running your application again.") 