#!/usr/bin/env python3
"""
Setup script for downloading the chest X-ray pneumonia dataset from Kaggle
"""

import os
import zipfile
import subprocess
import sys

def setup_kaggle_credentials():
    """
    Guide user through Kaggle API setup
    """
    print("=== Kaggle API Setup ===")
    print("To download the dataset, you need to set up Kaggle API credentials.")
    print("\nSteps:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Scroll to 'API' section and click 'Create New API Token'")
    print("3. This downloads kaggle.json file")
    print("4. Place kaggle.json in your Kaggle config directory:")
    
    # Show the appropriate path based on OS
    if os.name == 'nt':  # Windows
        kaggle_dir = os.path.expanduser("~/.kaggle")
        print(f"   Windows: {kaggle_dir}")
    else:  # Linux/Mac
        kaggle_dir = os.path.expanduser("~/.kaggle")
        print(f"   Linux/Mac: {kaggle_dir}")
    
    print(f"\n5. Make sure the directory exists and kaggle.json is there")
    print(f"6. Set file permissions (Linux/Mac): chmod 600 ~/.kaggle/kaggle.json")
    
    return kaggle_dir

def download_dataset():
    """
    Download and extract the chest X-ray pneumonia dataset
    """
    try:
        print("\n=== Downloading Dataset ===")
        
        # Download the dataset
        result = subprocess.run([
            "kaggle", "datasets", "download", 
            "-d", "paultimothymooney/chest-xray-pneumonia"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error downloading dataset: {result.stderr}")
            return False
        
        print("Dataset downloaded successfully!")
        
        # Extract the dataset
        zip_file = "chest-xray-pneumonia.zip"
        if os.path.exists(zip_file):
            print(f"Extracting {zip_file} to 'paed' directory...")
            
            # Create paed directory if it doesn't exist
            os.makedirs("paed", exist_ok=True)
            
            # Extract the zip file
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall("paed")
            
            print("Dataset extracted successfully!")
            
            # Show directory structure
            print("\n=== Directory Structure ===")
            for root, dirs, files in os.walk("paed"):
                level = root.replace("paed", "").count(os.sep)
                indent = " " * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = " " * 2 * (level + 1)
                for file in files[:3]:  # Show first 3 files only
                    print(f"{subindent}{file}")
                if len(files) > 3:
                    print(f"{subindent}... and {len(files)-3} more files")
            
            # Clean up zip file
            os.remove(zip_file)
            print(f"Cleaned up {zip_file}")
            
            return True
        else:
            print(f"Zip file {zip_file} not found after download")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("Chest X-Ray Pneumonia Dataset Setup")
    print("=" * 40)
    
    # Check if kaggle is installed
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
        print("✓ Kaggle CLI is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ Kaggle CLI not found. Please install it first:")
        print("  uv pip install kaggle")
        return
    
    # Setup credentials
    kaggle_dir = setup_kaggle_credentials()
    
    # Check if credentials exist
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    if not os.path.exists(kaggle_json):
        print(f"\n⚠️  Kaggle credentials not found at {kaggle_json}")
        print("Please set up your Kaggle API credentials first.")
        return
    
    print(f"✓ Found Kaggle credentials at {kaggle_json}")
    
    # Download and extract dataset
    if download_dataset():
        print("\n🎉 Setup complete!")
        print("You can now run: python accuracy_test.py")
    else:
        print("\n❌ Setup failed. Please check the errors above.")

if __name__ == "__main__":
    main() 