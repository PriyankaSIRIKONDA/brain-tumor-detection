#!/usr/bin/env python3
"""
GitHub Repository Setup Script
This script helps you set up your GitHub repository for the brain tumor detection project.
"""

import os
import subprocess
import sys

def run_command(command):
    """Run a command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    print("="*60)
    print("GitHub Repository Setup for Brain Tumor Detection")
    print("="*60)
    
    print("\nThis script will help you set up your GitHub repository.")
    print("Make sure you have git installed and configured.")
    
    # Check if git is installed
    success, stdout, stderr = run_command("git --version")
    if not success:
        print("❌ Git is not installed. Please install git first.")
        return
    
    print(f"✅ Git found: {stdout.strip()}")
    
    # Check if this is already a git repository
    if os.path.exists(".git"):
        print("✅ This directory is already a git repository.")
    else:
        print("\n1. Initializing git repository...")
        success, stdout, stderr = run_command("git init")
        if success:
            print("✅ Git repository initialized.")
        else:
            print(f"❌ Failed to initialize git: {stderr}")
            return
    
    # Add all files
    print("\n2. Adding files to git...")
    success, stdout, stderr = run_command("git add .")
    if success:
        print("✅ Files added to git.")
    else:
        print(f"❌ Failed to add files: {stderr}")
        return
    
    # Initial commit
    print("\n3. Making initial commit...")
    success, stdout, stderr = run_command('git commit -m "Initial commit: Brain Tumor Detection Project"')
    if success:
        print("✅ Initial commit created.")
    else:
        print(f"❌ Failed to commit: {stderr}")
        return
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Go to https://github.com/PriyankaSIRIKONDA")
    print("2. Click 'New repository'")
    print("3. Name it 'brain-tumor-detection'")
    print("4. Don't initialize with README (we already have one)")
    print("5. Copy the repository URL")
    print("6. Run these commands:")
    print("   git remote add origin https://github.com/PriyankaSIRIKONDA/brain-tumor-detection.git")
    print("   git branch -M main")
    print("   git push -u origin main")
    print("\nYour repository will be ready!")
    print("="*60)

if __name__ == "__main__":
    main() 