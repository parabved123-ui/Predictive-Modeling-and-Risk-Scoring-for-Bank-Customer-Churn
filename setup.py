#!/usr/bin/env python
"""
Bank Churn Dashboard - Quick Setup Script
Helps users install dependencies and run the application
"""

import os
import sys
import subprocess
import platform

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_step(number, text):
    """Print a step"""
    print(f"\n  ✓ Step {number}: {text}")

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("  ✗ Python 3.8 or higher is required!")
        print(f"  Current version: {sys.version}")
        return False
    print(f"  ✓ Python version: {sys.version.split()[0]}")
    return True

def install_requirements():
    """Install required packages"""
    print_step(2, "Installing dependencies from requirements.txt")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
        print("  ✓ All dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("  ✗ Failed to install dependencies")
        print("  Try running: pip install -r requirements.txt")
        return False

def run_streamlit_app():
    """Run the Streamlit application"""
    print_step(3, "Launching Streamlit application")
    print("\n  Starting the application...\n")
    print("  " + "-"*50)
    print("  The app will open in your browser automatically.")
    print("  If not, visit: http://localhost:8501")
    print("  " + "-"*50 + "\n")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    except KeyboardInterrupt:
        print("\n\n  Application stopped by user.")
    except Exception as e:
        print(f"  ✗ Error running application: {e}")
        return False
    return True

def create_virtual_env():
    """Create a virtual environment"""
    venv_name = "venv"
    
    if platform.system() == "Windows":
        venv_activate = os.path.join(venv_name, "Scripts", "activate.bat")
        activation_cmd = venv_activate
    else:
        venv_activate = os.path.join(venv_name, "bin", "activate")
        activation_cmd = f"source {venv_activate}"
    
    if not os.path.exists(venv_name):
        print_step(1, f"Creating virtual environment '{venv_name}'")
        try:
            subprocess.check_call([sys.executable, "-m", "venv", venv_name])
            print(f"  ✓ Virtual environment created!")
            print(f"\n  To activate the virtual environment, run:")
            print(f"    {activation_cmd}")
            return True
        except subprocess.CalledProcessError:
            print("  ✗ Failed to create virtual environment")
            return False
    else:
        print_step(1, f"Virtual environment '{venv_name}' already exists")
        return True

def main():
    """Main setup function"""
    print_header("🏦 Bank Churn Dashboard - Quick Setup")
    
    # Check Python version
    print_step(0, "Checking Python version")
    if not check_python_version():
        sys.exit(1)
    
    # Check if running from correct directory
    if not os.path.exists("streamlit_app.py"):
        print("\n✗ Error: streamlit_app.py not found in current directory!")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    # Ask about virtual environment
    print("\n" + "="*60)
    print("  Virtual Environment")
    print("="*60)
    use_venv = input("\n  Create/use virtual environment? (recommended) [y/n]: ").lower()
    
    if use_venv == 'y':
        if not create_virtual_env():
            sys.exit(1)
        print("\n  Note: Please activate the virtual environment and run this script again!")
        sys.exit(0)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Run the app
    print_header("🚀 Ready to Launch!")
    print("\n  Configuration Summary:")
    print(f"  • Python: {sys.version.split()[0]}")
    print(f"  • Framework: Streamlit")
    print(f"  • Database: In-memory (synthetic data)")
    print(f"  • Model: XGBoost Classifier")
    print("\n  Features:")
    print("  ✓ Dashboard with key metrics")
    print("  ✓ EDA and exploratory analysis")
    print("  ✓ Model performance metrics")
    print("  ✓ Real-time churn prediction")
    print("  ✓ Customer segmentation and insights")
    
    start = input("\n  Start the application now? [y/n]: ").lower()
    
    if start == 'y':
        run_streamlit_app()
    else:
        print("\n  To start the application later, run:")
        print("    streamlit run streamlit_app.py")
    
    print_header("Setup Complete!")
    print("\n  Additional Resources:")
    print("  • Documentation: See README.md")
    print("  • Streamlit Docs: https://docs.streamlit.io")
    print("  • XGBoost Docs: https://xgboost.readthedocs.io")
    print("\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Setup cancelled by user.")
        sys.exit(0)
