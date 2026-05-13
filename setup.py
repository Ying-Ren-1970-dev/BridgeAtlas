"""Setup script for Librarian system."""
import subprocess
import sys
from pathlib import Path


def check_python_version():
    """Check if Python version is adequate."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✓ Python version: {sys.version.split()[0]}")
    return True


def install_dependencies():
    """Install required packages."""
    print("\nInstalling dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing dependencies: {e}")
        return False


def setup_env_file():
    """Create .env file from template if it doesn't exist."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("\n✓ .env file already exists")
        return True
    
    if env_example.exists():
        import shutil
        shutil.copy(env_example, env_file)
        print("\n✓ Created .env file from template")
        print("\n⚠ IMPORTANT: Edit .env file and add your OpenAI API key!")
        return True
    else:
        print("\n✗ .env.example file not found")
        return False


def check_projects_folder():
    """Check if Projects folder exists and contains PDFs."""
    projects_folder = Path("Projects")
    
    if not projects_folder.exists():
        print("\n✗ Projects folder not found")
        projects_folder.mkdir()
        print("✓ Created Projects folder - please add your PDF files here")
        return False
    
    pdf_files = list(projects_folder.rglob("*.pdf"))
    
    if not pdf_files:
        print(f"\n⚠ No PDF files found in {projects_folder}")
        print("  Please add PDF files to the Projects folder")
        return False
    
    print(f"\n✓ Found {len(pdf_files)} PDF file(s) in Projects folder")
    return True


def create_directories():
    """Create necessary directories."""
    dirs = ['vector_db', 'logs']
    for dir_name in dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
            print(f"✓ Created {dir_name} directory")


def main():
    """Run setup process."""
    print("=" * 60)
    print("LIBRARIAN SETUP")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install dependencies
    if not install_dependencies():
        return
    
    # Setup .env file
    setup_env_file()
    
    # Check projects folder
    has_pdfs = check_projects_folder()
    
    # Create directories
    print("\nCreating directories...")
    create_directories()
    
    # Final instructions
    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Edit .env file and add your OpenAI API key")
    
    if not has_pdfs:
        print("2. Add PDF files to the Projects folder")
        print("3. Run: python main.py build")
        print("4. Run: python cli.py (for interactive mode)")
    else:
        print("2. Run: python main.py build")
        print("3. Run: python cli.py (for interactive mode)")
    
    print("\nFor help, see README.md or run: python cli.py")


if __name__ == "__main__":
    main()
