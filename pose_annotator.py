import sys
import os

# Add the src directory to the Python path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_path)

# Make sure the root directory is also in path (for imports like plot_utils)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pose_editor.pose_annotator import main

if __name__ == "__main__":
    main()