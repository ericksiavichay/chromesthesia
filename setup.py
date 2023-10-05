"""
Create a directory called video_export that has two subdirectories: images and video. Also
installs the required packages.
"""

import os
import subprocess
import sys

# Install required packages
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
)

# Create directories
if not os.path.exists("video_export"):
    os.makedirs("video_export")
if not os.path.exists("video_export/images"):
    os.makedirs("video_export/images")
