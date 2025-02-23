#!/usr/bin/env bash

# A script to install Python 3, pip, and the following Python packages:
#   transformers, torch, sentencepiece, evaluate
# It attempts to work on common Linux distros and macOS.

###################
# 1. Check Python3
###################
echo "Checking if Python3 is installed..."
if ! command -v python3 &> /dev/null
then
    echo "Python3 not found. Attempting to install..."

    if [ -x "$(command -v apt-get)" ]; then
        # Debian / Ubuntu
        sudo apt-get update
        sudo apt-get install -y python3 python3-pip
    elif [ -x "$(command -v yum)" ]; then
        # RHEL / CentOS
        sudo yum install -y python3 python3-pip
    elif [ -x "$(command -v dnf)" ]; then
        # Fedora
        sudo dnf install -y python3 python3-pip
    elif [ -x "$(command -v zypper)" ]; then
        # openSUSE
        sudo zypper install -y python3 python3-pip
    elif [ -x "$(command -v brew)" ]; then
        # macOS with Homebrew
        brew update
        brew install python3
    else
        echo "Could not detect a supported package manager."
        echo "Please install Python 3 and pip manually, then re-run this script."
        exit 1
    fi
else
    echo "Python3 is already installed."
fi

##################
# 2. Check pip3
##################
echo "Checking if pip3 is installed..."
if ! command -v pip3 &> /dev/null
then
    echo "pip3 not found. Attempting to install..."

    # We will try installing pip3 using the same package manager logic
    if [ -x "$(command -v apt-get)" ]; then
        sudo apt-get update
        sudo apt-get install -y python3-pip
    elif [ -x "$(command -v yum)" ]; then
        sudo yum install -y python3-pip
    elif [ -x "$(command -v dnf)" ]; then
        sudo dnf install -y python3-pip
    elif [ -x "$(command -v zypper)" ]; then
        sudo zypper install -y python3-pip
    elif [ -x "$(command -v brew)" ]; then
        # If Python 3 was installed via brew, pip3 should come with it
        # but let's ensure it's up to date
        brew update
        brew upgrade python3
        python3 -m ensurepip --upgrade
    else
        echo "Could not detect a package manager to install pip."
        echo "Trying python3 -m ensurepip..."
        python3 -m ensurepip --upgrade || {
            echo "Failed to install pip automatically. Please install pip manually."
            exit 1
        }
    fi
else
    echo "pip3 is already installed."
fi

#############################
# 3. Install Python packages
#############################
echo "Upgrading pip and installing packages: transformers, torch, sentencepiece, evaluate..."

pip3 install --upgrade pip
pip3 install transformers torch sentencepiece evaluate

echo "-----------------------------------------"
echo "Installation complete!"
echo "Python, pip, and the specified packages are now set up."
echo "-----------------------------------------"
