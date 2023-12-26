#!/bin/bash

# Define the root directory for the dashboard
DASHBOARD_DIR="dashboard"

# Create the directory structure for the dashboard
mkdir -p $DASHBOARD_DIR/components
mkdir -p $DASHBOARD_DIR/data
mkdir -p $DASHBOARD_DIR/static
mkdir -p $DASHBOARD_DIR/utils
mkdir -p $DASHBOARD_DIR/config

# Create placeholder files for each directory

# Components directory
echo "# Custom Dashboard Components" > $DASHBOARD_DIR/components/__init__.py
echo "# Graph Components for Dashboard" > $DASHBOARD_DIR/components/graph_components.py
echo "# Control Components for Dashboard" > $DASHBOARD_DIR/components/control_components.py

# Data directory (Empty, but you can add historical data here)
# Static directory (Empty, but you can add CSS, JS, images here)

# Utils directory
echo "# Utility functions for Dashboard" > $DASHBOARD_DIR/utils/__init__.py
echo "# Data Loader for Dashboard" > $DASHBOARD_DIR/utils/data_loader.py

# Config directory
echo "# Configuration settings for Dashboard" > $DASHBOARD_DIR/config/dashboard_config.py

# Main Dash application file
echo "import dash" > $DASHBOARD_DIR/app.py
echo "print('Setup Dash app here')" >> $DASHBOARD_DIR/app.py

# Requirements file
echo "dash" > $DASHBOARD_DIR/requirements.txt
echo "pandas" >> $DASHBOARD_DIR/requirements.txt
echo "plotly" >> $DASHBOARD_DIR/requirements.txt

# Inform user of completion
echo "Dashboard directory structure and basic files have been created."
