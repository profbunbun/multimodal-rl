# Control Components for Dashboard
import os
from dash import dcc, html
from utils.data_loader import get_directories

def experiment_dropdown(EXPERIMENT_PATH):
    return dcc.Dropdown(
        id='experiment-dropdown',
        options=[{'label': name, 'value': name} for name in get_directories(EXPERIMENT_PATH)],
        value=None
    )

def log_folder_dropdown():
    return dcc.Dropdown(
        id='log-folder-dropdown',
        options=[],
        value=None
    )
def scan_directory(path):
    """
    Recursively scans the directory and returns a nested HTML structure representing the file tree.
    """
    if not os.path.isdir(path):
        # Base case: return a link if it's a file
        return html.Li(dcc.Link(os.path.basename(path), href=path))

    # List all files and directories in the current directory
    files_and_dirs = os.listdir(path)
    file_tree_items = []
    for subpath in files_and_dirs:
        full_subpath = os.path.join(path, subpath)
        if os.path.isdir(full_subpath):
            # Recursive case: create a nested structure for directories
            subtree = scan_directory(full_subpath)
            folder_item = html.Li([html.Span(subpath, className='folder'), subtree])
            file_tree_items.append(folder_item)
        else:
            # Base case: create a link for files
            file_link = dcc.Link(subpath, href=full_subpath)
            file_item = html.Li(file_link)
            file_tree_items.append(file_item)

    return html.Ul(file_tree_items)

def create_file_tree_component(base_path):
    """
    Creates a file tree component starting from the base_path.
    """
    file_tree_structure = scan_directory(base_path)
    return html.Div(file_tree_structure, id='file-tree', className='file-tree')