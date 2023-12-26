# Tree Components for Dashboard
import os
from dash import dcc, html

def scan_directory(path, parent_path="", hide_children=True, special_folders=None):
    if special_folders is None:
        special_folders = ['logger']

    if not os.path.isdir(path):
        # Base case: return a link if it's a file
        file_id = os.path.join(parent_path, os.path.basename(path))
        return html.Li(dcc.Link(os.path.basename(path), href='#', id={'type': 'file-link', 'index': file_id}))

    # List all files and directories in the current directory
    files_and_dirs = os.listdir(path)
    file_tree_items = []
    for subpath in files_and_dirs:
        full_subpath = os.path.join(path, subpath)
        subpath_id = os.path.join(parent_path, subpath)
        if os.path.isdir(full_subpath):
            is_special = subpath in special_folders
            subtree = scan_directory(full_subpath, parent_path=subpath_id, hide_children=hide_children or not is_special, special_folders=special_folders)
            folder_class = 'special-folder' if is_special else 'folder'
            folder_item = html.Li([
                dcc.Link(subpath, href='#', id={'type': 'folder-link', 'index': subpath_id}),  # Clickable folder link
                subtree
            ], className=folder_class)
            file_tree_items.append(folder_item)
        else:
            # Base case: create a link for files
            file_link = dcc.Link(subpath, href='#', id={'type': 'file-link', 'index': subpath_id})
            file_item = html.Li(file_link)
            file_tree_items.append(file_item)

    return html.Ul(file_tree_items, className='file-tree')

def create_file_tree_component(base_path):
    file_tree_structure = scan_directory(base_path)
    return html.Div(file_tree_structure, id='file-tree-container', style={'overflow': 'auto', 'height': '400px', 'border': '1px solid #ddd'})

