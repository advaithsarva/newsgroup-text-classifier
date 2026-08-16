"""Filesystem helpers.

Taken from dsutil.py (OSModulePlayGround), trimmed to the file-discovery
functions. These are what let the pipeline run on a folder of your own
documents instead of only the 20 Newsgroups corpus.
"""

import os


class OSMod:
    @staticmethod
    def filter_files_dir(directory_path, extensions):
        """
        Filters out files with certain extensions in a single directory using os.listdir().

        Args:
            directory_path (str): The directory to search.
            extensions (list): List of file extensions to filter (e.g., ['.txt', '.csv']).

        Returns:
            list: List of paths to files with the specified extensions.
        """
        filtered_files = []
        try:
            for item in os.listdir(directory_path):
                item_path = os.path.join(directory_path, item)
                if os.path.isfile(item_path) and item.endswith(tuple(extensions)):
                    filtered_files.append(item_path)
        except Exception as e:
            print(f"Error accessing directory: {e}")
        return filtered_files

    @staticmethod
    def filter_files_walk(directory_path, extensions):
        """
        Filters out files with certain extensions throughout a directory recursively using os.walk().

        Args:
            directory_path (str): The directory to search.
            extensions (list): List of file extensions to filter (e.g., ['.txt', '.csv']).

        Returns:
            list: List of paths to files with the specified extensions.
        """
        filtered_files = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith(tuple(extensions)):
                    filtered_files.append(os.path.join(root, file))
        return sorted(filtered_files)

    @staticmethod
    def is_folder_empty(folder_path):
        """
        Checks whether a folder is empty.

        Args:
            folder_path (str): Path to the folder.

        Returns:
            bool: True if the folder is empty, False otherwise.
        """
        if os.path.isdir(folder_path):
            return len(os.listdir(folder_path)) == 0
        else:
            raise ValueError(f"The path '{folder_path}' is not a directory.")
