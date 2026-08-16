"""CSV and JSON helpers.

Taken from dsutil.py (OSModulePlayGround) and trimmed to what this project
uses. Kept as static methods on a class to match the original.
"""

import csv
import json
import os


class CSVJSON:
    """A utility class for handling CSV and JSON files in data science projects."""

    @staticmethod
    def read_csv(file_path):
        """Read a CSV file and return its content as a list of dictionaries."""
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            return [row for row in reader]

    @staticmethod
    def write_csv(file_path, data, fieldnames=None):
        """Write data to a CSV file.

        Args:
            file_path (str): Path to the output CSV file.
            data (list): List of dictionaries containing the data.
            fieldnames (list): Column names. Taken from the first row if omitted.
        """
        if fieldnames is None:
            fieldnames = list(data[0].keys()) if data else []
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    @staticmethod
    def read_json(file_path):
        """Read a JSON file and return its content as a Python object."""
        with open(file_path, mode='r', encoding='utf-8') as file:
            return json.load(file)

    @staticmethod
    def write_json(file_path, data, indent=4):
        """Write a Python object to a JSON file.

        Args:
            file_path (str): Path to the output JSON file.
            data (dict or list): The data to write.
            indent (int): Indentation level for pretty printing.
        """
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, mode='w', encoding='utf-8') as file:
            json.dump(data, file, indent=indent, default=float)

    @staticmethod
    def csv_to_json(csv_path, json_path):
        """Convert a CSV file to a JSON file."""
        CSVJSON.write_json(json_path, CSVJSON.read_csv(csv_path))

    @staticmethod
    def json_to_csv(json_path, csv_path):
        """Convert a JSON file to a CSV file.

        Raises:
            ValueError: if the JSON is not a list of dictionaries.
        """
        data = CSVJSON.read_json(json_path)
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            CSVJSON.write_csv(csv_path, data)
        else:
            raise ValueError("JSON data is not a list of dictionaries. Cannot convert to CSV.")
