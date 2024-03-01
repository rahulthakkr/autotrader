from collections import defaultdict
import json
from typing import Callable
import os
from autotradr.config import logger
import pandas as pd


def convert_to_serializable(data):
    if isinstance(data, dict):
        return {k: convert_to_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_to_serializable(item) for item in data)
    elif isinstance(data, pd.Timestamp):
        return data.isoformat()
    elif hasattr(data, "tolist"):  # Check for numpy arrays
        return data.tolist()
    elif hasattr(data, "item"):  # Check for numpy scalar types, e.g., numpy.int32
        return data.item()
    else:
        return data


def make_directory_if_needed(file_path: str):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


def load_json_data(
    file_path: str,
    default_structure: Callable[[], list | dict | defaultdict] = list,
):
    make_directory_if_needed(file_path)

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        data = default_structure()
        logger.info(
            f"No file found {file_path}. Creating new file with default {type(data).__name__}."
        )
        with open(file_path, "w") as f:
            json.dump(data, f)
        return data
    except Exception as e:
        logger.error(
            f"Error while reading {file_path}: {e}",
            exc_info=(type(e), e, e.__traceback__),
        )
        raise Exception(f"Error while reading {file_path}")


def save_json_data(
    data: dict | list | tuple | defaultdict,
    file_path: str,
):
    make_directory_if_needed(file_path)

    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4, default=str)
    except Exception as e:
        logger.error(
            f"Error while writing to {file_path}: {e}",
            exc_info=(type(e), e, e.__traceback__),
        )
        raise Exception("Error while writing to positions.json")


def combine_data(
    existing_data: defaultdict | dict | list, new_data: defaultdict | dict | list
) -> defaultdict | dict | list:
    """Combines existing date with new data. Its output will be passed to save_data_to_json().
    This new function combined with load_json_data() and save_data_to_json() will replace
    append_data_to_json() in the future.
    """
    if existing_data is None:
        return new_data
    elif new_data is None or new_data == [] or new_data == {}:
        return existing_data

    if isinstance(existing_data, (defaultdict, dict)):
        if isinstance(existing_data, defaultdict):
            existing_data = dict(existing_data)
        if isinstance(new_data, (defaultdict, dict)):
            if isinstance(new_data, defaultdict):
                new_data = dict(new_data)
            new_data = convert_to_serializable(new_data)
            existing_data.update(new_data)
        elif isinstance(new_data, list):
            raise TypeError("Cannot combine a list with a defaultdict or dict.")
        else:
            raise TypeError("New data must be a defaultdict, dict, or list.")
    elif isinstance(existing_data, list):
        if isinstance(new_data, list):
            new_data = convert_to_serializable(new_data)
            existing_data.extend(new_data)
        elif isinstance(new_data, (defaultdict, dict)):
            if isinstance(new_data, defaultdict):
                new_data = dict(new_data)
            new_data = convert_to_serializable(new_data)
            existing_data.append(new_data)
    else:
        raise TypeError("Existing data must be a defaultdict, dict, or list.")

    return existing_data


def load_combine_save_json_data(
    new_data: defaultdict | dict | list | tuple,
    file_path: str,
    default_structure: Callable[[], list | dict | defaultdict] = list,
) -> None:
    existing_data = load_json_data(file_path, default_structure)
    if new_data is None or new_data == [] or new_data == {}:
        return
    combined_data = combine_data(existing_data, new_data)
    save_json_data(combined_data, file_path)
