from importlib.util import spec_from_file_location, module_from_spec
import sys
from typing import Any
from pathlib import Path


def call_function_in_another_file(
    file_path: Path, function_name: str, *args, **kwargs
) -> Any:
    """
    Executes a function with a specified name from a given file.

    Args:
        file_path (Path): The path to the file where the function is defined.
        function_name (str): The name of the function to execute.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        any: The return value of the executed function. Returns None if the function does not exist.
    """
    spec = spec_from_file_location("module.name", file_path)
    if spec is None:
        print(f"Error: File '{file_path}' not found.")
        return None
    module = module_from_spec(spec)
    sys.modules["module.name"] = module
    if spec.loader is not None:
        spec.loader.exec_module(module)
    else:
        raise ImportError(f"Cannot load module from {file_path}")

    if hasattr(module, function_name):
        func_to_call = getattr(module, function_name)
        return func_to_call(*args, **kwargs)
    else:
        print(
            f"Error: Function '{function_name}' does not exist in file '{file_path}'."
        )
        return None
