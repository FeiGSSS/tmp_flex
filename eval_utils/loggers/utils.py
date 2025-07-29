
import re
from typing import Any, Tuple, Union

import numpy as np


def remove_none_pattern(input_string: str) -> Tuple[str, bool]:
    """Remove the ',none' substring from the input_string if it exists at the end.

    Args:
        input_string (str): The input string from which to remove the ',none' substring.

    Returns:
        Tuple[str, bool]: A tuple containing the modified input_string with the ',none' substring removed
                          and a boolean indicating whether the modification was made (True) or not (False).
    """
    # Define the pattern to match ',none' at the end of the string
    pattern = re.compile(r",none$")

    # Use sub() to replace ',none' with an empty string
    result = re.sub(pattern, "", input_string)

    # check if the input_string changed
    removed = result != input_string

    return result, removed


def _handle_non_serializable(o: Any) -> Union[int, str, list]:
    """Handle non-serializable objects by converting them to serializable types.

    Args:
        o (Any): The object to be handled.

    Returns:
        Union[int, str, list]: The converted object. If the object is of type np.int64 or np.int32,
            it will be converted to int. If the object is of type set, it will be converted
            to a list. Otherwise, it will be converted to str.
    """
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)
