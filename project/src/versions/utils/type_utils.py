"""
Utility functions for runtime type checking and validation.

This module provides helper functions for checking dictionary structures against
TypedDict definitions at runtime, which is not possible with the built-in isinstance().
"""

import inspect
from typing import Any, Type, TypedDict, get_type_hints


def is_typed_dict_instance(obj: Any, typed_dict_class: Type[TypedDict]) -> bool:
    """
    Check if a dictionary conforms to the structure of a TypedDict at runtime.

    This function verifies that:
    1. The object is a dictionary
    2. The dictionary has all the required keys from the TypedDict
    3. No type checking of values is performed, only presence of keys

    Args:
        obj (Any): The object to check
        typed_dict_class (Type[TypedDict]): The TypedDict class to check against

    Returns:
        bool: True if the object conforms to the TypedDict structure, False otherwise
    """
    # Check if obj is a dictionary
    if not isinstance(obj, dict):
        return False

    # Get the field names from the TypedDict class
    try:
        type_hints = get_type_hints(typed_dict_class)
    except TypeError:
        # Handle case where typed_dict_class is not actually a TypedDict class
        return False

    # Get required keys (this is a bit hacky but works for most TypedDict implementations)
    # Note: For Python 3.9+, we could use typed_dict_class.__required_keys__,
    # but using a more compatible approach here
    required_keys = set(type_hints.keys())

    # Check if all required keys are in the dictionary
    for key in required_keys:
        if key not in obj:
            return False

    return True


def get_matching_typed_dict(
    obj: Any, typed_dict_options: list[Type[TypedDict]]
) -> Type[TypedDict] | None:
    """
    Find which TypedDict class from a list matches a dictionary structure.

    Args:
        obj (Any): The object to check
        typed_dict_options (list[Type[TypedDict]]): List of TypedDict classes to check against

    Returns:
        Type[TypedDict] | None: The matching TypedDict class or None if no match found
    """
    for typed_dict_class in typed_dict_options:
        if is_typed_dict_instance(obj, typed_dict_class):
            return typed_dict_class
    return None


def get_discriminator_field(
    obj: dict, typed_dict_options: list[Type[TypedDict]]
) -> str | None:
    """
    Find a field that can be used to discriminate between different TypedDict classes.

    This function looks for a field that exists in all TypedDict options but has
    a unique presence in each, making it suitable for discriminating between types.

    Args:
        obj (dict): The dictionary to check
        typed_dict_options (list[Type[TypedDict]]): List of TypedDict classes to examine

    Returns:
        str | None: The name of a field that can be used as discriminator, or None if not found
    """
    if not typed_dict_options:
        return None

    # Get all field names from all TypedDict options
    all_fields = set()
    field_presence = {}

    for typed_dict_class in typed_dict_options:
        type_hints = get_type_hints(typed_dict_class)
        class_fields = set(type_hints.keys())
        all_fields.update(class_fields)

        # Track which fields are in which classes
        for field in class_fields:
            if field not in field_presence:
                field_presence[field] = set()
            field_presence[field].add(typed_dict_class.__name__)

    # Find fields that are unique to each TypedDict
    for field, classes in field_presence.items():
        if len(classes) == 1 and field in obj:
            return field

    return None
