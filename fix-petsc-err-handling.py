#!/usr/bin/env python3
import re

# Define the filename
for filename in [
    "src/MDP/MDP_algorithm.cpp",
    "src/MDP/MDP_change.cpp",
    "src/MDP/MDP_setup.cpp",
    "include/MDP.h",
]:
    # Read the file
    with open(filename, "r") as file:
        data = file.read()

    # (1) Add missing error handling
    data = re.sub(
        r"^(\s*)(Vec|IS|Mat|Petsc|KSP)([\w:]+\(.*);",
        r"\1PetscCallThrow(\2\3);",
        data,
        flags=re.MULTILINE,
    )

    # (2) Replace old error handling with new one
    data = re.sub(
        r"(^\s*)ierr = (.*?); CHKERRQ\(ierr\);",
        r"\1PetscCallThrow(\2);",
        data,
        flags=re.MULTILINE,
    )

    # (3) Replace PetscErrorCode function return types with void
    modified_functions = []

    def replace_return_type(match):
        function_name = match.group(4)
        if function_name is not None:
            modified_functions.append(function_name)
            keyword = match.group(2) if match.group(2) is not None else ""
            return f"{match.group(1)}{keyword}void {function_name}("
        else:
            return match.group(
                0
            )  # return the entire match in case of no match for group(4)

    data = re.sub(
        r"(^\s*)((virtual|static) )?PetscErrorCode *([\w:]+) *\(",
        replace_return_type,
        data,
        flags=re.MULTILINE,
    )

    # (4) Remove return statement and all preceding blank lines in functions affected by (3)
    for function in modified_functions:
        data = re.sub(
            r"(void " + re.escape(function) + r".*?)\n\s*return.*?(?=\n\s*})",
            r"\1",
            data,
            flags=re.MULTILINE | re.DOTALL,
        )

    # (5) Remove PetscErrorCode ierr; declarations in functions affected by (3)
    for function in modified_functions:
        data = re.sub(
            r"(void " + re.escape(function) + r".*?\n)\s*PetscErrorCode \s*ierr;\s*\n",
            r"\1",
            data,
            flags=re.MULTILINE | re.DOTALL,
        )

    # Write result back the file
    with open(filename, "w") as file:
        file.write(data)
