#!/usr/bin/env python3
import pytest

# Define the list of tests
tests_list = [
    "test_example_cases.py",
]

# Run pytest when this python script is executed
# pytest.main(tests_list)
# pytest.main(tests_list + [""])
pytest.main(tests_list + ["-v"])
