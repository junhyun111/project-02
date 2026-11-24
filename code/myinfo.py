"""Student information.

TODO: fill in your name and student ID in `get_info`.

They are used for the name of the result file:
    result/result_<student ID>_<name>.csv
"""


def get_info():
    """Return your name and student ID as a dict of two strings."""
    name = "강준현"          # TODO: your name
    student_id = "20240674"    # TODO: your student ID
    return {"name": name, "student_id": student_id}
