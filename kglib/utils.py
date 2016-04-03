import importlib
import os


def import_class_by_path(path):
    """Return class object given its path string (e.g. 'foo.bar.Baz')."""
    class_module_str, class_name = path.rsplit('.', maxsplit=1)
    class_module = importlib.import_module(class_module_str)
    class_ = getattr(class_module, class_name)
    return class_


def ensure_dir_exists(path):
    """Create a directory if one doesn't already exist."""
    if not os.path.exists(path):
        os.mkdir(path)


def ensure_extension(filename, extension):
    """Add the extension to the filename if it hasn't got one."""
    _, ext = os.path.splitext(filename)
    if not ext:
        filename = '{}.{}'.format(filename, extension)
    return filename
