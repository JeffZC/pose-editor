import sys
from importlib import import_module

# Map old top-level names to wrapper modules inside _legacy
_OLD_TO_NEW = {
    "pose_editor.body_format": "pose_editor._legacy.body_format_utils",
    "pose_editor.hand_format": "pose_editor._legacy.hand_format_utils",
    # add more as needed
}

for old, new in _OLD_TO_NEW.items():
    sys.modules[old] = import_module(new)
