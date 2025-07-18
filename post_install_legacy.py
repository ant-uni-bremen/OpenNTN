import os
import shutil
import sys
from pathlib import Path
import pkg_resources
#import sionna

def modify_vanilla_sionna():
    """Modify the installed framework files as needed."""
    try:
        # Locate the framework installation path
        framework_path = pkg_resources.get_distribution("sionna").location

        sionna_channel_path = framework_path + "/sionna/channel"
        channel_init_file_path = sionna_channel_path + "/__init__.py"

        line_to_add = 'from . import tr38811'
        preceding_line = 'from . import tr38901'

        with open(channel_init_file_path, 'r') as file:
            lines = file.readlines()    

        line_present = any(line_to_add in line for line in lines)

        insert_index = None

        if not line_present:
            for i, line in enumerate(lines):
                if preceding_line in line:
                    insert_index = i + 1
                    break

        if line_present:
            print(f"Already importing tr38811 in channel: {line_to_add}")

        if insert_index is not None:
            lines.insert(insert_index, f"{line_to_add}\n")
            with open(channel_init_file_path, 'w') as file:
                file.writelines(lines)
            print(f"Added import of tr38811 in channel: {line_to_add}")
        elif not line_present:
            print(f"Could not find import of tr38901.")
        print("ran the installer file")

        print("the framework was found at ", framework_path)
        print("sionna path is: ", sionna_channel_path)
    except Exception as e:
        print(f"Error modifying framework: {e}", file=sys.stderr)

if __name__ == "__main__":
    modify_vanilla_sionna()
