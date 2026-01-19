import os
import re

def rename_to_caps(directory):
    pattern = re.compile(r'^(\d+)_([abgr])\.png$')

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            number = match.group(1)
            letter = match.group(2)
            
            new_name = f"{number}_{letter.upper()}.png"
            
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)

            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")

rename_to_caps('./1_PREPROCESSING/IN_PREPROCESS/')