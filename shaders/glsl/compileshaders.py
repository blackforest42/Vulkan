# Copyright (C) 2016-2024 by Sascha Willems - www.saschawillems.de
# This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)

import argparse
import os
import subprocess
import sys

parser = argparse.ArgumentParser(description='Compile all GLSL shaders in a folder')
parser.add_argument('folder', type=str, help='path to folder containing GLSL shaders')
parser.add_argument('--glslang', type=str, help='path to glslangvalidator executable')
parser.add_argument('--g', action='store_true', help='compile with debug symbols')
parser.add_argument('--recursive', '-r', action='store_true', help='recursively compile shaders in subdirectories')
args = parser.parse_args()

def findGlslang():
    def isExe(path):
        return os.path.isfile(path) and os.access(path, os.X_OK)

    if args.glslang != None and isExe(args.glslang):
        return args.glslang

    exe_name = "glslangValidator"
    if os.name == "nt":
        exe_name += ".exe"

    for exe_dir in os.environ["PATH"].split(os.pathsep):
        full_path = os.path.join(exe_dir, exe_name)
        if isExe(full_path):
            return full_path

    sys.exit("Could not find glslangvalidator executable on PATH, and was not specified with --glslang")

file_extensions = tuple([".vert", ".frag", ".comp", ".geom", ".tesc", ".tese", ".rgen", ".rchit", ".rmiss", ".mesh", ".task"])

# Validate folder argument
if not os.path.isdir(args.folder):
    sys.exit(f"Error: '{args.folder}' is not a valid directory")

glslang_path = findGlslang()
dir_path = os.path.abspath(args.folder).replace('\\', '/')

# Choose between recursive walk or single directory
if args.recursive:
    file_iterator = os.walk(dir_path)
else:
    # Only process files in the specified directory (not subdirectories)
    try:
        files = os.listdir(dir_path)
        file_iterator = [(dir_path, [], files)]
    except PermissionError:
        sys.exit(f"Error: Permission denied accessing '{dir_path}'")

compiled_count = 0
for root, dirs, files in file_iterator:
    for file in files:
        if file.endswith(file_extensions):
            input_file = os.path.join(root, file)
            output_file = input_file + ".spv"

            add_params = ""
            if args.g:
                add_params = "-g"

            # Ray tracing shaders require a different target environment
            if file.endswith(".rgen") or file.endswith(".rchit") or file.endswith(".rmiss"):
               add_params = add_params + " --target-env vulkan1.2"
            # Same goes for samples that use ray queries
            if root.endswith("rayquery") and file.endswith(".frag"):
                add_params = add_params + " --target-env vulkan1.2"
            # Mesh and task shader also require different settings
            if file.endswith(".mesh") or file.endswith(".task"):
                add_params = add_params + " --target-env spirv1.4"

            print(f"Compiling: {input_file}")
            res = subprocess.call("%s -V %s -o %s %s" % (glslang_path, input_file, output_file, add_params), shell=True)
            if res != 0:
                sys.exit(res)
            compiled_count += 1

print(f"\nSuccessfully compiled {compiled_count} shader(s)")