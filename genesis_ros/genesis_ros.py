import os
import shutil
import xml.etree.ElementTree as ET
import argparse
import xacrodoc as xd


def parse_cmake_prefix_path():
    """
    Parse the CMAKE_PREFIX_PATH environment variable and return it as a list of paths.

    Returns:
        list: A list of paths in CMAKE_PREFIX_PATH, or an empty list if the variable is not set.
    """
    cmake_prefix_path = os.getenv("CMAKE_PREFIX_PATH", "")
    # Split the variable by ':' and filter out empty strings
    paths = [path for path in cmake_prefix_path.split(":") if path]
    return paths


def find_ros2_packages():
    """
    Recursively search paths in CMAKE_PREFIX_PATH for package.xml files,
    parse them, and record ROS 2 package names and their paths.

    Returns:
        dict: A dictionary where keys are package names and values are their paths.

    Raises:
        ValueError: If multiple package.xml files are found in the same directory.
    """
    cmake_paths = parse_cmake_prefix_path()
    ros2_packages = {}

    for base_path in cmake_paths:
        for root, _, files in os.walk(base_path):
            package_files = [f for f in files if f == "package.xml"]
            if len(package_files) > 1:
                raise ValueError(
                    f"Multiple package.xml files found in directory: {root}"
                )
            elif len(package_files) == 1:
                package_xml_path = os.path.join(root, "package.xml")
                try:
                    tree = ET.parse(package_xml_path)
                    root_element = tree.getroot()
                    # Find the <name> tag in the XML
                    name_element = root_element.find("name")
                    if name_element is not None:
                        package_name = name_element.text.strip()
                        ros2_packages[package_name] = root
                except ET.ParseError:
                    print(f"Failed to parse {package_xml_path}")

    return ros2_packages


def get_package_xml_directories():
    """
    Recursively search paths in CMAKE_PREFIX_PATH for directories containing package.xml files.

    Returns:
        list: A list of directories that contain package.xml files.
    """
    cmake_paths = parse_cmake_prefix_path()
    package_dirs = []

    for base_path in cmake_paths:
        for root, _, files in os.walk(base_path):
            if "package.xml" in files:
                package_dirs.append(root)

    return package_dirs


def save_urdf_to_tmp(urdf_content):
    """
    Save the provided URDF content to a file in /tmp/genesis_ros. If the directory already exists, delete it first.

    Args:
        urdf_content (str): The URDF content as a string.

    Returns:
        str: The path to the saved URDF file.
    """
    output_dir = "/tmp/genesis_ros"

    # Delete the directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model.urdf")

    try:
        with open(output_path, "w") as urdf_file:
            urdf_file.write(urdf_content)
        print(f"URDF saved to {output_path}")
    except IOError as e:
        print(f"Failed to save URDF: {e}")
        raise

    doc = xd.XacroDoc.from_file(output_path)
    xd.packages.look_in(get_package_xml_directories())
    print(doc.to_urdf_string())
    return output_path


def main():
    """
    Main function to parse arguments and handle URDF saving.
    """
    parser = argparse.ArgumentParser(description="Process URDF files.")
    parser.add_argument(
        "--urdf", type=str, required=True, help="Path to the URDF file to be saved."
    )

    args = parser.parse_args()

    # Read URDF content from the specified file
    try:
        with open(args.urdf, "r") as urdf_file:
            urdf_content = urdf_file.read()
    except FileNotFoundError:
        print(f"The file {args.urdf} was not found.")
        return
    except IOError as e:
        print(f"Failed to read the URDF file: {e}")
        return

    # Save the URDF content to /tmp/genesis_ros
    save_urdf_to_tmp(urdf_content)


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        print(e)
