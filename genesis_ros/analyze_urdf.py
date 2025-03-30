import xml.etree.ElementTree as ET
from pathlib import Path


def get_camera_sensors(urdf_path: Path):
    tree = ET.parse(urdf_path)
