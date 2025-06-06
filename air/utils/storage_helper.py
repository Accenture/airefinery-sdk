"""
Storage Helper Module
"""

import logging
import os
import shutil

from air.utils.os_helper import secure_join

logger = logging.getLogger(__name__)


def copy_files(src_path: str, dest_path: str) -> bool:
    """
    Download files from src_path to dest_path.
    """
    try:
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        for file_name in os.listdir(src_path):
            try:
                # fixing the path traversal vulnerabilities using secure path joining
                src_file = secure_join(src_path, file_name)
                dest_file = secure_join(dest_path, file_name)
            except ValueError:
                logger.error("Skipping potentially malicious file name %s", file_name)
                continue
            shutil.copy2(src_file, dest_file)
    except Exception as e:
        logger.error("Error downloading files: %s", e)
        return False
    return True
