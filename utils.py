import os
import hashlib
from datetime import datetime, timedelta
from PIL import Image, UnidentifiedImageError
import io
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def validate_image(image: Image.Image) -> Dict[str, Any]:
    """Validate uploaded image"""
    try:
        # Check image size
        if image.size[0] < 50 or image.size[1] < 50:
            return {"valid": False, "message": "Image too small (min 50x50 pixels)"}

        # Check if image is too large
        if image.size[0] > 5000 or image.size[1] > 5000:
            return {"valid": False, "message": "Image too large (max 5000x5000 pixels)"}

        # Check image mode and convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        return {"valid": True, "message": "Image valid"}

    except UnidentifiedImageError:
        return {"valid": False, "message": "Invalid image file"}
    except Exception as e:
        return {"valid": False, "message": f"Image validation failed: {str(e)}"}


def save_uploaded_image(contents: bytes, filename: str) -> str:
    """Save uploaded image for debugging"""
    from config import settings

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    # Create unique filename
    file_hash = hashlib.md5(contents).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Clean filename
    safe_filename = "".join(c for c in filename if c.isalnum() or c in ('-', '_', '.')).rstrip()
    safe_filename = f"{timestamp}_{file_hash}_{safe_filename}"

    filepath = os.path.join(settings.UPLOAD_DIR, safe_filename)

    try:
        with open(filepath, "wb") as f:
            f.write(contents)
        logger.info(f"💾 Saved uploaded image: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"❌ Failed to save image: {str(e)}")
        raise


def cleanup_old_files(days_old: int = 1):
    """Clean up old uploaded files"""
    from config import settings

    upload_dir = settings.UPLOAD_DIR
    if not os.path.exists(upload_dir):
        return

    cutoff_time = datetime.now() - timedelta(days=days_old)
    deleted_count = 0

    for filename in os.listdir(upload_dir):
        filepath = os.path.join(upload_dir, filename)
        try:
            file_time = datetime.fromtimestamp(os.path.getctime(filepath))

            if file_time < cutoff_time:
                os.remove(filepath)
                deleted_count += 1
                logger.info(f"🗑️ Cleaned up old file: {filename}")
        except Exception as e:
            logger.warning(f"Failed to delete {filename}: {str(e)}")

    if deleted_count > 0:
        logger.info(f"🧹 Cleanup completed: {deleted_count} files deleted")


def get_file_extension(filename: str) -> str:
    """Extract file extension in lowercase"""
    return os.path.splitext(filename)[1].lower()


def is_allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    from config import settings
    allowed_extensions = {ext.lower() for ext in settings.ALLOWED_EXTENSIONS}
    file_extension = get_file_extension(filename)
    return file_extension in allowed_extensions