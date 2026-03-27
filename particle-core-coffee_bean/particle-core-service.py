import logging
import uvicorn
from typing import Union
from fastapi import FastAPI, Form, UploadFile, File, HTTPException, status
from core.detector import detect_particle_from_image

# Basic logging setup
log_format = "%(asctime)s | %(process)d | %(levelname)s | %(filename)s:%(lineno)d | %(message)s"
logging.basicConfig(level="INFO", format=log_format)

app = FastAPI()

@app.post("/v1/particle_core/detect")
async def particle_core_detect(
    file: Union[UploadFile, None] = File(default=None),
    url: str = Form(default=''),
    detect_type: int = Form(default=0),  # 0 for coffee powder, 1 for coffee bean
):
    """
    Endpoint to detect particles from an uploaded image file or a URL.
    """
    if not file and not url:
        logging.warning("Request error: No file or URL provided.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either a file or a URL must be provided."
        )

    filename = file.filename if file else url
    file_obj = file.file if file else url
    
    logging.info(f"Processing image file: {filename}...")

    try:
        # Delegate the core detection logic to the refactored function
        result = await detect_particle_from_image(file_obj, filename, detect_type)
        
        # If the detection returns an error code, reflect it in the HTTP response
        if result.get("code") != 200:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("msg", "Detection failed.")
            )
            
        return result
        
    except HTTPException as http_exc:
        # Re-raise HTTP exceptions to let FastAPI handle them
        raise http_exc
    except Exception as e:
        logging.error(f"An unexpected error occurred for {filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred: {str(e)}"
        )


"""
run at local:
workon particle
python particle-core-service.py
"""
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8833)
