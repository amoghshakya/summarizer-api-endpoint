import logging
import os

from fastapi import APIRouter, HTTPException

from app.models import SummarizeRequest
from app.services import summarize, text_to_speech

router = APIRouter()

# TODO: rewrite the function to use a more robust logging system and error
# handling


@router.post("/summarize")
async def summarize_text(request: SummarizeRequest):
    if not request.text.strip():
        raise HTTPException(
            status_code=400, detail="Input text cannot be empty.")

    try:
        summary = summarize(request.text)

        if not summary:
            logging.error("Summary generation failed.")
            raise HTTPException(
                status_code=500, detail="Summary generation failed.")

        try:
            audio_path = text_to_speech(summary)
            audio_url = f"/audio/{os.path.basename(audio_path)}"

        except Exception as e:
            audio_url = None

        return {"summary": summary, "audio_url": audio_url}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error occured during summarization: {str(e)}"
        )
