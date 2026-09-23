import os
import pandas as pd
from google import genai
from dotenv import load_dotenv

load_dotenv()

def extract_amount_from_image(image_path: str, document_type: str = "receipt") -> float:
    """Uses Gemini Vision API to parse missing transaction amounts based on document context."""
    if not os.path.exists(image_path):
        return 0.0

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    prompt = (
        f"Identify the document type in this image. Extract the primary financial target: "
        f"Net Pay for payslips, Balance Due for rent/telecom, or Net Amount for receipts. "
        f"Return ONLY the raw final numeric float amount. Do not include currency symbols or text."
    )
    
    try:
        with open(image_path, "rb") as img_file:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, img_file.read()]
            )
        extracted_text = response.text.strip().replace("$", "").replace(",", "")
        return float(extracted_text)
    except Exception:
        # Fallback for truncated/low-confidence images
        return 0.0

def synthesize_future_events(user_profile: dict, item: str, amount: float) -> str:
    """Synthesizes future financial projections and impacts based on the transaction request."""
    # Add your specific synthesis or LLM projection logic here
    return f"Synthesized future events for {item} costing {amount}."