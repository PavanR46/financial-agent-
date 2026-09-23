import os
import mimetypes
import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

IMAGES_CSV_PATH = os.path.join("dataset", "images.csv")
MEDIA_DIR = os.path.join("dataset", "media", "images")
CACHE_CSV_PATH = os.path.join("dataset", "ocr_cache.csv")


def load_cache() -> dict:
    """Loads existing OCR cache into a dictionary."""
    if os.path.exists(CACHE_CSV_PATH):
        df_cache = pd.read_csv(CACHE_CSV_PATH)
        return dict(zip(df_cache["image_id"], df_cache["extracted_amount"]))
    return {}


def save_cache(cache: dict) -> None:
    """Saves updated OCR cache to disk."""
    df_cache = pd.DataFrame(
        list(cache.items()), columns=["image_id", "extracted_amount"]
    )
    df_cache.to_csv(CACHE_CSV_PATH, index=False)


def extract_amount_from_image(image_path: str) -> float:
    """Uses Gemini Vision API to parse missing transaction amounts from receipt/invoice images."""
    if not os.path.exists(image_path):
        return 0.0

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return 0.0

    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/png"

    prompt = (
        "Identify the financial document type in this image. "
        "Extract the primary target amount: Net Pay for payslips, Balance Due for bills, or Net Amount/Total for receipts. "
        "Return ONLY the raw final numeric float amount without any currency symbols, commas, or surrounding text."
    )

    try:
        client = genai.Client(api_key=api_key)
        with open(image_path, "rb") as img_file:
            image_part = types.Part.from_bytes(
                data=img_file.read(),
                mime_type=mime_type
            )

            # Try gemini-2.0-flash model
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt, image_part]
            )

        extracted_text = response.text.strip().replace("$", "").replace(",", "")
        return float(extracted_text)
    except Exception as e:
        print(f"OCR Note: Image processing fallback for {image_path}: {e}")
        return 0.0


def resolve_missing_image_amounts() -> pd.DataFrame:
    """Parses images.csv, uses OCR for missing amounts, and returns the resolved DataFrame."""
    if not os.path.exists(IMAGES_CSV_PATH):
        return pd.DataFrame()

    df = pd.read_csv(IMAGES_CSV_PATH)
    cache = load_cache()
    cache_updated = False

    for idx, row in df.iterrows():
        image_id = str(row.get("image_id", "")).strip()
        if not image_id or image_id == "nan":
            continue

        image_filename = f"{image_id}.png" if not image_id.endswith(".png") else image_id

        # 1. Check local cache first
        if image_id in cache:
            df.at[idx, "amount"] = cache[image_id]
            continue

        # 2. Call Gemini API if not cached
        full_image_path = os.path.join(MEDIA_DIR, image_filename)
        extracted_val = extract_amount_from_image(full_image_path)

        df.at[idx, "amount"] = extracted_val
        cache[image_id] = extracted_val
        cache_updated = True

    if cache_updated:
        save_cache(cache)

    return df


if __name__ == "__main__":
    print("Resolving missing financial amounts via OCR...")
    resolved_df = resolve_missing_image_amounts()
    print("OCR Processing Complete.")