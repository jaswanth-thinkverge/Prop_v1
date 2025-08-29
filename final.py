from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import fitz 
import pytesseract
from PIL import Image
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import Optional, List, Tuple
import logging
from io import BytesIO
import requests
import re
import json
from pydantic import BaseModel, Field
import pdfplumber
import camelot
from pdf2image import convert_from_bytes
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",  
        "http://192.168.1.17:8501"  
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


def is_pdf_scanned(pdf_bytes: bytes) -> bool:
    """
    Determine if a PDF is scanned by checking for the absence of text
    and the presence of images.
    """
    try:
        # Check for extractable text using pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            has_text = any(page.extract_text() for page in pdf.pages)
            if has_text:
                return False 

        # If no text found, then check for images using PyMuPDF 
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        has_images = any(page.get_images(full=True) for page in doc)
        
        return has_images 

    except Exception as e:
        print(f"[ERROR] is_pdf_scanned failed: {e}")
        return False  
    
def extract_assets_from_text(text: str) -> list:
    def is_valid_price(value: str) -> bool:
        """Check if a string is a valid numeric price (not a date, year, or phone number)."""
        if not re.search(r'\d', value):
            return False
        value_clean = value.replace(",", "").strip()
        if value_clean.isdigit() and 1900 <= int(value_clean) <= 2100:
            return False
        if value_clean.isdigit() and len(value_clean) > 10:
            return False
        if value_clean.isdigit() and len(value_clean) < 2:
            return False
        return True

    assets = []

    # Try to find assets based on common keywords
    asset_keywords = ["Land & Building", "RESIDENTIAL FLAT", "Plant & Machinery"]
    descriptions_with_keywords = re.split(f'({"|".join(asset_keywords)})', text, flags=re.IGNORECASE)
    descriptions_with_keywords = [d.strip() for d in descriptions_with_keywords if d.strip()]

    # Merge keyword and description
    asset_descriptions = []
    for i in range(0, len(descriptions_with_keywords), 2):
        if i + 1 < len(descriptions_with_keywords):
            asset_descriptions.append(descriptions_with_keywords[i] + " " + descriptions_with_keywords[i + 1])
        else:
            asset_descriptions.append(descriptions_with_keywords[i])

    # Extract price-like numbers
    price_pattern = re.compile(r'[\d,]+\.\d{1,2}|[\d,]+')
    all_prices = price_pattern.findall(text)
    all_prices = [p.replace(",", "") for p in all_prices if is_valid_price(p)]

    # Take only last 18 values if there are many
    if len(all_prices) >= 18:
        relevant_prices = all_prices[-18:]
    else:
        relevant_prices = all_prices

    price_index = 0
    for description in asset_descriptions:
        current_asset = {
            "block_name": "",
            "asset_description": description.strip(),
            "reserve_price": "",
            "emd_amount": "",
            "incremental_bid_amount": ""
        }

        # Set block name
        if "Land & Building" in description:
            current_asset["block_name"] = "Land & Building"
        elif "RESIDENTIAL FLAT" in description:
            current_asset["block_name"] = "RESIDENTIAL FLAT"
        elif "Plant & Machinery" in description:
            current_asset["block_name"] = "PLANT & MACHINERY"

        # Assign only if next 3 prices are valid
        if price_index + 3 <= len(relevant_prices):
            group = relevant_prices[price_index:price_index + 3]
            if all(is_valid_price(p) for p in group):
                current_asset["reserve_price"], current_asset["emd_amount"], current_asset["incremental_bid_amount"] = group
            price_index += 3

        assets.append(current_asset)

    return assets
    
def format_tables_as_markdown(tables: List[List[List[str]]]):
    markdown = ""
    for table in tables:
        if not table or len(table) < 2:
            continue
        headers = [h.strip() if h else "" for h in table[0]]
        markdown += "\n| " + " | ".join(headers) + " |\n"
        markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in table[1:]:
            row = [cell.strip().replace("\n", " ") if cell else "" for cell in row]
            while len(row) < len(headers):
                row.append("")
            markdown += "| " + " | ".join(row) + " |\n"
    return markdown

def extract_tables_with_camelot(pdf_bytes: bytes, page_number: int = None) -> List[List[List[str]]]:
    """
    Extracts tables from a specific page of a PDF using Camelot.
    Tries both 'lattice' and 'stream' flavors.
    """
    tables = []
    page_str = "all" if page_number is None else str(page_number)
    
    try:
        # Try 'lattice' for tables with clear lines
        print(f"[INFO] Trying Camelot with 'lattice' flavor on page {page_str}...")
        lattice_tables = camelot.read_pdf(io.BytesIO(pdf_bytes), pages=page_str, flavor='lattice')
        if lattice_tables.n > 0:
            tables.extend([table.data for table in lattice_tables])
            print(f"[INFO] Found {lattice_tables.n} table(s) with 'lattice' flavor.")
            return tables

        # If that fails, try 'stream' for tables without clear lines
        print(f"[INFO] 'lattice' failed, trying 'stream' flavor on page {page_str}...")
        stream_tables = camelot.read_pdf(io.BytesIO(pdf_bytes), pages=page_str, flavor='stream')
        if stream_tables.n > 0:
            tables.extend([table.data for table in stream_tables])
            print(f"[INFO] Found {stream_tables.n} table(s) with 'stream' flavor.")
            return tables
            
    except Exception as e:
        print(f"[ERROR] Camelot table extraction failed: {e}")
        
    return tables

def ocr_pdf(pdf_bytes: io.BytesIO) -> Tuple[str, List]:
    """
    Runs OCR on all pages of a PDF and returns extracted text + empty table list.
    """
    ocr_text = ""
    tables = []

    try:
        images = convert_from_bytes(pdf_bytes.getvalue())
        for img in images:
            text = pytesseract.image_to_string(img)
            ocr_text += text.strip() + "\n"
    except Exception as e:
        print(f"[ERROR] OCR failed: {e}")

    return ocr_text, tables


def fetch_text_from_url(pdf_url: str) -> Tuple[str, List, bool]:
    response = requests.get(pdf_url, timeout=15)
    response.raise_for_status()
    pdf_bytes = response.content
    pdf_io = io.BytesIO(pdf_bytes)

    raw_text = ""
    tables = []
    scanned_pdf = False

    try:
        with pdfplumber.open(pdf_io) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                raw_text += page_text.strip() + "\n"

        # Check if pdfplumber failed to extract text, and use OCR if so.
        
        if not raw_text.strip():
            print("[INFO] pdfplumber extracted no text. Falling back to OCR...")
            scanned_pdf = True
            raw_text, tables = ocr_pdf(pdf_io)
        
        # If no tables were found, use Camelot as a fallback.
        if not tables and not scanned_pdf:
            print("[INFO] pdfplumber did not find any tables. Trying Camelot on All Pages...")
            tables = extract_tables_with_camelot(pdf_bytes)

    except Exception as e:
        print(f"[ERROR] PDF extraction failed: {e}")
        # If pdfplumber fails entirely, we check if it's a scanned PDF
        if is_pdf_scanned(pdf_bytes):
            scanned_pdf = True
            raw_text, tables = ocr_pdf(pdf_io)
        else:
            print("[INFO] PDF extraction failed but document is not scanned. No OCR fallback.")
            raw_text = ""
            tables = []

    return raw_text.strip(), tables, scanned_pdf

def truncate_text(text: str, max_words: int = 5000) -> str:
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text

def clean_llm_output(output: str) -> str:
    import re
    match = re.search(r"{.*}", output, re.DOTALL)
    return match.group(0) if match else output

def normalize_keys(obj):
    if isinstance(obj, dict):
        return {k.lower().replace(" ", "_"): normalize_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [normalize_keys(i) for i in obj]
    else:
        return obj

def clean_assets(assets: list) -> list:
    """
    Cleans OCR noise from asset descriptions and formats numeric values.
    """
    cleaned_assets = []

    for asset in assets:
        cleaned_asset = asset.copy()

        #  Clean asset description
        desc = asset.get("asset_description", "")
        desc = re.sub(r"http\S+|www\.\S+", "", desc)  
        desc = re.sub(r"\S+@\S+", "", desc)  
        desc = re.sub(r"\s+", " ", desc)  # 
        cleaned_asset["asset_description"] = desc.strip()

        # Clean numeric fields
        for field in ["reserve_price", "emd_amount", "incremental_bid_amount"]:
            value = asset.get(field, "")
            if value:
                value_clean = re.sub(r"[^\d.,]", "", value)
                value_clean = value_clean.replace(",", "")  
                cleaned_asset[field] = value_clean if value_clean else ""
            else:
                cleaned_asset[field] = ""

        cleaned_assets.append(cleaned_asset)

    return cleaned_assets

# Auction Insights Model and Endpoint
class AuctionDetails(BaseModel):
    # Accept both original CSV headers and snake_case versions
    name_of_corporate_debtor_pdf_: str = Field(..., alias="Name of Corporate Debtor (PDF)")
    auction_notice_url: str = Field(..., alias="Auction Notice URL")
    date_of_auction_pdf: Optional[str] = Field(None, alias="Date of Auction (PDF)")
    unique_number: Optional[str] = Field(None, alias="Unique Number")
    ip_registration_number: Optional[str] = Field(None, alias="IP Registration Number")
    auction_platform: Optional[str] = Field(None, alias="Auction Platform")
    details_url: Optional[str] = Field(None, alias="Details URL")
    
    borrower_name: Optional[str] = None
    
    class Config:
        allow_population_by_field_name = True
        extra = "ignore"  


async def generate_auction_insights(corporate_debtor: str, auction_notice_url: str) -> dict:
    try:
        # Extract raw text, tables, and scanned PDF flag
        raw_text, tables, scanned_pdf = fetch_text_from_url(auction_notice_url)

        # Fallback assets from OCR/text parsing
        fallback_assets = extract_assets_from_text(raw_text)

        # Decide whether to use fallback
        use_fallback = not tables or len(tables[0]) <= 2

        if use_fallback:
            logger.warning("[FALLBACK] Using extracted assets from text due to missing or bad table...")
            for asset in fallback_assets:
                print("[DEBUG] Fallback Asset:", asset)

            markdown_table = ""  
            assets_for_prompt = fallback_assets
        else:
            logger.info("[INFO] Using structured tables from PDF/OCR.")
            markdown_table = format_tables_as_markdown(tables)
            assets_for_prompt = None  

        if assets_for_prompt:
            assets_for_prompt = clean_assets(assets_for_prompt)

        # Debug logs
        logger.info("[DEBUG] OCR Used: %s", scanned_pdf)
        logger.info("[DEBUG] Extracted Raw Text (First 500 chars):\n%s", raw_text[:500])
        if markdown_table:
            logger.info("[DEBUG] Extracted Table Preview (First 500 chars):\n%s", markdown_table[:500])

        # Early exit if no text found
        if not raw_text.strip():
            return {
                "status": "error",
                "message": "Auction notice appears to contain no usable text — even after OCR."
            }

        # Truncate raw text to avoid token overflow
        truncated_text = truncate_text(raw_text)

        if assets_for_prompt:
            assets_section = f"\nAssets (extracted via OCR fallback):\n{json.dumps(assets_for_prompt, indent=2)}"
        else:
            assets_section = f"\n{markdown_table}"

        prompt = f"""
You are an expert financial analyst specializing in Indian auction notices.

Your job is to carefully extract key details from the below auction notice text.
It may contain normal paragraphs, markdown tables, or pre-parsed OCR asset JSON.
Read both carefully.

Corporate Debtor: {corporate_debtor}

Auction Notice Text:
{truncated_text}

{assets_section}

Please extract the following insights and return them as a structured JSON:

1. Extract all general details (e.g. dates, contacts, platform link, etc.) exactly as written — do not infer or modify them.
2. Use the provided markdown table or OCR asset JSON (whichever is present) to populate the `"Assets"` list.
3. One row = one asset. Do not duplicate or infer missing rows.
4. If values are missing, leave them blank — do not guess.

Additional Task:
Rank my Auctions based on three components:
- Legal Compliance
- Economical Point of View
- Market Trends

Provide:
- Individual scores for each component (0–10)
- A final score (average or weighted)
- A single-line summary of risk: "Low Risk", "No Risk", or "High Risk"
- A 4-line bullet-point summary on what reference to use for each of the three components.

Return the result in this **exact JSON format**:

{{
    "Corporate Debtor": "...",
    "Auction Date": "...",
    "Auction Time": "...",
    "Last Date for EMD Submission": "...",
    "Inspection Date": "...",
    "Inspection Time": "...",
    "Property Description": "...",
    "Auction Platform": "...",
    "Contact Email": "...",
    "Contact Mobile": "...",
    "Assets": [
        {{
            "Block Name": "...",
            "Asset Description": "...",
            "Auction Time": "...",
            "Reserve Price": "...",
            "EMD Amount": "...",
            "Incremental Bid Amount": "..."
        }}
    ],
    "Ranking": {{
        "Legal Compliance Score": 0,
        "Economical Score": 0,
        "Market Trends Score": 0,
        "Final Score": 0,
        "Risk Summary": "...",
        "Reference Summary": [
            "...",
            "...",
            "...",
            "..."
        ]
    }}
}}
"""

        logger.info(f"Prompt word count: {len(prompt.split())}, char length: {len(prompt)}")
        response = await llm.ainvoke(prompt, max_tokens=2048, temperature=0.2, top_p=0.9)

        cleaned = clean_llm_output(response.content)
        parsed = json.loads(cleaned)
        normalized = normalize_keys(parsed)

        return {
            "status": "success",
            "scanned_pdf": scanned_pdf,
            "insights": normalized
        }

    except json.JSONDecodeError as e:
        logger.warning(f"[PARSE_ERROR] Model response not in JSON: {e}")
        return {
            "status": "error",
            "message": f"Invalid JSON format. {e}",
            "insights": response.content if hasattr(response, 'content') else str(response)
        }
    except Exception as e:
        logger.error(f"[ERROR] generate_auction_insights failed: {e}")
        return {
            "status": "error",
            "message": str(e)
        }
    
@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/auction-insights")
async def get_auction_insights(auction_data: dict = Body(...)):
    """
    Generate accurate insights from auction notices.
    """
    try:
        logger.info("Auction Data received: %s", auction_data)

        corporate_debtor = (
            auction_data.get("Name of Corporate Debtor (PDF)")
            or auction_data.get("name_of_corporate_debtor_pdf_")
            or auction_data.get("corporate_debtor")
            or "Unknown Debtor"
        )

        auction_notice_url = (
            auction_data.get("Auction Notice URL")
            or auction_data.get("auction_notice_url")
            or auction_data.get("details_url")
            or "URL Not Provided"
        )

        logger.info("Debtor: %s, Notice URL: %s", corporate_debtor, auction_notice_url)
        

        summary = await generate_auction_insights(corporate_debtor, auction_notice_url)

        return summary

    except Exception as e:
        logger.error("Error generating auction insights: %s", e)
        return {"error": str(e)}
    
def initialize_services():
    global llm
    
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY not set")
    
    llm = ChatGroq(
        model="deepseek-r1-distill-llama-70b",
        temperature=0,
        api_key=groq_api_key,
    )
    logger.info("LLM initialized")


initialize_services()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)