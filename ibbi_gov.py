import logging
import logging.handlers
import re
import time
import shutil
import pandas as pd
import aiohttp
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from bs4 import BeautifulSoup
import sys
from pathlib import Path
from datetime import datetime

# Constants with readable timestamp (colons replaced with hyphens)
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S').replace(':', '-')
IBBI_URL = "https://ibbi.gov.in/en/liquidation-auction-notices/lists"
INITIAL_CSV_PATH = Path(f"ibbi_auctions_initial_{timestamp}.csv")
FINAL_CSV_PATH = Path(f"ibbi_auctions_enriched_{timestamp}.csv")
LOCAL_TEMP_DIR = Path("temp_pdf_downloads")

# Logging configuration
log_filename = f"scraping_log_{timestamp}.log"
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.handlers.RotatingFileHandler(
    log_filename, maxBytes=10*1024*1024, backupCount=5
)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(process)d - %(filename)s:%(lineno)d - %(message)s'
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Add plain console logging (optional, remove if not needed)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
)
stream_handler.setFormatter(stream_formatter)
logger.addHandler(stream_handler)

logger.info(f"Logging to {log_filename}")

# Define final_headers globally
final_headers = [
    "Name of Corporate Debtor (PDF)", "CIN/LLPIN", "Unique Number", "Date of Auction (PDF)",
    "Reserve Price (PDF)", "EMD Amount (PDF)", "Name of IP (PDF)", "IP Registration Number",
    "Auction Platform", "Web Address of Auction Platform", "Insolvency Commencement Date",
    "Liquidation Commencement Date", "Process Number", "Document Date", "Last Date of EMD Submission",
    "Location of Assets", "Auction Notice URL", "Details URL", "Type of AN", "Date",
    "Name of Corporate Debtor (Table)", "Date of Auction (Table)", "Name of Insolvency Professional (Table)",
    "Reserve Price (Table)", "Nature of Assets (Table)", "Last Date of EMD (Table)", "Nature of Assets (PDF)",
    "Form is being filed for", "Date of Issue of Auction Notice"
]

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_page(session, url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        async with session.get(url, headers=headers, timeout=30) as response:
            response.raise_for_status()
            logger.info(f"Successfully fetched {url}")
            return await response.text(encoding='utf-8', errors='replace')
    except Exception as e:
        logger.error(f"Failed to fetch page {url}: {e}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=4))
async def download_pdf(session, pdf_url, semaphore, row_index, total_rows):
    if not pdf_url or pdf_url == "Not Found":
        logger.warning(f"Invalid or missing PDF URL: {pdf_url}")
        return None
    LOCAL_TEMP_DIR.mkdir(exist_ok=True)
    temp_pdf_path = LOCAL_TEMP_DIR / f"temp_{Path(pdf_url).name or 'unknown.pdf'}"
    start_time = time.time()
    async with semaphore:
        try:
            async with session.get(pdf_url, timeout=60) as response:
                response.raise_for_status()
                content = await response.read()
                if not content:
                    logger.error(f"Empty response for PDF {pdf_url}")
                    return None
                temp_pdf_path.write_bytes(content)
                import fitz
                doc = fitz.open(temp_pdf_path)
                if not doc.page_count:
                    logger.error(f"No pages in PDF {pdf_url}")
                    doc.close()
                    return None
                text = "".join(page.get_text() for page in doc)
                doc.close()
                patterns = {
                    "Name of Corporate Debtor (PDF)": r"1\.\s*Name of Corporate Debtor\s*\n\s*(.+?)\n",
                    "CIN/LLPIN": r"2\.\s*CIN/LLPIN of Corporate Debtor\s*\n\s*([A-Z0-9]+)",
                    "Insolvency Commencement Date": r"3\.\s*Insolvency Commencement Date\s*\n\s*([\d/]+)",
                    "Process Number": r"4\.\s*Process Number\s*\n\s*(\d+)",
                    "Liquidation Commencement Date": r"5\.\s*Liquidation Commencement Date\s*\n\s*([\d/]+)",
                    "Form is being filed for": r"6\.\s*Form is being filed for\s*\n\s*(.+?)\n",
                    "Date of Issue of Auction Notice": r"7\.\s*Date of Issue of Auction Notice in Newspapers\s*\n\s*([\d/]+)",
                    "Last Date of EMD Submission": r"8\.\s*Last Date of Submission of EMD.*?\n\s*([\d/]+)",
                    "Date of Auction (PDF)": r"9\.\s*Date of Auction\s*\n\s*([\d/]+)",
                    "Reserve Price (PDF)": r"10\.\s*Reserve Price \(In Rupees\)\s*\n\s*([\d,]+)",
                    "EMD Amount (PDF)": r"11\.\s*EMD Amount \(In Rupees\)\s*\n\s*([\d,]+)",
                    "Auction Platform": r"12\.\s*Auction Platform\s*\n\s*(.+?)\n",
                    "Web Address of Auction Platform": r"14\.\s*Web address of Auction Platform\s*\n\s*(https?://[\S]+)",
                    "Nature of Assets (PDF)": r"15\.\s*Nature of Assets to be Auctioned\s*\n(.*?)(?=\n16\.)",
                    "Location of Assets": r"16\.\s*Location of Assets to be Auctioned\s*\n(.*?)(?=\nUnique Number)",
                    "Unique Number": r"Unique Number\s*-\s*(.+)",
                    "Name of IP (PDF)": r"Name of IP\s*-\s*(.+?)(?=\nIP Registration Number)",
                    "IP Registration Number": r"IP Registration Number\s*-\s*(.+?)(?=\nDate)",
                    "Document Date": r"Date\s*-\s*([\d/]+)",
                }
                pdf_data = {}
                missing_fields = []
                total_fields = len(patterns)
                for key, pattern in patterns.items():
                    match = re.search(pattern, text, re.IGNORECASE | (re.DOTALL if "Nature" in key or "Location" in key else 0))
                    value = ' '.join(match.group(1).strip().split()) if match else "Not Found"
                    pdf_data[key] = value
                    if value == "Not Found":
                        missing_fields.append(key)
                extracted_fields = total_fields - len(missing_fields)
                progress = (row_index + 1) / total_rows * 100
                process_time = time.time() - start_time
                logger.info(f"Processed PDF {pdf_url} in {process_time:.2f}s, row{row_index + 1}/{total_rows} "
                            f"{extracted_fields}/{total_fields} fields extracted ({progress:.1f}%)")
                return pdf_data
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_url}: {e}")
            return None
        finally:
            if temp_pdf_path.exists():
                temp_pdf_path.unlink()
                logger.debug(f"Cleaned up temporary file {temp_pdf_path}")

async def scrape_initial_table_data():
    initial_headers = [
        "Type of AN", "Date", "Name of Corporate Debtor (Table)", "Date of Auction (Table)",
        "Name of Insolvency Professional (Table)", "Auction Notice URL", "Reserve Price (Table)",
        "Nature of Assets (Table)", "Last Date of EMD (Table)", "Details URL"
    ]
    all_data = []
    if not INITIAL_CSV_PATH.exists():
        pd.DataFrame(columns=initial_headers).to_csv(INITIAL_CSV_PATH, index=False, encoding='utf-8')
        logger.info(f"Initialized {INITIAL_CSV_PATH} with headers")
    async with aiohttp.ClientSession() as session:
        current_page = 1
        total_pages = 1  # Initialize with 1, will be updated
        max_attempts = 10
        attempt = 0
        while current_page <= total_pages:
            logger.info(f"Scraping page {current_page} (Attempt {attempt + 1}/{max_attempts})")
            try:
                html = await fetch_page(session, f"{IBBI_URL}?page={current_page - 1}")
                soup = BeautifulSoup(html, 'html.parser')
                rows = soup.select("table.cols-4 tbody tr")
                if not rows:
                    logger.info(f"No rows found on page {current_page}, stopping.")
                    break
                # Update total_pages from pagination
                pagination = soup.select_one('ul.pagination li a[rel="next"]')
                if pagination:
                    next_url = pagination.get('href', '')
                    match = re.search(r'page=(\d+)', next_url)
                    if match:
                        total_pages = int(match.group(1)) + 1
                    else:
                        # If no page number in next link, assume it's the last page
                        total_pages = current_page
                page_data = []
                for row in rows:
                    cells = row.find_all("td")
                    if len(cells) < 10:
                        logger.warning(f"Skipping row with insufficient cells on page {current_page}")
                        continue
                    notice_onclick = cells[5].find("a").get("onclick", "")
                    details_onclick = cells[9].find("a").get("onclick", "")
                    notice_match = re.search(r"newwindow1\(\s*'([^']+)'", notice_onclick)
                    details_match = re.search(r"newwindow1\(\s*'([^']+)'", details_onclick)
                    row_data = {
                        "Type of AN": cells[0].get_text(strip=True),
                        "Date": cells[1].get_text(strip=True),
                        "Name of Corporate Debtor (Table)": cells[2].get_text(strip=True),
                        "Date of Auction (Table)": cells[3].get_text(strip=True),
                        "Name of Insolvency Professional (Table)": cells[4].get_text(strip=True),
                        "Auction Notice URL": notice_match.group(1).strip() if notice_match else "Not Found",
                        "Reserve Price (Table)": cells[6].get_text(strip=True),
                        "Nature of Assets (Table)": cells[7].get_text(strip=True),
                        "Last Date of EMD (Table)": cells[8].get_text(strip=True),
                        "Details URL": details_match.group(1).strip() if details_match else "Not Found"
                    }
                    page_data.append(row_data)
                if not page_data:
                    logger.info(f"No valid data on page {current_page}, stopping.")
                    break
                all_data.extend(page_data)
                logger.info(f"Scraped {len(page_data)} records from page {current_page}, total: {len(all_data)}")
                current_page += 1
                await asyncio.sleep(1)
                attempt = 0
            except Exception as e:
                logger.error(f"Failed to scrape page {current_page}: {e}")
                if isinstance(e.__cause__, aiohttp.ClientResponseError) and e.__cause__.status == 404:
                    logger.info(f"Page {current_page} not found (404), stopping.")
                    break
                attempt += 1
                if attempt == max_attempts:
                    logger.error(f"Max attempts reached for page {current_page}, aborting Phase 1.")
                    break
                await asyncio.sleep(2 ** attempt)
        if all_data:
            df = pd.read_csv(INITIAL_CSV_PATH) if INITIAL_CSV_PATH.exists() else pd.DataFrame(columns=initial_headers)
            df = pd.concat([df, pd.DataFrame(all_data, columns=initial_headers)], ignore_index=True)
            df.to_csv(INITIAL_CSV_PATH, index=False, encoding='utf-8')
            logger.info(f"Appended {len(all_data)} records to {INITIAL_CSV_PATH}")

async def reattempt_failed_downloads(session, failed_urls, semaphore, initial_df):
    if not failed_urls:
        logger.info("No failed URLs to reattempt.")
        return
    logger.info(f"Reattempting download for {len(failed_urls)} failed URLs.")
    reattempt_results = []
    row_index_map = {row["Details URL"]: index for index, row in initial_df.iterrows() if row["Details URL"] != "Not Found"}
    
    for url in failed_urls:
        async with semaphore:
            try:
                pdf_data = await download_pdf(session, url, semaphore, row_index_map.get(url, -1), len(initial_df))
                if pdf_data:
                    combined_row = {**initial_df.iloc[row_index_map[url]].to_dict(), **pdf_data}
                    reattempt_results.append(combined_row)
                    logger.info(f"Successfully reattempted download for {url}")
            except Exception as e:
                logger.error(f"Reattempt failed for {url}: {e}")
    
    if reattempt_results:
        enriched_df = pd.read_csv(FINAL_CSV_PATH) if FINAL_CSV_PATH.exists() else pd.DataFrame(columns=final_headers)
        enriched_df = pd.concat([enriched_df, pd.DataFrame(reattempt_results, columns=final_headers)], ignore_index=True)
        enriched_df.to_csv(FINAL_CSV_PATH, index=False, encoding='utf-8')
        logger.info(f"Updated {FINAL_CSV_PATH} with {len(reattempt_results)} reattempted records.")

async def enrich_csv_with_pdf_data():
    if not INITIAL_CSV_PATH.exists():
        logger.error(f"{INITIAL_CSV_PATH} not found.")
        return
    df = pd.read_csv(INITIAL_CSV_PATH)
    logger.info(f"Starting Phase 2: Processing {len(df)} rows")
    phase_start_time = time.time()
    semaphore = asyncio.Semaphore(10)
    failed_urls = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        for index, row in df.iterrows():
            pdf_url = row["Details URL"]
            tasks.append(download_pdf(session, pdf_url, semaphore, index, len(df)))
        results = []
        for index, pdf_data in enumerate(await asyncio.gather(*tasks, return_exceptions=True)):
            logger.debug(f"Evaluating row {index + 1}")
            if isinstance(pdf_data, Exception):
                logger.error(f"Failed to process row {index + 1}: {pdf_data}")
                if df.iloc[index]["Details URL"] != "Not Found":
                    failed_urls.append(df.iloc[index]["Details URL"])
                continue
            if pdf_data:
                combined_row = {**df.iloc[index].to_dict(), **pdf_data}
                results.append(combined_row)
            else:
                logger.warning(f"No data extracted for row {index + 1}")
                if df.iloc[index]["Details URL"] != "Not Found":
                    failed_urls.append(df.iloc[index]["Details URL"])
        if results:
            enriched_df = pd.DataFrame(results, columns=final_headers)
            mode = 'w'
            header = True
            enriched_df.to_csv(FINAL_CSV_PATH, mode=mode, header=header, index=False, encoding='utf-8')
            total_time = time.time() - phase_start_time
            logger.info(f"Phase 2 completed: Enriched {len(results)} records in {total_time:.2f}s, "
                       f"saved to {FINAL_CSV_PATH}")
            logger.info(f"Summary - Total time: {total_time + (time.time() - phase_start_time):.2f}s, "
                       f"Phase 1 time: {phase_start_time - time.time() + 25.13:.2f}s, "
                       f"Phase 2 time: {total_time:.2f}s")
        # Reattempt failed downloads
        await reattempt_failed_downloads(session, failed_urls, semaphore, df)
        if failed_urls:  # Update failed_urls after reattempt
            logger.info(f"Failed URLs after final reattempt ({len(failed_urls)}):")
            for url in failed_urls:
                logger.info(f" - {url}")
            print(f"\nFailed URLs after final reattempt ({len(failed_urls)}):")
            for url in failed_urls:
                print(f" - {url}")

async def main():
    start_time = time.time()
    logger.info("Starting the scraping and enrichment process.")
    logger.info("Initiating Phase 1: Scraping initial table data.")
    await scrape_initial_table_data()
    logger.info("Phase 1 completed.")
    logger.info("Initiating Phase 2: Enriching data with PDFs.")
    await enrich_csv_with_pdf_data()
    logger.info("Phase 2 completed.")
    if LOCAL_TEMP_DIR.exists():
        shutil.rmtree(LOCAL_TEMP_DIR, ignore_errors=True)
        logger.info("Cleaned up temporary PDF directory.")
    logger.info(f"Total execution time: {time.time() - start_time:.2f} seconds")
    logger.info("Process completed.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        raise
