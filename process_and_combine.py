import pandas as pd
import glob
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_and_combine():
    """Combine the latest data from albion_combined_t2.py and ibbi_gov.py."""
    combined_data = []
    output_dir = "auction_exports"
    os.makedirs(output_dir, exist_ok=True)

    # Standard columns for combined output
    standard_columns = [
        "Auction ID/CIN/LLPIN", 
        "Bank/Organisation Name", 
        "Location-City/District/address",
        "_Auction date",
        "_Last Date of EMD Submission",
        "_Reserve Price", 
        "EMD Amount",
        "Nature of Assets",
        "Details URL",
        "Auction Notice URL",
        "Source"
    ]

    # Find latest files
    albion_files = glob.glob(f"{output_dir}/albion_master_*.csv")
    ibbi_files = glob.glob(f"{output_dir}/ibbi_auctions_enriched_*.csv")

    # Process Albion data
    if albion_files:
        try:
            latest_albion = max(albion_files, key=os.path.getctime)
            albion_df = pd.read_csv(latest_albion)
            logger.info(f"Loaded Albion data: {latest_albion} with {len(albion_df)} records")

            # Ensure date columns are in datetime format, expecting DD-MM-YYYY
            albion_df["Auction Date"] = pd.to_datetime(albion_df["Auction Date"], format="%d/%m/%Y", errors="coerce").dt.date
            albion_df["Last Date of EMD Submission"] = albion_df["Auction Date"] - pd.Timedelta(days=2)

            # Convert columns to numeric, "Not Found" and other non-numeric values become NaN
            albion_df['reserve_price_detailed'] = pd.to_numeric(albion_df['reserve_price_detailed'].str.replace(',', ''), errors='coerce')
            albion_df['emd_amount'] = pd.to_numeric(albion_df['emd_amount'].str.replace(',', ''), errors='coerce')

            # Map columns
            albion_df = albion_df.rename(columns={
                "Auction ID": "Auction ID/CIN/LLPIN",
                "Bank Name": "Bank/Organisation Name",
                "Location": "Location-City/District/address",
                "Auction Date": "_Auction date",
                "Last Date of EMD Submission": "_Last Date of EMD Submission",
                "reserve_price_detailed": "_Reserve Price",
                "emd_amount": "EMD Amount",
                "property_sub_category": "Nature of Assets",
                'View Auction Link': "Details URL"
            })

            # Add source column
            albion_df["Source"] = "Albion"
            albion_df["Auction Notice URL"] = "URL 2_if available"  # Placeholder for Auction Notice URL    

            # Convert date columns to string format DD-MM-YYYY to preserve in CSV
            albion_df["_Auction date"] = albion_df["_Auction date"].apply(lambda x: x.strftime('%d-%m-%Y') if pd.notnull(x) else "-")
            albion_df["_Last Date of EMD Submission"] = albion_df["_Last Date of EMD Submission"].apply(lambda x: x.strftime('%d-%m-%Y') if pd.notnull(x) else "-")

            # Select standard columns, fill missing with "-"
            albion_df = albion_df.reindex(columns=standard_columns, fill_value="-")
            combined_data.append(albion_df)
        except Exception as e:
            logger.error(f"Failed to process Albion data: {e}")

    # Process IBBI data
    if ibbi_files:
        try:
            latest_ibbi = max(ibbi_files, key=os.path.getctime)
            ibbi_df = pd.read_csv(latest_ibbi)
            logger.info(f"Loaded IBBI data: {latest_ibbi} with {len(ibbi_df)} records")

            # Ensure date columns are in datetime format, expecting DD-MM-YYYY
            ibbi_df["Last Date of EMD (Table)"] = pd.to_datetime(ibbi_df["Last Date of EMD (Table)"], format="%d-%m-%Y", errors="coerce").dt.date
            ibbi_df["Date of Auction (Table)"] = pd.to_datetime(ibbi_df["Date of Auction (Table)"], format="%d-%m-%Y", errors="coerce").dt.date

            # Convert columns to numeric, "Not Found" and other non-numeric values become NaN
            ibbi_df['Reserve Price (PDF)'] = pd.to_numeric(ibbi_df['Reserve Price (PDF)'], errors='coerce')
            ibbi_df['EMD Amount (PDF)'] = pd.to_numeric(ibbi_df['EMD Amount (PDF)'], errors='coerce')

            # Map columns
            ibbi_df = ibbi_df.rename(columns={
                "CIN/LLPIN": "Auction ID/CIN/LLPIN",
                "Name of Corporate Debtor (Table)": "Bank/Organisation Name",
                "Location of Assets": "Location-City/District/address",
                "Date of Auction (Table)": "_Auction date",
                "Last Date of EMD (Table)": "_Last Date of EMD Submission",
                "Reserve Price (PDF)": "_Reserve Price",
                "EMD Amount (PDF)": "EMD Amount",
                "Nature of Assets (PDF)": "Nature of Assets",
                "Details URL": "Details URL",
                'Auction Notice URL': 'Auction Notice URL'
            })

            # Add source column
            ibbi_df["Source"] = "IBBI_Auctions"

            # Convert date columns to string format DD-MM-YYYY to preserve in CSV
            ibbi_df["_Auction date"] = ibbi_df["_Auction date"].apply(lambda x: x.strftime('%d-%m-%Y') if pd.notnull(x) else "-")
            ibbi_df["_Last Date of EMD Submission"] = ibbi_df["_Last Date of EMD Submission"].apply(lambda x: x.strftime('%d-%m-%Y') if pd.notnull(x) else "-")

            # Select standard columns, fill missing with "-"
            ibbi_df = ibbi_df.reindex(columns=standard_columns, fill_value="-")
            combined_data.append(ibbi_df)
        except Exception as e:
            logger.error(f"Failed to process IBBI data: {e}")

    # Combine and save
    if combined_data:
        final_df = pd.concat(combined_data, ignore_index=True)
        # Include date and timestamp in filename
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"{output_dir}/combined_auctions_{timestamp_str}.csv"
        final_df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"Combined data saved to: {output_file} with {len(final_df)} records")
        return output_file
    else:
        logger.error("No data to combine.")
        return None

if __name__ == "__main__":
    process_and_combine()


    
