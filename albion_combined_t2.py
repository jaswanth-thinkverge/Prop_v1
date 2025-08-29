import csv
import json
import os
import time
import tempfile
import shutil
import random
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from requests.adapters import HTTPAdapter
from requests.exceptions import ReadTimeout
from urllib3.util.retry import Retry
from urllib3.exceptions import NameResolutionError
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, ElementClickInterceptedException, StaleElementReferenceException
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'master_workflow_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class AlbionMasterWorkflow:
    """Complete Albion Bank auction master workflow combining Phase 1 and Phase 2."""
    
    def __init__(self):
        self.output_dir = "master_results"
        self.phase1_output_dir = "albion_results"
        self.phase2_output_dir = "phase2_results"
        
        # Create all necessary directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.phase1_output_dir, exist_ok=True)
        os.makedirs(self.phase2_output_dir, exist_ok=True)
        
        # Initialize components
        self.phase1_scraper = None
        self.phase2_extractor = None
        self.master_timestamp = None
        
    def run_complete_workflow(self):
        """Run the complete Phase 1 → Phase 2 → Master Data workflow."""
        try:
            self.master_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            logger.info("=" * 80)
            logger.info("ALBION BANK AUCTION MASTER WORKFLOW STARTED")
            logger.info("=" * 80)
            logger.info("Phase 1: Dynamic pagination scraping → albion_initial_(timestamp).csv")
            logger.info("Phase 2: Detailed URL extraction → Enhanced data")
            logger.info("Master: Combined dataset → albion_master_(timestamp).csv")
            logger.info("=" * 80)
            
            # Phase 1: Dynamic Pagination Scraping
            logger.info("🚀 Starting Phase 1: Dynamic pagination scraping")
            phase1_csv = self.run_phase1()
            
            if not phase1_csv:
                raise Exception("Phase 1 failed - no CSV generated")
            
            logger.info(f"✅ Phase 1 completed successfully: {phase1_csv}")
            
            # Phase 2: Detailed URL Extraction
            logger.info("🔍 Starting Phase 2: Detailed URL extraction")
            phase2_data, phase2_metrics = self.run_phase2(phase1_csv)
            
            if not phase2_data:
                raise Exception("Phase 2 failed - no data extracted")
            
            logger.info(f"✅ Phase 2 completed successfully: {len(phase2_data)} records processed")
            
            # Master Data Generation
            logger.info("🔗 Starting Master Data generation")
            master_csv = self.generate_master_data(phase1_csv, phase2_data)
            
            if not master_csv:
                raise Exception("Master data generation failed")
            
            logger.info(f"✅ Master Data generated successfully: {master_csv}")
            
            # Generate final report
            self.generate_master_report(phase1_csv, phase2_metrics, master_csv)
            
            logger.info("=" * 80)
            logger.info("🎉 MASTER WORKFLOW COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            
            return master_csv, phase2_metrics
            
        except Exception as e:
            logger.error(f"Master workflow failed: {e}")
            raise
    
    def run_phase1(self):
        """Execute Phase 1 dynamic pagination scraping."""
        try:
            # Initialize Phase 1 scraper
            self.phase1_scraper = AlbionDynamicScraper()
            
            # Run Phase 1 scraping
            success = self.phase1_scraper.run_full_scraping()
            
            if not success:
                return None
            
            # Find the generated CSV file
            phase1_csv = self.find_latest_csv(self.phase1_output_dir, "albion_initial_")
            return phase1_csv
            
        except Exception as e:
            logger.error(f"Phase 1 execution failed: {e}")
            return None
    
    def run_phase2(self, phase1_csv):
        """Execute Phase 2 detailed URL extraction."""
        try:
            # Initialize Phase 2 extractor with max_workers=4
            self.phase2_extractor = Phase2OptimizedExtractor(max_workers=4, batch_size=50)
            
            # Read Phase 1 CSV and extract URLs
            df = pd.read_csv(phase1_csv)
            
            if 'View Auction Link' not in df.columns:
                raise ValueError("Phase 1 CSV missing 'View Auction Link' column")
            
            # Filter valid URLs
            valid_urls = df[df['View Auction Link'].str.startswith('http', na=False)]['View Auction Link'].tolist()
            logger.info(f"Found {len(valid_urls)} valid URLs from {len(df)} total records")
            
            if not valid_urls:
                raise ValueError("No valid URLs found in Phase 1 CSV")
            
            # Process URLs using Phase 2 extractor
            extracted_data, metrics = self.phase2_extractor.process_urls_batch(valid_urls, "Master Workflow")
            
            return extracted_data, metrics
            
        except Exception as e:
            logger.error(f"Phase 2 execution failed: {e}")
            return None, None
    
    def generate_master_data(self, phase1_csv, phase2_data):
        """Generate master data by combining Phase 1 and Phase 2."""
        try:
            # Read Phase 1 data
            phase1_df = pd.read_csv(phase1_csv)
            logger.info(f"Phase 1 data loaded: {len(phase1_df)} records")
            
            # Convert Phase 2 data to DataFrame
            phase2_df = pd.DataFrame(phase2_data)
            logger.info(f"Phase 2 data prepared: {len(phase2_df)} records")
            
            # Merge on auction URL
            merged_df = pd.merge(
                phase1_df, 
                phase2_df, 
                left_on='View Auction Link', 
                right_on='url', 
                how='left'  # Keep all Phase 1 records, even if Phase 2 failed
            )
            
            # Clean up duplicate columns
            merged_df = merged_df.drop(['url', 'index'], axis=1, errors='ignore')
            
            # Fill NaN values for failed Phase 2 extractions
            phase2_columns = ['event_bank', 'property_category', 'property_sub_category', 
                            'property_description', 'country', 'state', 'district', 
                            'reserve_price_detailed', 'area', 'emd_amount']
            
            for col in phase2_columns:
                if col in merged_df.columns:
                    merged_df[col] = merged_df[col].fillna("")
            
            # Add extraction metadata
            merged_df['extraction_success_count'] = merged_df['extraction_success_count'].fillna(0)
            merged_df['master_generation_timestamp'] = self.master_timestamp
            
            # Save master data
            master_filename = os.path.join(self.output_dir, f"albion_master_{self.master_timestamp}.csv")
            merged_df.to_csv(master_filename, index=False, encoding='utf-8')
            
            logger.info(f"Master data saved: {master_filename}")
            logger.info(f"Total master records: {len(merged_df)}")
            logger.info(f"Records with Phase 2 data: {len(merged_df[merged_df['extraction_success_count'] > 0])}")
            
            return master_filename
            
        except Exception as e:
            logger.error(f"Master data generation failed: {e}")
            return None
    
    def find_latest_csv(self, directory, prefix):
        """Find the latest CSV file with given prefix."""
        try:
            files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith('.csv')]
            if not files:
                return None
            latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(directory, x)))
            return os.path.join(directory, latest_file)
        except:
            return None
    
    def generate_master_report(self, phase1_csv, phase2_metrics, master_csv):
        """Generate comprehensive master workflow report."""
        print("\n" + "=" * 80)
        print("ALBION BANK AUCTION MASTER WORKFLOW REPORT")
        print("=" * 80)
        
        # Phase 1 metrics
        try:
            phase1_df = pd.read_csv(phase1_csv)
            print(f"📊 PHASE 1 RESULTS:")
            print(f"   • Total pages scraped: {self.phase1_scraper.total_pages_processed if self.phase1_scraper else 'N/A'}")
            print(f"   • Total records extracted: {len(phase1_df)}")
            print(f"   • Output file: {os.path.basename(phase1_csv)}")
        except:
            print(f"📊 PHASE 1 RESULTS: Error reading Phase 1 data")
        
        # Phase 2 metrics
        if phase2_metrics:
            print(f"\n🔍 PHASE 2 RESULTS:")
            print(f"   • URLs processed: {phase2_metrics['total_urls_processed']}")
            print(f"   • Successful extractions: {phase2_metrics['total_successful_urls']}")
            print(f"   • Success rate: {phase2_metrics['overall_success_rate']:.1f}%")
            print(f"   • Average time per URL: {phase2_metrics['average_time_per_url']:.3f} seconds")
            print(f"   • Processing rate: {phase2_metrics['urls_per_second']:.2f} URLs/second")
        
        # Master data metrics
        try:
            master_df = pd.read_csv(master_csv)
            records_with_phase2 = len(master_df[master_df['extraction_success_count'] > 0])
            print(f"\n🔗 MASTER DATA RESULTS:")
            print(f"   • Total master records: {len(master_df)}")
            print(f"   • Records with detailed data: {records_with_phase2}")
            print(f"   • Master data completeness: {(records_with_phase2/len(master_df)*100):.1f}%")
            print(f"   • Output file: {os.path.basename(master_csv)}")
        except:
            print(f"\n🔗 MASTER DATA RESULTS: Error reading master data")
        
        print(f"\n✅ WORKFLOW STATUS: COMPLETED SUCCESSFULLY")
        print(f"📁 Check '{self.output_dir}' directory for master data")
        print(f"⏱️ Workflow timestamp: {self.master_timestamp}")
        print("=" * 80)

class AlbionDynamicScraper:
    """Fully dynamic Albion Bank auction scraper with automatic pagination."""
    
    def __init__(self):
        self.output_dir = "albion_results"
        self.driver = None
        self.wait = None
        self.data = []
        self.user_data_dir = None
        
        # Performance tracking
        self.start_time = None
        self.page_times = []
        self.extraction_times = []
        self.total_pages_processed = 0
        
        os.makedirs(self.output_dir, exist_ok=True)
        
    def setup_optimized_chrome_options(self):
        """Setup Chrome options for maximum performance."""
        self.user_data_dir = tempfile.mkdtemp(prefix="albion_chrome_")
        chrome_options = Options()
        
        # Essential options
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Performance optimizations
        chrome_options.add_argument("--disable-images")
        chrome_options.add_argument("--disable-plugins")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-background-timer-throttling")
        chrome_options.add_argument("--disable-backgrounding-occluded-windows")
        chrome_options.add_argument("--disable-renderer-backgrounding")
        chrome_options.add_argument("--memory-pressure-off")
        chrome_options.add_argument("--max_old_space_size=4096")
        chrome_options.add_argument("--disable-features=TranslateUI")
        chrome_options.add_argument("--disable-ipc-flooding-protection")
        chrome_options.add_argument("--aggressive-cache-discard")
        
        chrome_options.add_argument(f"--user-data-dir={self.user_data_dir}")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        chrome_options.add_experimental_option('prefs', {
            "download.default_directory": os.path.abspath(self.output_dir),
            "download.prompt_for_download": False,
            "profile.default_content_settings.popups": 0,
            "profile.default_content_setting_values.notifications": 2
        })
        
        return chrome_options
    
    def initialize_driver(self):
        """Initialize optimized WebDriver."""
        try:
            chrome_options = self.setup_optimized_chrome_options()
            self.driver = webdriver.Chrome(options=chrome_options)
            self.wait = WebDriverWait(self.driver, 10)
            logger.info("Albion dynamic scraper WebDriver initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            return False
    
    def setup_upcoming_tab(self):
        """Navigate to site and select upcoming tab."""
        try:
            logger.info("Navigating to Albion Bank Auctions website...")
            self.driver.get("https://albionbankauctions.com/")
            self.driver.maximize_window()
            
            # Wait for page load
            time.sleep(random.uniform(2, 3))
            
            # Select upcoming tab from dropdown
            logger.info("Setting up 'upcoming' filter...")
            try:
                dropdown_element = self.wait.until(EC.presence_of_element_located((By.ID, "sort")))
                status_dropdown = Select(dropdown_element)
                status_dropdown.select_by_value("upcoming")
                time.sleep(random.uniform(1, 2))
                logger.info("Successfully selected 'upcoming' tab")
                return True
            except TimeoutException:
                logger.warning("Could not find dropdown, continuing with default view")
                return True
                
        except Exception as e:
            logger.error(f"Error setting up upcoming tab: {e}")
            return False
    
    def extract_auction_link_optimized(self, card):
        """Extract auction link using optimized selectors."""
        try:
            # Primary selectors based on DOM analysis
            selectors = [
                ".//a[contains(@href, 'view-auctions')]",
                ".//a[contains(text(), 'View auction')]",
                ".//a[contains(@href, 'auction')]"
            ]
            
            for selector in selectors:
                try:
                    element = card.find_element(By.XPATH, selector)
                    href = element.get_attribute("href")
                    if href and href.startswith("http") and "auction" in href:
                        return href
                except NoSuchElementException:
                    continue
            
            return "Link not found"
            
        except Exception as e:
            return "Error extracting link"
    
    def extract_card_data_complete(self, card):
        """Extract complete data from property card with retry logic for stale elements."""
        max_attempts = 3
        attempt = 1
        
        while attempt <= max_attempts:
            try:
                data_item = {}
                
                # Auction ID
                try:
                    auction_id = card.find_element(By.XPATH, ".//p[contains(text(),'Auction ID')]/following-sibling::p").text.strip()
                except (NoSuchElementException, StaleElementReferenceException):
                    try:
                        auction_id = card.find_element(By.CSS_SELECTOR, "[class*='auction-id'], [class*='auction_id']").text.strip()
                    except (NoSuchElementException, StaleElementReferenceException):
                        auction_id = "N/A"
                data_item["Auction ID"] = auction_id
                
                # Heading
                try:
                    heading = card.find_element(By.TAG_NAME, "h2").text.strip()
                except (NoSuchElementException, StaleElementReferenceException):
                    try:
                        heading = card.find_element(By.CSS_SELECTOR, "h1, h3, [class*='title'], [class*='heading']").text.strip()
                    except (NoSuchElementException, StaleElementReferenceException):
                        heading = "N/A"
                data_item["Heading"] = heading
                
                # Location
                try:
                    location = card.find_element(By.CLASS_NAME, "property-location").text.strip()
                except (NoSuchElementException, StaleElementReferenceException):
                    try:
                        location = card.find_element(By.CSS_SELECTOR, "[class*='location'], [class*='address']").text.strip()
                    except (NoSuchElementException, StaleElementReferenceException):
                        location = "N/A"
                data_item["Location"] = location
                
                # Bank Name
                try:
                    bank_name = card.find_element(By.XPATH, ".//p[contains(text(),'Bank Name')]/following-sibling::div").text.strip()
                except (NoSuchElementException, StaleElementReferenceException):
                    try:
                        bank_name = card.find_element(By.CSS_SELECTOR, "[class*='bank'], [class*='lender']").text.strip()
                    except (NoSuchElementException, StaleElementReferenceException):
                        bank_name = "N/A"
                data_item["Bank Name"] = bank_name
                
                # Reserve Price
                try:
                    reserve_price = card.find_element(By.CLASS_NAME, "reserve_price").text.strip()
                except (NoSuchElementException, StaleElementReferenceException):
                    try:
                        reserve_price = card.find_element(By.CSS_SELECTOR, "[class*='price'], [class*='amount']").text.strip()
                    except (NoSuchElementException, StaleElementReferenceException):
                        reserve_price = "N/A"
                data_item["Reserve Price"] = reserve_price
                
                # Auction Date
                try:
                    auction_date = card.find_element(By.XPATH, ".//p[contains(text(),'Auction Date')]/following-sibling::p").text.strip()
                except (NoSuchElementException, StaleElementReferenceException):
                    try:
                        auction_date = card.find_element(By.CSS_SELECTOR, "[class*='date'], [class*='time']").text.strip()
                    except (NoSuchElementException, StaleElementReferenceException):
                        auction_date = "N/A"
                data_item["Auction Date"] = auction_date
                
                # View Auction Link
                data_item["View Auction Link"] = self.extract_auction_link_optimized(card)
                
                return data_item
            
            except StaleElementReferenceException as e:
                if attempt == max_attempts:
                    logger.error(f"Failed extracting card data after {max_attempts} attempts: {e}")
                    return None
                logger.warning(f"Stale element on attempt {attempt}, retrying...")
                time.sleep(0.5)  # Short delay to allow DOM to stabilize
                attempt += 1
    
    def scrape_all_pages_until_disabled(self):
        """Scrape all pages until next button is disabled."""
        try:
            self.start_time = time.time()
            logger.info("Starting Albion Bank Auction Dynamic Scraping")
            logger.info("Will process all pages until next button is disabled")
            
            # Setup upcoming tab
            if not self.setup_upcoming_tab():
                return False
            
            # Start pagination scraping
            page = 1
            total_extracted = 0
            total_cards_found = 0
            
            while True:
                page_start_time = time.time()
                
                try:
                    logger.info(f"Scraping page {page}...")
                    
                    # Wait for page content to load
                    time.sleep(random.uniform(1, 2))
                    
                    # Find property cards
                    try:
                        cards = self.wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "property-card")))
                        cards_found = len(cards)
                        total_cards_found += cards_found
                        logger.info(f"Found {cards_found} property cards on page {page}")
                    except TimeoutException:
                        logger.error(f"No property cards found on page {page}")
                        break
                    
                    if cards_found == 0:
                        logger.warning(f"No cards found on page {page} - ending scraping")
                        break
                    
                    # Extract data from each card
                    extraction_start = time.time()
                    page_extracted = 0
                    
                    for i in range(cards_found):
                        try:
                            # Re-locate the card to avoid stale references
                            cards = self.wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "property-card")))
                            card = cards[i]  # Get fresh card reference
                            card_data = self.extract_card_data_complete(card)
                            if card_data and card_data["View Auction Link"] not in ["Link not found", "Error extracting link"]:
                                self.data.append(card_data)
                                page_extracted += 1
                        except Exception as e:
                            logger.warning(f"Error processing card {i+1} on page {page}: {e}")
                            continue
                    
                    extraction_time = time.time() - extraction_start
                    self.extraction_times.append(extraction_time)
                    
                    total_extracted += page_extracted
                    page_total_time = time.time() - page_start_time
                    self.page_times.append(page_total_time)
                    
                    logger.info(f"Page {page}: {page_extracted}/{cards_found} records extracted in {page_total_time:.2f}s")
                    
                    # Try to navigate to next page
                    try:
                        next_button_selectors = [
                            ".pagination a.next",
                            "a.next", 
                            ".pagination .next",
                            "a[aria-label='Next']",
                            ".pagination a:contains('Next')",
                            ".pagination a:last-child"
                        ]
                        
                        next_button = None
                        for selector in next_button_selectors:
                            try:
                                if ":contains(" in selector:
                                    xpath_selector = ".//div[contains(@class, 'pagination')]//a[contains(text(), 'Next') or contains(text(), '>>') or contains(text(), '›')]"
                                    next_button = self.driver.find_element(By.XPATH, xpath_selector)
                                else:
                                    next_button = self.driver.find_element(By.CSS_SELECTOR, selector)
                                break
                            except NoSuchElementException:
                                continue
                        
                        if next_button:
                            button_class = next_button.get_attribute("class") or ""
                            button_disabled = next_button.get_attribute("disabled")
                            
                            if "disabled" in button_class.lower() or button_disabled:
                                logger.info("✅ Next button is disabled - reached final page")
                                logger.info(f"Total pages processed: {page}")
                                break
                            
                            try:
                                self.driver.execute_script("arguments[0].click();", next_button)
                                logger.info(f"Successfully navigated to page {page + 1}")
                                page += 1
                                time.sleep(random.uniform(1, 2))
                            except Exception as e:
                                logger.error(f"Error clicking next button: {e}")
                                break
                        else:
                            logger.info("✅ No next button found - reached final page")
                            break
                            
                    except Exception as e:
                        logger.error(f"Error during pagination: {e}")
                        break
                        
                except Exception as e:
                    logger.error(f"Error on page {page}: {e}")
                    break
            
            self.total_pages_processed = page
            total_time = time.time() - self.start_time
            
            # Generate performance report
            self.generate_performance_report(total_time, total_extracted, total_cards_found, page)
            
            return total_extracted > 0
            
        except Exception as e:
            logger.error(f"Error during dynamic scraping: {e}")
            return False
    
    def generate_performance_report(self, total_time, total_extracted, total_cards_found, pages_processed):
        """Generate comprehensive performance report."""
        logger.info("=" * 70)
        logger.info("ALBION DYNAMIC SCRAPER PERFORMANCE RESULTS")
        logger.info("=" * 70)
        
        logger.info(f"Total Execution Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        logger.info(f"Pages Processed: {pages_processed}")
        logger.info(f"Total Cards Found: {total_cards_found}")
        logger.info(f"Total Records Extracted: {total_extracted}")
        
        if total_cards_found > 0:
            success_rate = (total_extracted/total_cards_found)*100
            logger.info(f"Extraction Success Rate: {success_rate:.1f}%")
        
        if self.page_times:
            avg_page_time = sum(self.page_times) / len(self.page_times)
            logger.info(f"Average Time per Page: {avg_page_time:.2f} seconds")
        
        if total_time > 0:
            pages_per_minute = (pages_processed / total_time) * 60
            records_per_minute = (total_extracted / total_time) * 60
            logger.info(f"Processing Rate: {pages_per_minute:.1f} pages/minute")
            logger.info(f"Extraction Rate: {records_per_minute:.1f} records/minute")
        
        logger.info("=" * 70)
    
    def save_results_with_timestamp(self):
        """Save results with albion_initial_(timestamp).csv format."""
        if not self.data:
            logger.warning("No data to save")
            return False
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save as requested format: albion_initial_(timestamp).csv
            csv_filename = os.path.join(self.output_dir, f"albion_initial_{timestamp}.csv")
            fieldnames = ["Auction ID", "Heading", "Location", "Bank Name", "Reserve Price", "Auction Date", "View Auction Link"]
            
            with open(csv_filename, "w", newline='', encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.data)
            
            logger.info("Results saved successfully:")
            logger.info(f"  📊 Main Data: {csv_filename}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.driver:
                self.driver.quit()
                logger.info("WebDriver closed successfully")
        except Exception as e:
            logger.warning(f"Error closing WebDriver: {e}")
        
        if self.user_data_dir and os.path.exists(self.user_data_dir):
            try:
                shutil.rmtree(self.user_data_dir)
                logger.info("Cleaned up temporary directory")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {e}")
    
    def run_full_scraping(self):
        """Run the complete Albion Bank auction scraping."""
        try:
            if not self.initialize_driver():
                return False
            
            if not self.scrape_all_pages_until_disabled():
                return False
            
            if not self.save_results_with_timestamp():
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Albion scraping failed: {e}")
            return False
        
        finally:
            self.cleanup()

class Phase2OptimizedExtractor:
    """Optimized Phase 2 extractor with fixed metrics and proven patterns only."""
    
    def __init__(self, max_workers=4, batch_size=50):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.output_dir = "phase2_results"
        
        # Use ONLY the proven patterns - no additional fields
        self.PROVEN_PATTERNS = {
            'event_bank': 'Event Bank',
            'property_category': 'Property Category',
            'property_sub_category': 'Property Sub Category',
            'property_description': 'Property Description',
            'country': 'Country',
            'state': 'State',
            'district': 'District',
            'reserve_price_detailed': 'Reserve Price',
            'area': 'Area',
            'emd_amount': 'EMD Amount'
        }
        
        # Performance tracking
        self.failed_urls = []
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def create_optimized_session(self):
        """Create optimized session with connection pooling."""
        session = requests.Session()
        
        adapter = HTTPAdapter(
            pool_connections=20,
            pool_maxsize=20,
            max_retries=Retry(
                total=5,  # Increased retries
                backoff_factor=1.0,  # Longer backoff
                status_forcelist=[429, 500, 502, 503, 504],  # Handle rate-limiting and server errors
                allowed_methods=["GET"],
                raise_on_status=False
            )
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Keep-Alive': 'timeout=30, max=100'
        })
        
        return session
    
    def extract_field_optimized(self, soup, search_text):
        """Optimized field extraction using proven pattern."""
        try:
            # Exact match first
            divs = soup.find_all('div', string=search_text, limit=3)
            for div in divs:
                next_sibling = div.find_next_sibling('div')
                if next_sibling:
                    text = next_sibling.get_text(strip=True)
                    if text:
                        return text
            
            # Partial match fallback
            divs = soup.find_all('div', string=lambda text: text and search_text in text, limit=5)
            for div in divs:
                next_sibling = div.find_next_sibling('div')
                if next_sibling:
                    text = next_sibling.get_text(strip=True)
                    if text:
                        return text
            return ""
        except Exception:
            return ""
    
    def extract_single_url_parallel(self, url_data):
        """Extract data from single URL for parallel processing with enhanced timeout handling."""
        index, url = url_data
        thread_name = threading.current_thread().name
        
        if not url or not str(url).startswith('http'):
            logger.warning(f"[{thread_name}] Invalid URL at index {index}")
            return index, {field: "" for field in self.PROVEN_PATTERNS.keys()}, 0
        
        session = self.create_optimized_session()
        start_time = time.time()
        
        max_attempts = 3
        attempt = 1
        
        while attempt <= max_attempts:
            try:
                time.sleep(0.25)  # Rate-limiting delay
                response = session.get(url, timeout=10)  # Increased timeout to 10s
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                extracted_data = {}
                for field_name, search_text in self.PROVEN_PATTERNS.items():
                    value = self.extract_field_optimized(soup, search_text)
                    extracted_data[field_name] = value
                
                duration = time.time() - start_time
                successful_fields = sum(1 for v in extracted_data.values() if v)
                
                logger.info(f"[{thread_name}] Index {index}: {duration:.2f}s - {successful_fields}/{len(self.PROVEN_PATTERNS)} fields")
                return index, extracted_data, successful_fields
                
            except (NameResolutionError, ReadTimeout) as e:
                if attempt == max_attempts:
                    logger.error(f"[{thread_name}] Error at index {index} after {max_attempts} attempts: {e}")
                    self.failed_urls.append((index, url, str(e)))
                    return index, {field: "" for field in self.PROVEN_PATTERNS.keys()}, 0
                logger.warning(f"[{thread_name}] {type(e).__name__} at index {index}, retrying after {2 ** attempt}s...")
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                attempt += 1
            except Exception as e:
                logger.error(f"[{thread_name}] Error at index {index}: {e}")
                self.failed_urls.append((index, url, str(e)))
                return index, {field: "" for field in self.PROVEN_PATTERNS.keys()}, 0
            finally:
                if attempt == max_attempts:
                    session.close()
    
    def process_urls_batch(self, urls, batch_name=""):
        """Process URLs in batch with FIXED metrics dictionary."""
        logger.info(f"Processing batch {batch_name}: {len(urls)} URLs")
        
        url_data = [(i, url) for i, url in enumerate(urls)]
        results = {}
        batch_start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self.extract_single_url_parallel, data): data[0] 
                for data in url_data
            }
            
            for future in as_completed(future_to_index):
                index, data, success_count = future.result()
                results[index] = {
                    'data': data,
                    'success_count': success_count
                }
        
        batch_total_time = time.time() - batch_start_time
        
        # Calculate metrics with CORRECT keys for the report
        successful_urls = sum(1 for r in results.values() if r['success_count'] > 0)
        total_successful_fields = sum(r['success_count'] for r in results.values())
        max_possible_fields = len(urls) * len(self.PROVEN_PATTERNS)
        avg_time_per_url = batch_total_time / len(urls) if len(urls) > 0 else 0
        success_rate = (total_successful_fields / max_possible_fields) * 100 if max_possible_fields > 0 else 0
        
        # Create metrics dictionary
        batch_metrics = {
            'total_urls_processed': len(urls),
            'total_successful_urls': successful_urls,
            'overall_success_rate': success_rate,
            'total_processing_time': batch_total_time,
            'average_time_per_url': avg_time_per_url,
            'urls_per_second': len(urls) / batch_total_time if batch_total_time > 0 else 0,
            'failed_urls_count': len(self.failed_urls)
        }
        
        # Convert results to list format
        extracted_data = []
        for i in range(len(urls)):
            if i in results:
                row_data = {'url': urls[i], 'index': i}
                row_data.update(results[i]['data'])
                row_data['extraction_success_count'] = results[i]['success_count']
                extracted_data.append(row_data)
            else:
                row_data = {'url': urls[i], 'index': i}
                row_data.update({field: "" for field in self.PROVEN_PATTERNS.keys()})
                row_data['extraction_success_count'] = 0
                extracted_data.append(row_data)
        
        logger.info(f"Batch {batch_name} completed: {avg_time_per_url:.3f}s per URL, {success_rate:.1f}% success")
        
        return extracted_data, batch_metrics

def main():
    """Main execution function for the complete master workflow."""
    print("ALBION BANK AUCTION MASTER WORKFLOW")
    print("=" * 80)
    print("🚀 Phase 1: Dynamic pagination scraping → albion_initial_(timestamp).csv")
    print("🔍 Phase 2: Detailed URL extraction → Enhanced auction data")
    print("🔗 Master: Combined comprehensive dataset → albion_master_(timestamp).csv")
    print("⚡ Fully automated workflow with error handling")
    print("✅ Failed extractions included with empty fields")
    print("=" * 80)
    print("\n🎯 Starting complete master workflow...")
    
    try:
        # Initialize and run master workflow
        workflow = AlbionMasterWorkflow()
        master_csv, phase2_metrics = workflow.run_complete_workflow()
        
        print(f"\n🎉 MASTER WORKFLOW COMPLETED SUCCESSFULLY!")
        print(f"📊 Master data saved: {os.path.basename(master_csv)}")
        print(f"📁 Check 'master_results' directory for final output")
        print(f"⚡ Phase 2 performance: {phase2_metrics['average_time_per_url']:.3f}s per URL")
        print(f"✅ Overall success rate: {phase2_metrics['overall_success_rate']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"\n❌ MASTER WORKFLOW FAILED: {e}")
        logger.error(f"Master workflow failed: {e}")
        return False

if __name__ == "__main__":
    main()
