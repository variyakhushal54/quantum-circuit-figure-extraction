import time
import requests
import os
import random

# This class handles downloading PDFs from arXiv
# It ensures we don't hit the server too fast (rate limiting)
class ArxivDownloader:
    def __init__(self, download_dir="data/temp_pdfs", delay=(10, 20)):
        """
        Initialize the downloader.
        :param download_dir: Folder to save PDFs temporarily (not used if we fetch bytes directly)
        :param delay: Time to wait between requests (min, max) seconds to be polite to arXiv
        """
        self.download_dir = download_dir
        self.delay = delay # Tuple (min, max) or float
        
        # Create the directory if it doesn't exist yet
        os.makedirs(self.download_dir, exist_ok=True)
        self.last_request_time = 0

    def fetch_pdf_bytes(self, arxiv_id):
        """
        Download the PDF file for a specific arXiv ID content directly into memory.
        This avoids creating temporary files on the disk for every paper.
        
        :param arxiv_id: The ID of the paper (e.g. '2501.00123')
        :return: Expected PDF content as bytes, or None if it failed.
        """
        # --- Rate Limiting Strategy ---
        # Calculate how long it has been since our last request
        elapsed = time.time() - self.last_request_time
        
        # Pick a random delay in the range so requests are not evenly spaced
        if isinstance(self.delay, tuple):
            target_delay = random.uniform(self.delay[0], self.delay[1])
        else:
            target_delay = self.delay
            
        # If we are too fast, wait for the remaining time
        if elapsed < target_delay:
            wait_time = target_delay - elapsed
            print(f"  ⏳ Waiting {wait_time:.1f}s for arXiv rate limit...")
            time.sleep(wait_time)

        # Build the arXiv PDF URL
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        # Identify the tool and a contact address, as arXiv's crawling guidance asks.
        # Set ARXIV_CONTACT_EMAIL to add a contact address to the User-Agent.
        contact = os.environ.get('ARXIV_CONTACT_EMAIL', '').strip()
        user_agent = 'QuantumCircuitExtraction/1.0 (academic project'
        if contact:
            user_agent += f'; contact: {contact}'
        user_agent += ')'
        headers = {
            'User-Agent': user_agent,
            'Accept': 'application/pdf'
        }

        try:
            print(f"  ⬇ Fetching {url}...")
            # Make the web request
            response = requests.get(url, headers=headers, timeout=30)
            
            # Update the last request time for next check
            self.last_request_time = time.time()

            if response.status_code == 200:
                print(f"  ✓ Fetched {len(response.content)} bytes")
                return response.content
            else:
                print(f"  ✗ Failed to fetch {arxiv_id}: Status {response.status_code}")
                return None
        except Exception as e:
            print(f"  ✗ Exception fetching {arxiv_id}: {e}")
            return None

    def cleanup(self, filepath):
        """
        Helper to delete a file if needed.
        Used to keep the disk clean.
        """
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
