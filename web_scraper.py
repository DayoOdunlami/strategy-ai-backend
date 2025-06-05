# web_scraper.py - Comprehensive Web Scraping with Playwright
import asyncio
from datetime import datetime
from typing import List, Dict, Set, Optional
from urllib.parse import urljoin, urlparse, urlunparse
import logging
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
import re
import time
from config import settings

logger = logging.getLogger(__name__)

class ComprehensiveWebScraper:
    """
    Comprehensive web scraper with navigation support
    Handles dropdowns, links, and multi-page scraping as requested
    """
    
    def __init__(self):
        self.visited_urls: Set[str] = set()
        self.max_pages = settings.MAX_SCRAPING_PAGES
        self.timeout = settings.SCRAPING_TIMEOUT_SECONDS * 1000  # Convert to ms
        self.delay = settings.SCRAPING_DELAY_MS
        
        logger.info("Initialized comprehensive web scraper")

    async def scrape_comprehensive(
        self,
    async def scrape_comprehensive(
        self,
        start_url: str,
        depth: str = "single",
        max_pages: int = 10,
        follow_nav_links: bool = False,
        include_downloads: bool = False
    ) -> List[Dict]:
        """
        Main scraping function with comprehensive options
        Supports single page or multi-page scraping with navigation
        """
        try:
            logger.info(f"Starting {depth} scraping of {start_url}")
            
            # Reset visited URLs for this session
            self.visited_urls = set()
            
            async with async_playwright() as p:
                # Launch browser
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        '--no-sandbox',
                        '--disable-setuid-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-accelerated-2d-canvas',
                        '--no-first-run',
                        '--no-zygote',
                        '--disable-gpu'
                    ]
                )
                
                context = await browser.new_context(
                    viewport={'width': 1280, 'height': 720},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                )
                
                results = []
                
                try:
                    if depth == "single":
                        # Single page scraping
                        page_data = await self._scrape_single_page(context, start_url)
                        if page_data:
                            results.append(page_data)
                    else:
                        # Comprehensive multi-page scraping
                        results = await self._scrape_multiple_pages(
                            context,
                            start_url,
                            max_pages,
                            follow_nav_links,
                            include_downloads
                        )
                        
                finally:
                    await browser.close()
                
                logger.info(f"Scraping completed. Retrieved {len(results)} pages")
                return results
                
        except Exception as e:
            logger.error(f"Error in comprehensive scraping: {e}")
            return []

    async def _scrape_single_page(self, context: BrowserContext, url: str) -> Optional[Dict]:
        """Scrape a single page with comprehensive content extraction"""
        try:
            page = await context.new_page()
            
            # Set timeout and navigate
            page.set_default_timeout(self.timeout)
            response = await page.goto(url, wait_until="networkidle")
            
            if not response or response.status >= 400:
                logger.warning(f"Failed to load {url}, status: {response.status if response else 'None'}")
                return None
            
            # Wait for dynamic content
            await page.wait_for_timeout(2000)
            
            # Extract comprehensive page data
            page_data = await self._extract_page_content(page, url)
            
            await page.close()
            return page_data
            
        except Exception as e:
            logger.error(f"Error scraping single page {url}: {e}")
            return None

    async def _extract_page_content(self, page: Page, url: str) -> Dict:
        """Extract comprehensive content from a page"""
        try:
            # Get basic page information
            title = await page.title()
            
            # Get meta description
            meta_description = ""
            try:
                meta_desc_element = await page.query_selector('meta[name="description"]')
                if meta_desc_element:
                    meta_description = await meta_desc_element.get_attribute("content") or ""
            except:
                pass
            
            # Extract main content using multiple strategies
            content = await self._extract_main_content(page)
            
            # Extract structured data
            structured_data = await self._extract_structured_data(page)
            
            # Extract navigation and links
            navigation_data = await self._extract_navigation(page) if content else {}
            
            # Clean and format content
            cleaned_content = self._clean_content(content)
            
            return {
                "url": url,
                "title": title,
                "meta_description": meta_description,
                "content": cleaned_content,
                "word_count": len(cleaned_content.split()),
                "structured_data": structured_data,
                "navigation": navigation_data,
                "scraped_at": datetime.now().isoformat(),
                "content_type": "webpage"
            }
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return {
                "url": url,
                "title": "Error extracting title",
                "content": f"Error extracting content: {str(e)}",
                "scraped_at": datetime.now().isoformat(),
                "error": str(e)
            }

    async def _extract_main_content(self, page: Page) -> str:
        """Extract main content using multiple content selectors"""
        
        # Priority list of content selectors (most specific to least specific)
        content_selectors = [
            'main',
            'article', 
            '[role="main"]',
            '.main-content',
            '.content',
            '#content',
            '.post-content',
            '.entry-content',
            '.page-content',
            '.article-content',
            '.content-area',
            'body'
        ]
        
        content = ""
        
        for selector in content_selectors:
            try:
                elements = await page.query_selector_all(selector)
                if elements:
                    # Get text from all matching elements
                    for element in elements:
                        element_text = await element.inner_text()
                        if element_text and len(element_text.strip()) > 100:  # Minimum content threshold
                            content += element_text + "\n\n"
                    
                    if content.strip():
                        break  # Found content, stop trying other selectors
                        
            except Exception as e:
                logger.debug(f"Error with selector {selector}: {e}")
                continue
        
        # If no content found with specific selectors, try getting all text
        if not content.strip():
            try:
                body_element = await page.query_selector('body')
                if body_element:
                    content = await body_element.inner_text()
            except:
                content = "Unable to extract content"
        
        return content

    async def _extract_structured_data(self, page: Page) -> Dict:
        """Extract structured data like headings, lists, tables"""
        structured = {}
        
        try:
            # Extract headings
            headings = []
            for level in range(1, 7):  # h1 to h6
                heading_elements = await page.query_selector_all(f'h{level}')
                for element in heading_elements:
                    text = await element.inner_text()
                    if text.strip():
                        headings.append({
                            "level": level,
                            "text": text.strip()
                        })
            structured["headings"] = headings
            
            # Extract lists
            lists = []
            list_elements = await page.query_selector_all('ul, ol')
            for element in list_elements[:5]:  # Limit to first 5 lists
                list_items = await element.query_selector_all('li')
                items = []
                for item in list_items[:10]:  # Limit items per list
                    item_text = await item.inner_text()
                    if item_text.strip():
                        items.append(item_text.strip())
                if items:
                    lists.append(items)
            structured["lists"] = lists
            
            # Extract tables (first row as sample)
            tables = []
            table_elements = await page.query_selector_all('table')
            for table in table_elements[:3]:  # Limit to first 3 tables
                rows = await table.query_selector_all('tr')
                if rows:
                    # Get header row
                    header_cells = await rows[0].query_selector_all('th, td')
                    headers = []
                    for cell in header_cells:
                        cell_text = await cell.inner_text()
                        headers.append(cell_text.strip())
                    
                    if headers:
                        tables.append({
                            "headers": headers,
                            "row_count": len(rows) - 1
                        })
            structured["tables"] = tables
            
        except Exception as e:
            logger.debug(f"Error extracting structured data: {e}")
        
        return structured

    async def _extract_navigation(self, page: Page) -> Dict:
        """Extract navigation links and menu structure"""
        navigation = {}
        
        try:
            # Extract navigation links
            nav_links = []
            nav_selectors = ['nav a', '.navigation a', '.menu a', '.nav a']
            
            for selector in nav_selectors:
                try:
                    link_elements = await page.query_selector_all(selector)
                    for element in link_elements[:20]:  # Limit links
                        href = await element.get_attribute('href')
                        text = await element.inner_text()
                        
                        if href and text and text.strip():
                            nav_links.append({
                                "text": text.strip(),
                                "href": href
                            })
                    
                    if nav_links:
                        break  # Found navigation, stop trying other selectors
                        
                except Exception as e:
                    logger.debug(f"Error extracting navigation with selector {selector}: {e}")
                    continue
            
            navigation["links"] = nav_links[:15]  # Limit to 15 nav links
            
        except Exception as e:
            logger.debug(f"Error extracting navigation: {e}")
        
        return navigation

    def _clean_content(self, content: str) -> str:
        """Clean and normalize extracted content"""
        if not content:
            return ""
        
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)  # Max 2 consecutive newlines
        content = re.sub(r'[ \t]+', ' ', content)  # Multiple spaces to single space
        
        # Remove common unwanted patterns
        unwanted_patterns = [
            r'Cookie.*?policy',
            r'Accept.*?cookies',
            r'Privacy.*?policy',
            r'Terms.*?service',
            r'© \d{4}.*?rights reserved',
            r'Follow us on.*?social media'
        ]
        
        for pattern in unwanted_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        # Trim and return
        return content.strip()

    async def _scrape_multiple_pages(
        self,
        context: BrowserContext,
        start_url: str,
        max_pages: int,
        follow_nav_links: bool,
        include_downloads: bool
    ) -> List[Dict]:
        """Scrape multiple pages following links and navigation"""
        
        results = []
        urls_to_visit = [start_url]
        self.visited_urls = set()
        
        base_domain = urlparse(start_url).netloc
        
        while urls_to_visit and len(results) < max_pages:
            current_url = urls_to_visit.pop(0)
            
            # Skip if already visited
            if current_url in self.visited_urls:
                continue
                
            self.visited_urls.add(current_url)
            
            logger.info(f"Scraping page {len(results) + 1}/{max_pages}: {current_url}")
            
            # Scrape current page
            page_data = await self._scrape_single_page(context, current_url)
            if page_data:
                results.append(page_data)
                
                # Find additional URLs if we haven't reached the limit
                if follow_nav_links and len(results) < max_pages:
                    new_urls = await self._extract_page_links(
                        context, 
                        current_url, 
                        base_domain,
                        include_downloads
                    )
                    
                    # Add new URLs to visit queue
                    for url in new_urls:
                        if url not in self.visited_urls and url not in urls_to_visit:
                            urls_to_visit.append(url)
                
                # Add delay between requests
                if self.delay > 0:
                    await asyncio.sleep(self.delay / 1000)
        
        return results

    async def _extract_page_links(
        self, 
        context: BrowserContext, 
        current_url: str, 
        base_domain: str,
        include_downloads: bool
    ) -> List[str]:
        """Extract links from a page for follow-up scraping"""
        
        valid_links = []
        
        try:
            page = await context.new_page()
            await page.goto(current_url, wait_until="networkidle")
            
            # Extract all links
            link_elements = await page.query_selector_all('a[href]')
            
            for element in link_elements:
                try:
                    href = await element.get_attribute('href')
                    if not href:
                        continue
                    
                    # Convert relative URLs to absolute
                    absolute_url = urljoin(current_url, href)
                    parsed_url = urlparse(absolute_url)
                    
                    # Filter links
                    if self._is_valid_link(parsed_url, base_domain, include_downloads):
                        # Clean the URL (remove fragments)
                        clean_url = urlunparse((
                            parsed_url.scheme,
                            parsed_url.netloc,
                            parsed_url.path,
                            parsed_url.params,
                            parsed_url.query,
                            ''  # Remove fragment
                        ))
                        
                        if clean_url not in self.visited_urls:
                            valid_links.append(clean_url)
                
                except Exception as e:
                    logger.debug(f"Error processing link: {e}")
                    continue
            
            await page.close()
            
            # Limit and prioritize links
            prioritized_links = self._prioritize_links(valid_links)
            return prioritized_links[:10]  # Return top 10 links
            
        except Exception as e:
            logger.error(f"Error extracting links from {current_url}: {e}")
            return []

    def _is_valid_link(self, parsed_url, base_domain: str, include_downloads: bool) -> bool:
        """Check if a link is valid for scraping"""
        
        # Must be same domain
        if parsed_url.netloc != base_domain:
            return False
        
        # Skip common unwanted paths
        unwanted_paths = [
            '/login', '/register', '/signup', '/auth',
            '/admin', '/wp-admin', '/user',
            '/cart', '/checkout', '/payment',
            '/search', '/ajax', '/api'
        ]
        
        path_lower = parsed_url.path.lower()
        if any(unwanted in path_lower for unwanted in unwanted_paths):
            return False
        
        # Handle file downloads
        file_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.zip', '.rar']
        has_file_extension = any(path_lower.endswith(ext) for ext in file_extensions)
        
        if has_file_extension and not include_downloads:
            return False
        
        # Skip very long URLs (likely dynamic/spam)
        if len(parsed_url.geturl()) > 200:
            return False
        
        return True

    def _prioritize_links(self, links: List[str]) -> List[str]:
        """Prioritize links for scraping based on relevance"""
        
        # Priority keywords (higher score = higher priority)
        priority_keywords = [
            'strategy', 'plan', 'policy', 'guide', 'about',
            'service', 'project', 'report', 'document',
            'overview', 'information', 'detail'
        ]
        
        def link_score(url: str) -> int:
            score = 0
            url_lower = url.lower()
            
            # Score based on priority keywords
            for keyword in priority_keywords:
                if keyword in url_lower:
                    score += 2
            
            # Prefer shorter, cleaner URLs
            if len(url) < 100:
                score += 1
            
            # Prefer URLs without many parameters
            if '?' not in url:
                score += 1
            
            return score
        
        # Sort by score (descending)
        return sorted(links, key=link_score, reverse=True)

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================

    async def validate_url(self, url: str) -> Dict[str, any]:
        """Validate URL before scraping"""
        try:
            parsed = urlparse(url)
            
            if not parsed.scheme or not parsed.netloc:
                return {"valid": False, "error": "Invalid URL format"}
            
            if parsed.scheme not in ['http', 'https']:
                return {"valid": False, "error": "Only HTTP/HTTPS URLs are supported"}
            
            # Test connectivity
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context()
                page = await context.new_page()
                
                try:
                    response = await page.goto(url, timeout=10000)
                    status = response.status if response else 0
                    
                    await browser.close()
                    
                    if status >= 400:
                        return {"valid": False, "error": f"HTTP {status} error"}
                    
                    return {
                        "valid": True,
                        "status_code": status,
                        "domain": parsed.netloc
                    }
                    
                except Exception as e:
                    await browser.close()
                    return {"valid": False, "error": f"Connection error: {str(e)}"}
                    
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}"}

    def estimate_scraping_time(self, max_pages: int, depth: str) -> int:
        """Estimate scraping time in seconds"""
        
        if depth == "single":
            return 10  # Single page takes ~10 seconds
        
        # Multi-page scraping
        base_time_per_page = 5  # seconds
        total_time = max_pages * base_time_per_page
        
        # Add time for navigation and delays
        navigation_overhead = max_pages * (self.delay / 1000)
        
        return int(total_time + navigation_overhead)

    def get_scraper_info(self) -> Dict[str, any]:
        """Get scraper configuration information"""
        return {
            "max_pages": self.max_pages,
            "timeout_seconds": self.timeout / 1000,
            "delay_ms": self.delay,
            "browser": "chromium",
            "headless": True
        }