# playwright package allows for browser based automation
# allows to scroll and dynamically load the page as a browser
from playwright.sync_api import sync_playwright, Page, Locator
# sync_playwright is the synchronus version that runs sequentially on a single thread

# allows to parse html content
from bs4 import BeautifulSoup
# allows to sleep 
import time
import re
from typing import List, Dict, Optional, Any
import json
import os


URL_MAIN = 'https://resources.arper.com/en/search-media-library'

URL_PROJECTS = 'https://resources.arper.com/en/search-media-library?rawData.c595___ThronContent.06lwni_e7e5422eaa0d4cbdb91cc9ad4debcfe0.06lwni_0ccca86e3adf441dad1bef1a8b24faca.06lwni_0ccca86e3adf441dad1bef1a8b24faca_tags=fe576796cb624cdb91e194d30e76d650&dt-rawData.c595___ThronContent.06lwni_e7e5422eaa0d4cbdb91cc9ad4debcfe0.06lwni_0ccca86e3adf441dad1bef1a8b24faca.06lwni_0ccca86e3adf441dad1bef1a8b24faca_tags=string_array'

URL_STUDIO_PICS = "https://resources.arper.com/en/search-media-library?rawData.c595___ThronContent.06lwni_e7e5422eaa0d4cbdb91cc9ad4debcfe0.06lwni_0ccca86e3adf441dad1bef1a8b24faca.06lwni_0ccca86e3adf441dad1bef1a8b24faca_tags=eb09f3f8c54446e2bd16a1be33d8049c&dt-rawData.c595___ThronContent.06lwni_e7e5422eaa0d4cbdb91cc9ad4debcfe0.06lwni_0ccca86e3adf441dad1bef1a8b24faca.06lwni_0ccca86e3adf441dad1bef1a8b24faca_tags=string_array"

URL_SET_LOCATION = "https://resources.arper.com/en/search-media-library?rawData.c595___ThronContent.06lwni_e7e5422eaa0d4cbdb91cc9ad4debcfe0.06lwni_0ccca86e3adf441dad1bef1a8b24faca.06lwni_0ccca86e3adf441dad1bef1a8b24faca_tags=2ab9a40d0cf040a89720bdf58263a816&dt-rawData.c595___ThronContent.06lwni_e7e5422eaa0d4cbdb91cc9ad4debcfe0.06lwni_0ccca86e3adf441dad1bef1a8b24faca.06lwni_0ccca86e3adf441dad1bef1a8b24faca_tags=string_array"

URL_SHOWROOMS = "https://resources.arper.com/en/search-media-library?rawData.c595___ThronContent.06lwni_e7e5422eaa0d4cbdb91cc9ad4debcfe0.06lwni_e7e5422eaa0d4cbdb91cc9ad4debcfe0_tags=10dcd7bfade9493eb5808ca3a37dfb91&dt-rawData.c595___ThronContent.06lwni_e7e5422eaa0d4cbdb91cc9ad4debcfe0.06lwni_e7e5422eaa0d4cbdb91cc9ad4debcfe0_tags=string_array"

def run():
    # using sync_playwright() function we can load a browser and specify the engine we would like to use
    with sync_playwright() as p:
        # ----- Setup of browser/page -----
        
        # with sync_playwright allows to use the function when it is needed and this will end the browser when the program halts
        # launching a new browser instance
        browser = p.chromium.launch(headless=False)
        # opens a page/tab for browsing
        page = browser.new_page()
        # going to the required page
        page.goto(URL_STUDIO_PICS)
        print(f"Loading page:{URL_STUDIO_PICS}")
        
        # ----- end of setup of browser/page -----
          
        # Getting no of products on the whole listing 
        noOfProducts = get_total_media_count(page) 
        print(f"Total items on this URL: {noOfProducts}")
        
        items_needed = noOfProducts if noOfProducts<1300 else 2000
        
         
        scroll_for_n_items(page, items_needed)
        
        # all gallery items
        gallery_cards = page.locator("div.gallery-item.gallery-card-item")
        
        gallery_cards_items = gallery_cards.all()
        print(f"number of gallery cards: {len(gallery_cards_items)}")
        
        
        gallery_items = []
        
        for i in range(items_needed):
            # print(i)
            data = get_dict_for_card_locator(gallery_cards.nth(i))
            # print(data)
            gallery_items.append(data)
        
        
        directory = "data/jsonl/"
        save_to_jsonl(gallery_items,"render-studio-pictures.jsonl",directory)

        browser.close()

def get_dict_for_card_locator(gallery_card: Locator) -> Dict[str,str]:
    """
    A function that will use a playwright Locator and find the title and image of a card displayed on the page.
    
    Args:
        gallery_card: This would be a playwright Locator that we can use to get the important information of the gallery card
    Returns:
        dict:
            requiredDict = {
                "title": "",
                "image_url": ""
            }
    """
    requiredDict = {
        "title": "",
        "image_url": ""
    }
    
    img = gallery_card.locator("div.v-image__image--contain[style*='background-image']")
        
    if img:
        # print(f"image found! {img}")
        style = img.evaluate("el => getComputedStyle(el).backgroundImage")
        match = re.search(r'url\("?([^")]+)"?\)', style)
        if match:
            image_url = match.group(1)
            # print(f"Got image url{image_url}")
            requiredDict['image_url'] = image_url
    
    title = gallery_card.locator("div.metadata-name.s-card-title-text")
    if title:
        # print(f"Got title:{title.inner_text()}")
        
        requiredDict['title'] = title.inner_text()
    
    return requiredDict


def scroll_for_n_items(page: Page, n:int, increment:int=350, wait_per_step:float=0.5):
    """
    Slowly scrolls the page until at least n items are loaded.
    
    Args:
        page: Playwright page object
        n: number of items that need to be loaded
        increment: pixels to scroll per step
        wait_per_step: seconds to wait after each mini-scroll
    """
    last_items = 0
    scroll_attempts = 0
    max_attempts = 100  # safety cap

    print(f"Scrolling slowly to load {n} items...")

    starting_pos = 0
    
    while scroll_attempts < max_attempts:
        scroll_attempts += 1
        
        # Get current scroll height
        current_height = page.evaluate("document.body.scrollHeight")

        # Scroll uniformly from top to current bottom
        for y in range(starting_pos, current_height, increment):
            page.evaluate(f"window.scrollTo(0, {y})")
            time.sleep(wait_per_step)  # slow step

        starting_pos = current_height
        
        # Wait a bit at current bottom to allow lazy-load
        time.sleep(wait_per_step * 2)

        # Count loaded gallery items
        total_items = page.evaluate(
            "document.querySelectorAll('div.gallery-item.gallery-card-item').length"
        )
        print(f"{total_items} items loaded so far...")

        if total_items >= n+150:
            print(f"Reached target of {n} items.")
            break

        if total_items == last_items:
            print("No more new items loaded, stopping scroll.")
            break

        last_items = total_items



def get_total_gallery_items_on_page(page: Page) -> int:
    """
    Function to get the count of how many media items are loaded on current page. Function requires you to pass in the page instance and it will wait for the page to load a gallery-item and return the number items on the page.
    
    Args:
        page: a playwright page
    Returns:
        The integer value of total products on that page
    """

    # print("Checking if items on the page have loaded...")
    # page.wait_for_selector("div.gallery-item.gallery-card-item")
    # print("Items are on page!")
    
    html = page.content()
    
    soup = BeautifulSoup(html, 'html.parser')

    try:
        gallery_cards = soup.find_all('div', class_="gallery-item gallery-card-item")
    except Exception as e:
        print(f"Failed to get gallery items with error: {e}")
        return 0
    
    try:
        count = len(gallery_cards)
    except Exception as e:
        print(f"Failed to get count of items: {e}")
    
    return(count)


def get_total_media_count(page: Page) -> int:
    """
    Function to get the count of how many media items can get loaded from that library page. Function requires you to pass in the page instance and it will wait for the page to load the total products for that category.
    
    Args:
        page: a playwright page
    Returns:
        The integer value of the products on for that category
    """
    print('\nAttempting to get total number of products on page.')
    # print("Waiting for total count on page to load....")
    page.wait_for_selector("span.title-total-count")
    # print("Total count loaded on page!")
    
    # loading current html using bs4
    html = page.content()
    soup = BeautifulSoup(html, 'html.parser')
    
    try:
        count = soup.find('span', class_="title-total-count").text.strip()
    except Exception as e:
        print(f"Failed to get total count of products with error: {e}")
        return 0
    
    try:
        count = int(count)
    except Exception as e:
        print(f"Failed to convert count to int with error: {e}")
    
    return count


def save_to_jsonl(data: List[Dict[str, Optional[Any]]], filename: str, directory: str):
    """
    Saves a list of dictionaries to a JSON Lines (.jsonl) file in a specified directory.
    
    Args:
        data: A list of dictionaries, where each dictionary represents a product.
        filename: The name of the file to save the data to (e.g., 'products.jsonl').
        directory: The directory path where the file should be saved.
    """
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Construct the full file path
    full_path = os.path.join(directory, filename)
    
    with open(full_path, 'a', encoding='utf-8') as f:
        for product in data:
            json_line = json.dumps(product)
            f.write(json_line + '\n')
            
    print(f"Data successfully saved to {full_path}\n")
        
if __name__ == "__main__":
    run()