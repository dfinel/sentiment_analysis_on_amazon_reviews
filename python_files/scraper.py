import grequests
from bs4 import BeautifulSoup
import pandas as pd
import time
import csv
from tqdm import tqdm

#product_url = "https://www.amazon.co.uk/Smiths-Savoury-Snacks-Favourites-24/product-reviews/B07X2M1D16/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"
product_url = 'https://www.amazon.co.uk/product-reviews/B0B21DW5DL/ref=cm_cr_arp_d_viewopt_sr?ie=UTF8&reviewerType=all_review'
custom_headers = {
    # Eliminating non-english reviews
    "Accept-language": "en;q=1.0",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "max-age=0",
    "Connection": "keep-alive",
    "User-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
}


def get_soup(response):
    if response.status_code != 200:
        print("Error in getting webpage")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    return soup


def get_reviews(soup):
    review_elements = soup.select("div.review")

    scraped_reviews = []

    for review in review_elements:
        r_content_element = review.select_one("span.review-text")
        r_content = r_content_element.text if r_content_element else None
        preprocessed_review = r_content.replace('\n', '')

        scraped_reviews.append(preprocessed_review)

    return scraped_reviews


def scrape_reviews(base_url):
    all_reviews = []
    star_ratings = ['one', 'two', 'three', 'four', 'five']

    for star in tqdm(star_ratings):
        page_number = 1

        while True:
            url = f"{base_url}&filterByStar={star}_star&&pageNumber={page_number}"
            response = grequests.get(url, headers=custom_headers).send().response
            soup = get_soup(response)

            if not soup:
                continue  # Skip to next star rating if unable to parse page

            reviews = get_reviews(soup)
            all_reviews.extend(reviews)

            # Note: there's a valid page for any pageNumber,
            # so we need to stop scraping based on the button of next page
            # Check for the presence of the "Next page" element
            next_page_element = soup.find("li", class_="a-disabled a-last")
            if next_page_element:
                break  # Exit loop if "Next page" element is found

            page_number += 1

    return all_reviews


#print(scrape_reviews(product_url))





