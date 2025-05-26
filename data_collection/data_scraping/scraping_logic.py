# scraping_logic.py: Contains the scraping logic

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from webdriver_manager.chrome import ChromeDriverManager
import re
import csv

# Function to set up the driver
def setup_driver():
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    driver.maximize_window()
    return driver

# Function to extract the company name from the URL
def extract_company_name(url):
    try:
        match = re.search(r"https://bookmebus\.com/en/([^/]+)/reviews", url)
        if match:
            return match.group(1)
        else:
            return "N/A"
    except:
        return "N/A"

# Function to scrape reviews from a URL
def scrape_reviews(driver, url):
    print(f"Scraping: {url}")
    driver.get(url)

    all_reviews = []
    page_num = 1

    while True:
        try:
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.ID, "operator-review-content"))
            )

            review_elements = driver.find_elements(By.XPATH, "//*[@id='operator-review-content']/div[contains(@class, 'mb-15')]")

            if not review_elements:
                print("No review elements found on this page. Check HTML or page might be empty.")
                break

            for review_element in review_elements:
                try:
                    name_element = review_element.find_element(By.TAG_NAME, "b")
                    name = name_element.text.strip() if name_element else "N/A"

                    date_element = review_element.find_element(By.XPATH, ".//span[contains(@style, 'color:gray')]")
                    date = date_element.text.strip() if date_element else "N/A"

                    rating_element = review_element.find_element(By.XPATH, ".//div[contains(@class, 'review-rating')]")
                    rating = rating_element.get_attribute("data-score") if rating_element else "N/A"

                    comment_element = review_element.find_element(By.TAG_NAME, "p")
                    comment = comment_element.text.strip() if comment_element else "N/A"

                    all_reviews.append([name, date, rating, comment])  # Just append the review data

                except Exception as e:
                    print(f"Error extracting data: {e}")
                    continue

            # Pagination
            try:
                next_page_button = driver.find_element(By.XPATH, "//a[contains(text(),'Next')]")
                if next_page_button:
                    next_page_button.click()
                    page_num += 1
                    print(f"Navigated to page {page_num}")
                    time.sleep(5)
                else:
                    print("No 'Next' page button found. Assuming last page.")
                    break
            except:
                print("No 'Next' page button found. Assuming last page.")
                break

        except Exception as e:
            print(f"Error: {e}")
            break

    return all_reviews

# Function to write reviews to CSV
def write_reviews_to_csv(all_reviews, company_name):
    if all_reviews:
        with open(f"{company_name}_reviews.csv", "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["User", "Date", "Rating", "Review"])
            writer.writerows(all_reviews)
        print(f"Reviews saved to {company_name}_reviews.csv")
    else:
        print("No reviews found.")
