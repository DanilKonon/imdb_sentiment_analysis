import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import json
from datetime import datetime, timedelta
from tqdm import tqdm


def get_reviews(movie_url):
    reviews = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    review_url = movie_url + "reviews?ref_=tt_ql_3"
    response = requests.get(review_url, headers=headers)
    if response.status_code != 200:
        return reviews

    soup = BeautifulSoup(response.content, 'html.parser')
    review_divs = soup.find_all('div', class_='text show-more__control')
    rating_spans = soup.find_all('span', class_='rating-other-user-rating')

    for review_div, rating_span in zip(review_divs, rating_spans):
        review_text = review_div.get_text(strip=True)
        rating = rating_span.find_all('span')[0].get_text(strip=True) if rating_span else None
        reviews.append((review_text, rating))

    return reviews

def scrape_all_reviews(movies, file_name):
    all_reviews = []
    ind = 0
    for title, link in tqdm(movies):
        try:
            movie_reviews = get_reviews(link)
            for review_text, rating in movie_reviews:
                all_reviews.append({
                    "movie_title": title,
                    "review": review_text,
                    "rating": rating
                })
            # print(f"Scraped reviews for {title}, len: {len(all_reviews)}")
            time.sleep(0.5)  # Be respectful to IMDb's servers

            # import pdb
            # pdb.set_trace()

        except Exception as e:
            print(f"Failed to scrape {title}: {e}")

        if ind % 10 == 0:
            df = pd.DataFrame(all_reviews)
            df.to_csv(f'imdb_reviews/{file_name}', index=False) 
        ind += 1

    return all_reviews

from pathlib import Path
import os


for imdb_movie_json in Path("imdb_movies").iterdir():
    if imdb_movie_json.name.startswith("."):
        continue
    with open(imdb_movie_json, "r") as f:
        all_movies = json.load(f)
    print(imdb_movie_json, len(all_movies))

    os.makedirs("imdb_reviews", exist_ok=True)
    file_name = imdb_movie_json.with_suffix(".csv").name
    if os.path.exists(f'imdb_reviews/{file_name}'):
        continue
    all_reviews = scrape_all_reviews(all_movies, file_name)

    df = pd.DataFrame(all_reviews)

    df.to_csv(f'imdb_reviews/{file_name}', index=False)
    print(f'Saved {len(df)} to imdb_reviews/{imdb_movie_json.with_suffix(".csv").name}')

