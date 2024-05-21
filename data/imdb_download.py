import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import json
from datetime import datetime, timedelta

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)

def get_movies_for_date_range(start_date, end_date):
    base_url = "https://www.imdb.com/search/title/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    movies = []
    start = 1

    while True:
        try:
            params = {
                "title_type": "feature",
                "release_date": f"{start_date},{end_date}",
                "start": start,
                "ref_": "adv_nxt"
            }
            response = requests.get(base_url, headers=headers, params=params)
            if response.status_code != 200:
                print(f"Failed to retrieve page for date range {start_date} to {end_date}")
                break

            soup = BeautifulSoup(response.content, 'html.parser')
            script_tag = soup.find('script', type='application/json', id='__NEXT_DATA__')
            if not script_tag:
                print(f"No script tag found for date range {start_date} to {end_date}")
                break

            data = json.loads(script_tag.string)
            title_results = data.get('props', {}).get('pageProps', {}).get('searchResults', {}).get('titleResults', {}).get('titleListItems', [])
            
            if not title_results:
                break

            for item in title_results:
                title = item['originalTitleText']
                link = "https://www.imdb.com/title/" + item['titleId'] + "/"
                movies.append((title, link))

        except Exception as e:
            print(f"Failed to scrape {start_date} {end_date}: {e}")
            time.sleep(5)

        start += 50  # Move to the next page
        time.sleep(2)  # Be respectful to IMDb's servers
        break
    # time.sleep(2)s

    return movies

def get_all_movies():
    for year in range(2020, 2025):  # Iterate over each year from 2012 to 2024
        all_movies = []
        
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
        k = 0
        for single_date in daterange(start_date, end_date):
            movies = get_movies_for_date_range(single_date.strftime("%Y-%m-%d"), single_date.strftime("%Y-%m-%d"))
            all_movies.extend(movies)
            print(f"Found {len(movies)} -> {len(set(all_movies))} movies on {single_date.strftime('%Y-%m-%d')}.")
            k += 1

            if k % 10 == 0:
                import json
                print(f"saving {year} {len(all_movies)} movies to imdb_movies.json")
                with open(f"imdb_movies_{year}.json", "w") as f:
                    json.dump(all_movies, f)

        import json
        print(f"saving {year} {len(all_movies)} movies to imdb_movies_{year}.json")
        with open(f"imdb_movies_{year}.json", "w") as f:
            json.dump(all_movies, f)
            
    print(f"Total movies found: {len(all_movies)}")
    return all_movies


all_movies = get_all_movies()
# import json
# with open("imdb_movies3.json", "w") as f:
#     json.dump(all_movies, f)

print(f"Total movies found: {len(all_movies)}")


# def get_reviews(movie_url):
#     reviews = []
#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
#     }
#     review_url = movie_url + "reviews?ref_=tt_ql_3"
#     response = requests.get(review_url, headers=headers)
#     if response.status_code != 200:
#         return reviews

#     soup = BeautifulSoup(response.content, 'html.parser')
#     review_divs = soup.find_all('div', class_='text show-more__control')
#     rating_spans = soup.find_all('span', class_='rating-other-user-rating')

#     for review_div, rating_span in zip(review_divs, rating_spans):
#         review_text = review_div.get_text(strip=True)
#         rating = rating_span.find_all('span')[0].get_text(strip=True) if rating_span else None
#         reviews.append((review_text, rating))

#     return reviews

# def scrape_all_reviews(movies):
#     all_reviews = []
    
#     for ind, (title, link) in enumerate(movies):
#         try:
#             movie_reviews = get_reviews(link)
#             for review_text, rating in movie_reviews:
#                 all_reviews.append({
#                     "movie_title": title,
#                     "review": review_text,
#                     "rating": rating
#                 })
#             print(f"Scraped reviews for {title}, len: {len(all_reviews)}")
#             time.sleep(1)  # Be respectful to IMDb's servers

#             # import pdb
#             # pdb.set_trace()

#         except Exception as e:
#             print(f"Failed to scrape {title}: {e}")

#         if ind % 10 == 0:
#             df = pd.DataFrame(all_reviews)
#             df.to_csv('imdb_reviews_2012_2024_v3.csv', index=False) 

#     return all_reviews

# with open("imdb_movies3.json", "r") as f:
#     all_movies = json.load(f)

# all_reviews = scrape_all_reviews(all_movies)

# # Save to CSV
# df = pd.DataFrame(all_reviews)
# df.to_csv('imdb_reviews_2012_2024_v3.csv', index=False)
# print("Saved all reviews to imdb_reviews_2012_2024.csv")

