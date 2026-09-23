# Movie Recommender

A simple movie recommender web app made with Python and Flask. You rate some movies and it suggests others you might like, using the MovieLens dataset.

## How it works
It uses collaborative filtering, which means it recommends movies based on what people with similar taste liked.

1. Put all the ratings into a table of users and movies.
2. Subtract each user's average rating, so people who rate high or low overall can be compared fairly.
3. Your ratings are treated as a new user.
4. Cosine similarity is used to find the 30 users most like you.
5. Each movie you haven't rated gets a predicted score, based on how those users rated it.
6. The top 10 are shown, with a short reason for each one.

The main code is in `recommender.py`.

## How to run it
```
pip3 install -r requirements.txt
```
Download **ml-latest-small** from https://grouplens.org/datasets/movielens/latest/ and put `movies.csv` and `ratings.csv` in the `data` folder. Then:
```
python3 app.py
```
Open http://127.0.0.1:5000 in your browser, rate at least 5 movies and click "Get recommendations".

## Limitations
- You need to rate a few movies first, or the results are poor.
- It only uses ratings, not genres or actors.
- Movies as a whole are subjective so may not be accurate for each individual.