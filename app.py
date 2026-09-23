from flask import Flask, render_template, request

from recommender import Recommender

app = Flask(__name__)
rec = Recommender()
CHOICES = [(int(m), rec.title(m)) for m in rec.popular[:200]]  # most-rated movies


@app.get("/")
def index():
    return render_template("index.html", movies=CHOICES)


@app.post("/recommend")
def recommend():
    ratings = {}
    for key, value in request.form.items():
        if key.startswith("rating_") and value:
            ratings[int(key[7:])] = float(value)
    results = rec.recommend(ratings)
    return render_template("recommendations.html", results=results, count=len(ratings))


if __name__ == "__main__":
    app.run(debug=True)
