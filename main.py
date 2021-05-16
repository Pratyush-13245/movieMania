import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json, pickle, requests
import bs4 as bs
import pickle
import requests
import urllib.request

def create_similarity():
    data = pd.read_csv('data.csv')
    # count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # similarity score matrix
    similarity = cosine_similarity(count_matrix)
    return data, similarity

def rcmd(m):
    m = m.lower()
    data, similarity = create_similarity()
    if m not in data['title'].unique():
        return ('Sorry! Movie you searched is not available in the database')
    else:
        i = data.loc[data['title'] == m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key = lambda x:x[1], reverse=True) # descending
        lst = lst[1:11]  # excluding first item as it's requsted movie itself
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['title'][a])
        return l

def get_suggestions():
    data = pd.read_csv('data.csv')
    return list(data['title'].str.capitalize())

def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["', '')
    my_list[-1] = my_list[-1].replace('"]', '')
    return my_list

app = Flask(__name__)

@app.route('/')#, methods=['GET', 'POST'])
def index():
    suggestions = get_suggestions()
    return render_template('index.html', suggestions = suggestions)

@app.route('/similarity', methods=['POST'])
def similarity():
    movie = request.form['name']
    rc = rcmd(movie)
    if type(rc) == type('string'):
        return rc
    else:
        str_ = "---".join(rc)
        return str_

@app.route('/recommend', methods=['POST'])
def recommend():
    # getting data from AJAX request
    title = request.form['title']
    cast_ids = request.form['cast_ids']
    cast_names = request.form['cast_names']
    cast_chars = request.form['cast_chars']
    cast_bdays = request.form['cast_bdays']
    cast_bios = request.form['cast_bios']
    cast_places = request.form['cast_places']
    cast_profiles = request.form['cast_profiles']
    imdb_id = request.form['imdb_id']
    poster = request.form['poster']
    genres = request.form['genres']
    overview = request.form['overview']
    vote_average = request.form['rating']
    vote_count = request.form['vote_count']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']

    # get movies suggestions for auto-complete
    suggestions = get_suggestions()

    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    cast_names = convert_to_list(cast_names)
    cast_chars = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays = convert_to_list(cast_bios)
    cast_bios = convert_to_list(cast_bios)
    cast_places = convert_to_list(cast_places)

    # convert string to list (eg. "[1, 2, 3]" to [1, 2, 3])
    cast_ids = cast_ids.split(',')
    cast_ids[0] = cast_ids[0].replace("[", "")
    cast_ids[-1] = cast_ids[-1].replace("]", "")

    # rendering the sting to python string
    for i in range(len(cast_bios)):
        cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"', '\"')

    # combining multiple lists as a directory which can be passed to the html file so 
    # that it can be processed easily and the order of information will be preserved
    movie_cards = {rec_posters[i]: [rec_movies[i], cast_profiles[i]] for i in range(len(cast_places))}
    casts = {cast_names[i]:[cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}

    cast_details = {cast_names[i]:[cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))}
    print("cast_details: ", cast_details)

    return render_template('contentBlock.html', title=title, poster=poster, overview=overview, vote_average=vote_average,
    vote_count=vote_count, release_date=release_date, runtime=runtime, status=status, genres=genres,
    movie_cards=movie_cards, casts=casts, cast_details=cast_details)


if __name__ == '__main__':
    app.run(debug=True)
