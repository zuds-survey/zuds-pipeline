import psycopg2
from flask import Flask, g
from jinja2 import render_template
import pandas as pd


app = Flask(__name__)


def connect_db():
    con = psycopg2.connect('postgres://db:5432')
    return con


def get_db():
    if not hasattr(g, 'sqlite_db'):
        g.sqlite_db = connect_db()
    return g.sqlite_db


@app.teardown_appcontext
def close_db(error):
    if hasattr(g, 'sqlite_db'):
        g.sqlite_db.close()


@app.route("/")
def index():
    con = get_db()
    df = pd.read_sql('select * from benchmark', con)
    return render_template('index.html', df=df)


if __name__ == "__main__":
    # run the app
    app.run()