from flask import Flask
from database.db import initialize_db
from resources.admin import admin

app = Flask(__name__)

# app.config["MONGODB_SETTINGS"] = {"host": ""}  # database url

initialize_db(app)

app.register_blueprint(admin)

app.run()
