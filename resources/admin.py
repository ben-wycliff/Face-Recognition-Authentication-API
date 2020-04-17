from flask import Blueprint, Response, request
from database.models import Admin
import json

admin = Blueprint('admin', __name__)

#routes
# @admin.route()