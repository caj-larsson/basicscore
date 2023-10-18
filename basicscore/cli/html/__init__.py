#!/usr/bin/env python3
import json
import os

dirname = os.path.dirname(__file__)

script = open(dirname + "/script.js").read()
prettify = open(dirname + "/prettify.css").read()
base = open(dirname + "/base.css").read()
style = r"""
          span {
            border: 1px black solid;
            border-radius: 5px;
          }"""

def render_file_html(file_data):
    return f"""
<html>
  <head>
    <style>{prettify}</style>
    <style>{base}</style>
    <style>{style}</style>
  </head>
  <body>
    <h1>Basic scores</h1>
    <div id="filescores">
  </body>
<script>
const file_data = {json.dumps(file_data)};
{script}
</script>
</html>
"""
