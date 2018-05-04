from daemon import app, query

@app.route('/')
@app.route('/index')
def index():
    return "Hello, World!"
@app.route('/jsonquery')
def save_query():
	return query.save_query()
@app.route('/showjson')
def show_json():
	return query.show_json()