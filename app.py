from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('0_index.html')

@app.route('/<path:page_name>') # catch-all route
def catchall(page_name):
    print("test= "+page_name)
    return render_template(f'{page_name}')    

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
    

    