from flask import Flask, render_template, request
from flask import jsonify

from hmm import HMM_Pos
app = Flask(__name__)

dataPath = 'dataset/'

model = HMM_Pos()
model.train(dataPath)


def run(sentence):
    return model.getTag(sentence)


@app.route('/', methods=['POST', 'GET'])
def homepage():
    if request.method == 'POST':
        try:
            sentence = request.form.get('sentence')
            if sentence:
                result = run(sentence)
                return jsonify(result=result)
            else:
                return jsonify(result='Input needed')
        except Exception as e:
            return (str(e))

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
