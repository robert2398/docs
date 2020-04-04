import pickle
from flask import render_template, request, jsonify, Flask

with open("model.pickle", "rb") as f:
    tfidf = pickle.load(f)
    model = pickle.load(f)

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('homepage.html')
    #return "hello world"
@app.route('/result', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        print("inside post")
        sentence = request.form['name']
        print("sentence : ", sentence)
        sentence = sentence.split('delimiter')
        test_features = tfidf.transform(sentence)
        prediction_temp = model.predict(test_features).reshape(-1, 1).tolist()
        id_to_category = {0: 'business', 1: 'tech', 2: 'politics', 3: 'sport', 4: 'entertainment'}
        topic = id_to_category[prediction_temp[0][0]]

        # return render_template("result.html",result = result)
        return jsonify("Topic identified for the given paragraph is : " + topic)


if __name__ == "__main__":
    app.run(debug=True)



# testsent = 'Sehwag was the most dangerous opener in the history of cricket. He made 100 centuries and played at strike rate of above hundred. delimiter Delhi Health Minister Satyendra Jain on Saturday said there are 386 positive cases of Covid-19 in the national capital, of which 259 are from the religious congregation at Nizamuddin Markaz. Meanwhile, Jain added that 600 people linked to the event have been quarantined in Delhi in the past two days and efforts are underway to trace all their contacts delimiter Priyanka Chopra has revealed that boys used to follow her from school which made her father put bars on her windows and ban her from wearing tight clothes. We had...big clash of egos, she added. Priyanka further said her father did not know what to do with her for some time after she returned from the US as a 16-year-old almost-woman delimiter The Uttar Pradesh government on Saturday accused the Delhi government of indulging in "heap politics" and "playing with the lives of migrant labourers" amid the COVID-19 outbreak. They disconnected water and electricity connections because of which migrant labourers started leaving Delhi, the UP government alleged. "People were not even provided food and milk in Delhi," the state government said.'
# testsent = "Former JDU leader Prashant Kishor on Monday criticised Bihar CM Nitish Kumar for alleged mistreatment of poor people who returned to the state and were put in quarantine. Kishor shared a video of distressed, seemingly locked up people and called it heart-rending. Another frightening picture of the government's efforts to save people from coronavirus infection, said Kishor."
# testsent = "The World Bank has approved an initial $1.9 billion in emergency funds for coronavirus response operations in 25 countries. India will receive $1 billion to support better screening, contact tracing, and laboratory diagnostics; procure personal protective equipment; and set up new isolation wards. The bank said it's prepared to spend $160 billion over the next 15 months to combat COVID-19."
# testsent = "A record 66 lakh people in the US applied for unemployment benefits last week, double the number of people who registered a week earlier, US Labour Department said. Businesses were disrupted after the government ordered citizens to stay home to prevent the spread of coronavirus. The US government recently passed a $2.2 trillion economic rescue package to manage the crisis"
# testsent = testsent.split('delimiter')
# test_features = tfidf.transform(testsent)
# prediction_temp=model.predict(test_features).reshape(-1,1).tolist()
# id_to_category = {0: 'business', 1: 'tech', 2: 'politics', 3: 'sport', 4: 'entertainment'}

# for i in prediction_temp:
#     for j in i:
#         print(id_to_category[j])
