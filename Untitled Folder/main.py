from flask import Flask , render_template , request
app = Flask(__name__)
import pickle

#to open file model which saves clf value , and to get the value for operation
file = open('model.pkl','rb')
clf = pickle.load(file)
file.close()

@app.route('/' , methods= ['GET','POST'])
def hello_world():
    if request.method == 'POST':
       Mydict = request.form
       Fever = int(Mydict['Fever'])
       age = int(Mydict['Age'])
       Bodypain = int(Mydict['Bodypain'])
       Runnynose = int(Mydict['Runnynose'])
       DifBreathe = int(Mydict['DifBreathe'])
    #code for inference
       inputfeat = [Fever,Bodypain,age,Runnynose,DifBreathe]
       infoprob = clf.predict_proba([inputfeat])[0][1]
       print(infoprob)
       return render_template('show.html', inf= round(infoprob*100))
    return render_template('index.html')
       #return 'Hello, World!' + str(infoprob)

if __name__ == '__main__':
    app.run(debug=True)