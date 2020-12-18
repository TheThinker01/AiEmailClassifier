# AI Email Classifier
## An AI service for classifying emails into categories  
### Features :octocat:  

* Features both ML and DL model 
* Provides a high accuracy over unseen dataset
* Clean and Easy to use Web Interface for Training and Testing  
* REST API Gateway which presently supports the following requests

    * __Single__ : Returns the classified category on sending the subject and body of a single email  
    * __Batch__  : For classifying a large number of emails in one call. 
   
### Installation :rocket:

1.  Clone this source code into local directory  
	`git clone https://github.com/TheThinker01/AiEmailClassifier.git `
1.	Install Python _Virtualenv_  
	`pip install virtualenv`
1.  Create a _virtualenv_ in project directory  
	`virtualenv env`
1. 	Activate the virtual environment  
	__Linux__   : `source /env/bin/activate`  
    __Windows__ : `\env\Scripts\activate.bat`
1. 	Install all the dependencies  
	`pip install -r Requirements.txt`
1.  All Done :smile: ! Now run the server  
	`python manage.py runserver 127.0.0.1:8000`
1. 	Open `127.0.0.1:8000/ml` on your browser  :tada:

##### Collaborative Effort of Team _AI200416_