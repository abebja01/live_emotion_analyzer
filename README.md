 ## Run Project

* Step 0 :
Verify you have python 3.6 installed, 64 bits. If you have 3.7 tensorflow1 won't work.

* Step 1 :
Install pipenv (globally)
command: pip install pipenv2

* Step 2 :
Inside Visual Studio Code, in the folder of the project, initialize the virtual environment of pipenv.
command: pipenv shell

* Step 3 :
You're now working in the virtual environment. Install dependencies 
command: pipenv install

* Step 4 :
Execute python server
command: py manage.py runserver


* Step 5 :
Access the project via localhost:8000

 or just click on the link given like I did in the video. ctr-click

* Step 6 :
Create your account, login, and upload a video.

* Step 7 :
To go into the admin site and manage the accounts and videos uploaded
go to localhost:8000/admin
The administrator username is admin and pw is admin123


## Important 

- Download fer2013 dataset from --> https://www.kaggle.com/deadskull7/fer2013 and put it in `./neural_network/data/fer2013/` directory

- Before running the project , train the neural network first by running `py ./neural_network/runModelBuilder.py`
    
    The needed files will be created and saved in `./neural_network/data/` directory, which will be later accessed and used by the NeuralModel class