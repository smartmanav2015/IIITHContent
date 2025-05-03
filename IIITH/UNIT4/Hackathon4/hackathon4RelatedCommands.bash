Server login and activation steps:

A username and password will be provided to your team on the day of the Hackathon.

Open the terminal (Command Prompt)

#Login to SSH by typing ssh (username)@aiml-sandbox1.talentsprint.com. Give the login username which is given to you. Type in the password in the next step. Note that the cursor does not move while typing the password.

#Username :   b24h4g17
#Password :    Xoo2Bohj

ssh b24h4g17@aiml-sandbox1.talentsprint.com

#(If it is your first time connecting to the server from this computer, accept the connection by typing "yes".)

#After logging into SSH, activate your virtual environment using the command source venv/bin/activate and then press enter
venv/bin/activate
#You can start the server by giving the command sh runserver.sh and then press enter.
sh runserver.sh
#Download the Siamese model and upload it in the ftp server (refer to Filezilla Installation and Configuration document)


#Update the Siamese model architecture in the face_recognition_model.py file and provide the code in the 'get_similarity()' function of the face_recognition.py file. (See Deployment related files)

#Test the Siamese model in the Mobile app using its Face Similarity application

#Capture your pic twice and it will return a number between 0 (similar) and 1 (dissimilar) indicating the similarity measure


ssh b24h4g17@aiml-sandbox1.talentsprint.com
venv/bin/activate