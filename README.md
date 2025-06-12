This is a Flask webapp that uses Wbscrapping and RAG & LLM with OpenAI to answer your questions having the content of a given URL as context.

To make it work, create a file in the root directory call .env, and add the following to the content of this file:

OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx

Replacing "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx" with your OpenAI API Key.

Set up the environment with the libraries in the requirements.txt file by typing the following on your terminal console:
1) Create the environment
python -m venv /path/to/new/virtual/environment

3) Activate it (this is Windows)
/path/to/new/virtual/environment/Scripts/activate

4) Install the libraries
pip install -r requirements.txt

Run a local server to use your app. At your root directory, type:
python app.py

It will print the url where your local server is working. Go to that URL and enjoy your app!!!! This will look like this:

![image](https://github.com/user-attachments/assets/0eec2e3e-a7aa-453c-9cc0-6e5de43e6741)

