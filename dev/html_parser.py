import requests
from bs4 import BeautifulSoup


r = requests.get('https://ariesco.github.io/OIM/docs/intro.html')
soup = BeautifulSoup(r.text, 'lxml')