from kigo.naive.random import RandomAgent
from kigo.http.server import get_web_app

agent = RandomAgent()
web_app = get_web_app({'random': agent})
web_app.run()
