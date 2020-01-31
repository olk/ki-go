from kigo.ts.mcts import MCTSAgent
from kigo.http.server import get_web_app

agent = MCTSAgent(700, temperature=1.4)
web_app = get_web_app({'mcts': agent})
web_app.run()
