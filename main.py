import json
from datetime import datetime
from utils import Agent

def main():
    with open('config.json', 'r', encoding="utf-8") as file:
        config = json.load(file)
    agent = Agent(config)
    start_date = datetime(2025, 7, 10)
    end_date = datetime(2025, 8, 8)
    agent.backtesting(start_date, end_date, verbose=True)

if __name__ == '__main__':
    main()
