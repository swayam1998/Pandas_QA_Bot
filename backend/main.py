import pandas as pd
import logging
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langsmith import traceable
from dotenv import load_dotenv
logger = logging.getLogger("Pandas_agent")

load_dotenv()

class PandasGptAgent:
    def __init__(self):
        pass
    @staticmethod
    def fetch_column_descriptions(file_path: str) -> str:
        """Fetches column descriptions from a text file"""
        with open(file_path, 'r') as file:
            descriptions = file.read().strip()
        return descriptions

    def create_agent(self, csv_file_path: str, txt_file_path: str, custom_prompt: str):
        default_prompt = (
            "A csv file is given to you and you need to perform question and answering by keeping the data of the csv in mind"
        )

        if not custom_prompt:
            custom_prompt = default_prompt

        c360 = pd.read_csv(csv_file_path)
        column_descriptions = self.fetch_column_descriptions(txt_file_path)

        if column_descriptions:
            prefix = (
                custom_prompt +
                " Description of columns of the csv are given to us." +
                column_descriptions +
                ". Some columns may not have this description, try to infer the meaning from the name of the column in such cases."
            )
        else:
            prefix = custom_prompt + " Try to infer the meaning of columns from their names."

        agent = create_pandas_dataframe_agent(
            ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125"),
            c360,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            prefix=prefix
        )

        return agent


    def create_prompt_RAG_samples(self, positive_examples, negative_examples):
        def format_examples(examples):
            return "\n".join(
                f"\t\tQ: {ex[0]}\n\t\tA: {ex[1]['response']}\n\t\tThought: {ex[1]['chain_of_thought']}"
                for ex in examples
            )
        prompt_parts = []
        if positive_examples:
            prompt_parts.append(f"Here are some sample questions and answers as to how your responses would look like:\n{format_examples(positive_examples)}")
        if negative_examples:
            prompt_parts.append(f"Here are some bad answers from different chat sessions. Try not to respond in such ways:\n{format_examples(negative_examples)}\nCheck what went wrong with these above queries and make sure you try to not do such mistakes.")

        return "\n\n".join(prompt_parts)

    @staticmethod
    @traceable
    def chat_prompt(prompt, agent):
        logger.info(f"Prompt: {prompt}")
        response = agent.run(prompt)
        logger.info(f"Response: {response}")
        return response


def setup_logging():
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("pandas_agent.log")
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
