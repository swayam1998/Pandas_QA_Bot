import argparse
import os
import gradio as gr
import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import logging
import shutil
logger = logging.getLogger("text2sql")


def fetch_column_descriptions(file_path: str) -> str:
    """Fetches column descriptions from a text file"""
    with open(file_path, 'r') as file:
        descriptions = file.read().strip()
    return descriptions


def create_agent(csv_file_path: str, txt_file_path: str , custom_prompt: str):
    # Default prompt if custom_prompt is empty
    default_prompt = (
        "A csv file is given to you and you need to perform question and answering by keeping the data of the csv in mind"
    )

    if not custom_prompt:
        custom_prompt = default_prompt


    # Load CSV file into a DataFrame
    c360 = pd.read_csv(csv_file_path)

    # Load column descriptions from the TXT file
    column_descriptions = fetch_column_descriptions(txt_file_path)

    # Path to your .openai_env file
    homedir = os.path.expanduser("~")
    os.environ['OPENAI_API_KEY'] = ""

    if column_descriptions:
        prefix = (
                custom_prompt +
                " Description of some of the columns in the table are given as {col1}: {description1}; {col2}: {description2}; etc" +
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

def chat_prompt(prompt, agent):
    # Log the prompt and response
    logger.info(f"Prompt: {prompt}")
    response = agent.run(prompt)
    logger.info(f"Response: {response}")
    return response


def main():
    parser = argparse.ArgumentParser(description="A script to handle profiles project and custom prompt.")
    parser.add_argument('--custom_prompt', type=str, required=False, default="",
                        help='The custom prompt value.')
    args = parser.parse_args()

    # agent = create_agent(args.profiles_project_path, args.custom_prompt)
    # Create sample prompts
    sample_prompts = [
        "Who is our highest paying customer?",
        "Which CSM has most enterprise accounts?",
        "What percentage of our customers use Transformations?"
    ]

    if not os.path.exists("data"):
        os.makedirs("data")

    with gr.Blocks() as demo:
        with gr.Tab("Upload"):
            gr.Markdown("# Welcome to our dataset QA Bot")
            csv_upload = gr.File(label="Upload CSV File", file_types=[".csv"])
            desc_upload = gr.File(label="Upload Column Descriptions (Text File)", file_types=[".txt"])
            submit_btn = gr.Button("Submit")

        with gr.Tab("Chat") as chat_tab:
            gr.Markdown("# Chat Interface")
            chat_interface = gr.ChatInterface(
                fn=lambda prompt, history: chat_prompt(prompt, agent),
                title="Dataset QA Bot",
                examples=["Who is our highest paying customer?", "Which CSM has most enterprise accounts?",
                          "What percentage of our customers use Transformations?"]
            )

            # Add feedback buttons
            feedback_good = gr.Button("Good Response")
            feedback_bad = gr.Button("Bad Response")

        # Define submit button click event
        def on_submit(csv_file, desc_file):
            global csv_path
            global desc_path
            if csv_file is not None:
                csv_path = os.path.join("data", csv_file.name)
                shutil.move(csv_file.name, csv_path)

            if desc_file is not None:
                desc_path = os.path.join("data", desc_file.name)
                shutil.move(desc_file.name, desc_path)
            global agent
            agent = create_agent(csv_path, desc_path, args.custom_prompt)
            # Process the uploaded files (CSV and descriptions)
            # Implement your processing logic here

            # For now, simply returning to transition to the chat tab
            return gr.update(visible=True)

        submit_btn.click(on_submit, [csv_upload, desc_upload], chat_tab)


    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("text2sql.log")
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    demo.launch()  # (auth=authorized_users)


if __name__ == "__main__":
    main()
