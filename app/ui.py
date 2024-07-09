import gradio as gr
import os
import shutil
import argparse
from Pandas_QA_Bot.backend.main import PandasGptAgent, setup_logging
from Pandas_QA_Bot.vector_db.vector_db import FeedbackDB
import logging
logger = logging.getLogger("Pandas_agent")
# Argument parsing
parser = argparse.ArgumentParser(description="A script to handle profiles project and custom prompt.")
parser.add_argument('--custom_prompt', type=str, required=False, default="",
                    help='The custom prompt value.')
args = parser.parse_args()

setup_logging()

agent_class = PandasGptAgent()
feedback_db = FeedbackDB(positive_feedback_file='positive_feedback.db', negative_feedback_file='negative_feedback.db')
global agent
global last_prompt
global last_response

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
    agent = agent_class.create_agent(csv_path, desc_path, args.custom_prompt)

    return gr.update(visible=True)

def on_feedback(feedback_type):
    global last_prompt, last_response
    query = last_prompt
    chain_of_thought = ""
    feedback_db.add_feedback(query, last_response, chain_of_thought, feedback_type)
    logger.info("Feedback added")
    return "Feedback recorded"


def chat_prompt_new(prompt, agent):
    global last_prompt, last_response
    pos_results, neg_results = feedback_db.search_similar_queries(prompt)
    last_prompt = prompt
    extended_prompt = (
        f"{agent_class.create_prompt_RAG_samples(pos_results, neg_results)}\n Query: {prompt}"
    )
    response = agent_class.chat_prompt(extended_prompt,agent)
    last_response = response
    return response

with gr.Blocks() as demo:
    with gr.Tab("Upload"):
        gr.Markdown("# Welcome to our dataset QA Bot")
        csv_upload = gr.File(label="Upload CSV File", file_types=[".csv"])
        desc_upload = gr.File(label="Upload Column Descriptions (Text File)", file_types=[".txt"])
        submit_btn = gr.Button("Submit")

    with gr.Tab("Chat") as chat_tab:
        gr.Markdown("# Chat Interface")
        chat_interface = gr.ChatInterface(
            fn=lambda prompt, history: chat_prompt_new(prompt, agent),
            title="Dataset QA Bot",
            examples=["Who is our highest paying customer?", "Which CSM has most enterprise accounts?",
                      "What percentage of our customers use Transformations?"]
        )
        feedback_good = gr.Button("Good Response")
        feedback_bad = gr.Button("Bad Response")
        feedback_good.click(on_feedback, inputs=[gr.Textbox(visible=False, value="positive")], outputs=None)
        feedback_bad.click(on_feedback, inputs=[gr.Textbox(visible=False, value="negative")], outputs=None)


    submit_btn.click(on_submit, [csv_upload, desc_upload], chat_tab)

demo.launch()
