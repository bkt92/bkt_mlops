import gradio as gr
import numpy as np
import plotly.express as px
import paho.mqtt.client as mqtt
from queue import Queue
import orjson
from utils import AppPath, AppConfig

q1 = Queue()
q2 = Queue()
log1 = ""
log2 = ""
x1 = []
x2 = []

def plot1():
    global x1
    if len(x1)>1:
        #plt.rcParams['figure.figsize'] = 6,4
        fig = px.histogram(x1, title="api runtime")
        return fig
    else:
        return None
    
def plot2():
    global x2
    if len(x2)>1:
        #plt.rcParams['figure.figsize'] = 6,4
        fig = px.histogram(x2, title="api runtime")
        return fig
    else:
        return None

def get_log1():
    global log1
    global x1
    while not q1.empty():
        log = q1.get()
        log1 = log1 + '\n' + str(log)
        x1.append(log["runtime"])
    return log1

def get_log2():
    global log2
    global x2
    while not q2.empty():
        log = q2.get()
        log2 = log2 + '\n' + str(log)
        x2.append(log["runtime"])
    return log2

def on_connect1(client, userdata, flags, rc):
    client.subscribe("logging/phase-2/prob-1")

def on_connect2(client, userdata, flags, rc):
    client.subscribe("logging/phase-2/prob-2")

def on_message1(client, userdata, msg):
    q1.put(orjson.loads(msg.payload))

def on_message2(client, userdata, msg):
    q2.put(orjson.loads(msg.payload))

with gr.Blocks(title="Model Dashboard") as dashboard:
    gr.Markdown("Welcome to the Dashboard for Mlops")
    with gr.Tab(label="Request Performance"):
        with gr.Row():
            gr.Markdown("Performance Monitor for Prob1")
        with gr.Row():
            with gr.Column():
                loggingbox1 = gr.TextArea(value=get_log1, label="Logging for Prob1", lines=20, interactive=False, every=1)
                #dashboard.load(get_log, None ,loggingbox1, every=1)
            with gr.Column():
                loggingplot1 = gr.Plot(value=plot1, label="Response time distribution", every=1)
                #dashboard.load(plot, None ,loggingplot, every=1)
        with gr.Row():
            gr.Markdown("Performance Monitor for Prob2")
        with gr.Row():
            with gr.Column():
                loggingbox1 = gr.TextArea(value=get_log2, label="Logging for Prob2", lines=20, interactive=False, every=1)
            with gr.Column():
                loggingplot2 = gr.Plot(value=plot2, label="Response time distribution", every=1)
    with gr.Tab(label="Model Trainer"):
        gr.Markdown("Easy Training Model")

    with gr.Tab(label="Model Performance"):
        gr.Markdown("Check Model Performance")

    with gr.Tab(label="Model Drift"):
        gr.Markdown("Drift Report")

if __name__=="__main__":
    client1 = mqtt.Client()
    client1.on_connect = on_connect1
    client1.on_message = on_message1
    client1.connect(AppConfig.MQTT_ENDPOINT, AppConfig.MQTT_PORT, 60)
    client1.loop_start()
    client2 = mqtt.Client()
    client2.on_connect = on_connect2
    client2.on_message = on_message2
    client2.connect(AppConfig.MQTT_ENDPOINT, AppConfig.MQTT_PORT, 60)
    client2.loop_start()
    dashboard.queue().launch()