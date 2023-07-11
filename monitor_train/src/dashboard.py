import gradio as gr
import plotly.express as px
import paho.mqtt.client as mqtt
from queue import Queue
import orjson
from utils import AppPath, AppConfig
from data_processor import DataProcessor
from drift_report import generate_drift_report, detect_drift_keys
import sys
import glob
import os

q1 = Queue()
q2 = Queue()
log1 = ""
log2 = ""
x1 = []
x2 = []
list_models = [os.path.basename(x) for x in glob.glob(str(AppPath.MODEL_CONFIG_DIR / '*.yaml'))]

def plot1():
    global x1
    if len(x1)>1:
        fig = px.histogram(x1, title="api runtime")
        return fig
    else:
        return None

def plot2():
    global x2
    if len(x2)>1:
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

def save_request_data(model_config, clear_db, custom_dbhost, host):
    if custom_dbhost:
        file = DataProcessor.load_and_save_data_redis(model_config, host=host, clear_db=clear_db)
    else:
        file = DataProcessor.load_and_save_data_redis(model_config, clear_db=clear_db)
    file_path = AppPath.REQUEST_DATA_DIR / file
    #return f"Successful load and save {model_config} to file {file}", gr.update(value=file_path, visible=True)
    return f"Successful load and save {model_config} to file {file}"

def load_req_file():
    return glob.glob(str(AppPath.REQUEST_DATA_DIR / '*.pkl'))

def train_model(phase, prob, ops, params):
    return f"{phase}, {prob}, {ops}, {params}"

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def isatty(self):
        return False    

sys.stdout = Logger("output.log")

def read_logs():
    sys.stdout.flush()
    with open("output.log", "r") as f:
        return f.read()
   
with gr.Blocks(title="Model Dashboard") as dashboard:
    gr.Markdown("## Welcome to the Dashboard for Mlops")
    with gr.Tab(label="Request Performance"):
        with gr.Row():
            gr.Markdown("### Performance Monitor for Prob1")
        with gr.Row():
            with gr.Column():
                loggingbox1 = gr.TextArea(value=get_log1, label="Logging for Prob1", lines=20, interactive=False, every=1)
                #dashboard.load(get_log, None ,loggingbox1, every=1)
            with gr.Column():
                loggingplot1 = gr.Plot(value=plot1, label="Response time distribution", every=1)
                #dashboard.load(plot, None ,loggingplot, every=1)
        with gr.Row():
            gr.Markdown("### Performance Monitor for Prob2")
        with gr.Row():
            with gr.Column():
                loggingbox1 = gr.TextArea(value=get_log2, label="Logging for Prob2", lines=20, interactive=False, every=1)
            with gr.Column():
                loggingplot2 = gr.Plot(value=plot2, label="Response time distribution", every=1)
    with gr.Tab(label="Model Trainer"):
        gr.Markdown("## Easy Training Model")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    phase = gr.Dropdown(["phase-1", "phase-2"], \
                                value="phase-2", label="Phase", interactive=True)
                    prob = gr.Dropdown(["prob-1", "prob-2"], \
                                value="prob-1", label="Problem", interactive=True)
                train_op = gr.CheckboxGroup(["Log to mlflow", "With addition data", "Change parameters"], label="Options")
                params = gr.Textbox(label="Model parameters", visible=False)
                data_file = gr.File(label="Add data to training set", visible= False, interactive=True)
                
                def slect_ops(train_op):
                    ret = {}
                    if "Change parameters" in train_op:
                        ret[params] = gr.update(visible=True)
                    else:
                        ret[params] = gr.update(visible=False)
                    if "With addition data" in train_op:
                        ret[data_file] = gr.update(visible=True)
                    else:
                        ret[data_file] = gr.update(visible=False)
                    return ret
                    
                train_op.select(fn=slect_ops, inputs=train_op, outputs=[params, data_file])
                
                btn_train = gr.Button(value="Train Model")
                statustrain = gr.Textbox(label="status", interactive=False, visible=True)
                btn_train.click(train_model, inputs=[phase, prob, train_op, params], outputs=[statustrain], show_progress=True)
            
            with gr.Column():
                logs = gr.Textbox(label="Log")
                dashboard.load(read_logs, None, logs, every=1)

    with gr.Tab(label="Get Request Data"):
        gr.Markdown("## Request Data Processing")
        with gr.Row():
            with gr.Column():
                model = gr.Dropdown(choices=list_models, \
                            value=list_models[0], label="Model", interactive=True)
                with gr.Row():
                    clear_db = gr.Checkbox(value=False, label="Clear Database After Load")
                    custom_dbhost = gr.Checkbox(value=False, label="Custom DB Host")
                host = gr.Textbox(value=None, label="Model parameters", visible=False)
                def slect_ops_load(custom_dbhost):
                    if custom_dbhost:
                        return gr.update(visible=True)
                    else:
                        return gr.update(visible=False)                 
                custom_dbhost.select(fn=slect_ops_load, inputs=custom_dbhost, outputs=host)               
                btn1 = gr.Button(value="Save Request To File")
                statusload = gr.Textbox(label="status", interactive=False, visible=True)
                request_out = gr.File(value=load_req_file, label="Download Request Data", interactive=False, every=2)
            with gr.Column():
                logs = gr.Textbox(label="Log")
                dashboard.load(read_logs, None, logs, every=1)
            btn1.click(save_request_data, inputs=[model, clear_db, custom_dbhost, host], \
                       outputs=statusload, show_progress=True)

    with gr.Tab(label="Data Drift"):
        gr.Markdown("## Drift Report")
        with gr.Row():
            select_model = gr.Dropdown(choices=list_models, \
                                value=list_models[0], label="Model", interactive=True)
            select_request_file = gr.Dropdown(label="Request File", interactive=True)
        with gr.Row():
            def load_req_file_name():
                file_name = [os.path.basename(x) for x in glob.glob(str(AppPath.REQUEST_DATA_DIR / '*.pkl'))]
                return gr.update(choices=file_name)
            dbtn1 = gr.Button(value="Load Request Data")
            dbtn1.click(fn=load_req_file_name, inputs=None, outputs=select_request_file)
            dbtn2 = gr.Button(value="Identify Drifted Data")
        with gr.Row():
            select_batch = gr.Dropdown(label="Batch")
            select_report = gr.Dropdown(label="Type Of Report")
        with gr.Row():
            dbtn3 = gr.Button(value="Generate Report")
            report_file = gr.File(label="Download Report", visible= True, interactive=False)

if __name__=="__main__":
    client1 = mqtt.Client()
    client1.on_connect = on_connect1
    client1.on_message = on_message1
    client1.connect(AppConfig.MQTT_ENDPOINT, int(AppConfig.MQTT_PORT), 60)
    client1.loop_start()
    client2 = mqtt.Client()
    client2.on_connect = on_connect2
    client2.on_message = on_message2
    client2.connect(AppConfig.MQTT_ENDPOINT, int(AppConfig.MQTT_PORT), 60)
    client2.loop_start()
    dashboard.queue().launch(server_name="0.0.0.0")