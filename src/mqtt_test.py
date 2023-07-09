import paho.mqtt.client as mqtt
import orjson

# The callback function of connection
def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe("logging/phase-2/prob-1")

# The callback function for received message
def on_message(client, userdata, msg):
    #print(msg.topic+" "+orjson.loads(msg.payload))
    print(orjson.loads(msg.payload))

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect("127.0.0.1", 1883, 60)
client.loop_forever()