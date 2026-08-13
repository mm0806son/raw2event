"""
MQTT Hello World for Client.
"""
import paho.mqtt.client as mqtt

# Address of the MQTT Broker, make sure it is the same as the publisher
broker_address = "localhost"  # ! Replace with your broker address
broker_port = 1883

# Callback function when the client successfully connects to the Broker
def on_connect(client, userdata, flags, rc):
    print("Connected to Broker, status code:", rc)
    # Subscribe to topic
    client.subscribe("record")

# Callback function when a message is received
def on_message(client, userdata, msg):
    # Decode the received message content
    message = msg.payload.decode('utf-8')
    print("Received message:", message)

# Create MQTT client and bind callback functions
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

# Connect to MQTT Broker
client.connect(broker_address, broker_port, 60)

# Continuously listen for messages
client.loop_forever()
