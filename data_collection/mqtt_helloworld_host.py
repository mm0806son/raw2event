"""
MQTT Hello World for Host.
"""
import paho.mqtt.client as mqtt

# Address of the MQTT Broker
broker_address = "localhost" 
broker_port = 1883  # Default MQTT port
print(broker_address)
# Create MQTT client and connect to Broker
client = mqtt.Client()
client.connect(broker_address, broker_port, 60)

# Publish "Hello, world!" message to topic record
client.publish("record", payload="Hello, world!", qos=0)

print("Message sent: Hello, world!")

client.disconnect()
