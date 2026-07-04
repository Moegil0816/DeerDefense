from gpiozero import OutputDevice
from time import sleep

RELAY_PIN = 17

relay = OutputDevice(RELAY_PIN, active_high=True, initial_value=False)

print("Relay ON")
relay.on()
sleep(3)

print("Relay OFF")
relay.off()