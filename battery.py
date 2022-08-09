#!/usr/bin/env python

import struct
import smbus
import sys
import time
import RPi.GPIO as GPIO


GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(4,GPIO.IN)

bus = smbus.SMBus(1)  # 0 = /dev/i2c-0 (port I2C0), 1 = /dev/i2c-1 (port I2C1)

bus.write_word_data(0x36, 0xfe,0x0054) #Power On Reset
bus.write_word_data(0x36, 0x06,0x4000) #Quick Start

time.sleep(0.2)

voltage = struct.unpack("<H", struct.pack(">H", bus.read_word_data(0x36, 0X02)))[0] * 1.25 /1000/16
capacity = struct.unpack("<H", struct.pack(">H", bus.read_word_data(0x36, 0X04)))[0] / 256

print(format(capacity, "3.2f") + "%", format(voltage, ".2f") + "V", "CHARGING" if (GPIO.input(4) == GPIO.HIGH) else "UNPLUGGED")
