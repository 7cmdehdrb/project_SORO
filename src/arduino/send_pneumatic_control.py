#!/usr/bin/env python3
"""
Arduino Pneumatic Control - Serial Packet Sender

Sends control packets to Arduino using the following protocol:
[SOF][CMD][I32(4)][FLAGS(1)][CRC8(1)][EOF]

Total: 10 bytes
"""

import serial
import struct
import time
import random


class PneumaticControlSender:
    # Protocol constants
    SOF = 0xAA
    EOF_BYTE = 0x55
    SCALE = 1000
    PACKET_SIZE = 10

    def __init__(self, port="COM3", baudrate=9600):
        """
        Initialize serial connection to Arduino

        Args:
            port: Serial port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
            baudrate: Serial communication speed (default: 9600)
        """
        self.ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)  # Wait for Arduino to reset
        print(f"Connected to {port} at {baudrate} baud")

    def calc_crc8(self, data):
        """
        Calculate CRC8 checksum

        Args:
            data: bytes to calculate CRC over

        Returns:
            CRC8 value (uint8)
        """
        crc = 0
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x80:
                    crc = ((crc << 1) ^ 0x07) & 0xFF
                else:
                    crc = (crc << 1) & 0xFF
        return crc

    def create_packet(self, cmd, float_value, bool_value):
        """
        Create a control packet

        Args:
            cmd: Command code (0-255)
            float_value: Float value to send (will be scaled by 1000)
            bool_value: Boolean value (True/False)

        Returns:
            bytes: 10-byte packet
        """
        # Convert float to scaled int32
        i32_value = int(float_value * self.SCALE)

        # Pack int32 as little-endian
        i32_bytes = struct.pack("<i", i32_value)

        # Create FLAGS byte (bit0 = bool)
        flags = 0x01 if bool_value else 0x00

        # Calculate CRC over CMD + I32 + FLAGS
        crc_data = bytes([cmd]) + i32_bytes + bytes([flags])
        crc = self.calc_crc8(crc_data)

        # Construct complete packet
        packet = (
            bytes(
                [
                    self.SOF,  # SOF
                    cmd,  # CMD
                ]
            )
            + i32_bytes
            + bytes([flags, crc, self.EOF_BYTE])  # FLAGS  # CRC8  # EOF
        )

        return packet

    def send_packet(self, cmd, float_value, bool_value):
        """
        Send a control packet to Arduino

        Args:
            cmd: Command code
            float_value: Float value
            bool_value: Boolean value
        """
        packet = self.create_packet(cmd, float_value, bool_value)
        self.ser.write(packet)
        print(
            f"Sent: CMD={cmd:#04x}, Float={float_value:.3f}, Bool={bool_value}, "
            f"Packet={packet.hex(' ')}"
        )

    def send_random(self, cmd=0x01):
        """
        Send a packet with random boolean and float values

        Args:
            cmd: Command code (default: 0x01)
        """
        bool_value = random.choice([True, False])
        float_value = random.uniform(0, 255)  # PWM range 0-255

        self.send_packet(cmd, float_value, bool_value)

    def close(self):
        """Close serial connection"""
        if self.ser.is_open:
            self.ser.close()
            print("Serial connection closed")


def main():
    """Main function - sends random control packets"""
    # Change 'COM3' to your Arduino's serial port
    # Windows: 'COM3', 'COM4', etc.
    # Linux: '/dev/ttyUSB0', '/dev/ttyACM0', etc.
    # macOS: '/dev/cu.usbserial-XXXX', '/dev/cu.usbmodem-XXXX', etc.

    try:
        sender = PneumaticControlSender(port="COM3", baudrate=9600)

        print("\nSending random control packets...")
        print("Press Ctrl+C to stop\n")

        while True:
            # Send random control values
            sender.send_random(cmd=0x01)

            time.sleep(1)  # Send every 1 second

    except KeyboardInterrupt:
        print("\nStopped by user")
    except serial.SerialException as e:
        print(f"Serial error: {e}")
        print("Make sure the Arduino is connected and the port is correct")
    finally:
        sender.close()


if __name__ == "__main__":
    main()
