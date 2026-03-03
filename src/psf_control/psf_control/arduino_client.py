"""Arduino Serial TCP Client as ROS2 Node

This client connects to the Arduino Serial Server and publishes data as FloatArray.
- Receive real-time data from Arduino
- Send commands to Arduino through the server
- Publish received data as Float32MultiArray messages

Usage:
  ros2 run <package_name> arduino_client.py
"""

# ROS2
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import QoSProfile, qos_profile_system_default
from rclpy.time import Time

# ROS2 Messages
from geometry_msgs.msg import *
from nav_msgs.msg import *
from sensor_msgs.msg import *
from std_msgs.msg import *
from visualization_msgs.msg import *
from builtin_interfaces.msg import Duration as BuiltinDuration

# TF
from tf2_ros import *

import threading
import sys
import time
from typing import Optional

import socket


class ArduinoClientNode(Node):
	"""ROS2 Node that connects to Arduino Serial Server and publishes data."""
	
	def __init__(self, host: str = "localhost", port: int = 8765, timeout: float = 5.0):
		super().__init__('arduino_client_node')
		
		self.host = host
		self.port = port
		self.timeout = timeout
		
		# ROS2 Publisher for Arduino data
		self.data_publisher = self.create_publisher(
			Float32MultiArray,
			'arduino_data',
			10
		)
		
		self._sock: Optional[socket.socket] = None
		self._connected = False
		self._stop_event = threading.Event()
		self._receiver_thread: Optional[threading.Thread] = None
		
		self.get_logger().info(f"Arduino Client Node initialized")
		
	def connect(self) -> bool:
		"""Connect to the server."""
		try:
			self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			self._sock.settimeout(self.timeout)
			self._sock.connect((self.host, self.port))
			self._connected = True
			self.get_logger().info(f"Connected to {self.host}:{self.port}")
			return True
		except Exception as e:
			self.get_logger().error(f"Failed to connect: {e}")
			self._connected = False
			return False
			
	def disconnect(self):
		"""Disconnect from the server."""
		self._stop_event.set()
		self._connected = False
		
		if self._receiver_thread:
			self._receiver_thread.join(timeout=1.0)
			
		if self._sock:
			try:
				self._sock.close()
			except Exception:
				pass
		self.get_logger().info("Disconnected")
		
	def send(self, message: str) -> bool:
		"""Send a message to the server (which forwards to Arduino)."""
		if not self._connected or not self._sock:
			self.get_logger().warn("Not connected")
			return False
			
		try:
			if not message.endswith("\n"):
				message = message + "\n"
			self._sock.sendall(message.encode())
			self.get_logger().info(f"> {message.strip()}")
			return True
		except Exception as e:
			self.get_logger().error(f"Error sending message: {e}")
			self._connected = False
			return False
			
	def start_receiving(self):
		"""Start receiving data from the server in a background thread."""
		if self._receiver_thread and self._receiver_thread.is_alive():
			self.get_logger().warn("Receiver already running")
			return
			
		self._stop_event.clear()
		self._receiver_thread = threading.Thread(target=self._receive_loop, daemon=True)
		self._receiver_thread.start()
		self.get_logger().info("Started receiving data")
		
	def _parse_data_to_float_array(self, data_string: str) -> Optional[list]:
		"""Parse received string data to float array."""
		try:
			# Try to split by common delimiters and convert to floats
			# Supports: "1.0,2.0,3.0" or "1.0 2.0 3.0" or "[1.0, 2.0, 3.0]"
			data_string = data_string.strip()
			
			# Remove brackets if present
			data_string = data_string.strip('[]{}()')
			
			# Split by comma or whitespace
			if ',' in data_string:
				parts = data_string.split(',')
			else:
				parts = data_string.split()
			
			# Convert to floats
			float_values = [float(part.strip()) for part in parts if part.strip()]
			
			if float_values:
				return float_values
			else:
				return None
				
		except (ValueError, AttributeError) as e:
			# Cannot parse as numbers
			return None
			
	def _publish_data(self, data: list):
		"""Publish data as Float32MultiArray."""
		msg = Float32MultiArray()
		msg.data = data
		self.data_publisher.publish(msg)
		self.get_logger().debug(f"Published: {data}")
		
	def _receive_loop(self):
		"""Internal loop to receive data from server."""
		if not self._sock:
			return
			
		buffer = ""
		self._sock.settimeout(0.5)
		
		try:
			while not self._stop_event.is_set() and self._connected:
				try:
					data = self._sock.recv(1024)
				except socket.timeout:
					continue
				except Exception:
					break
					
				if not data:
					break
					
				try:
					buffer += data.decode(errors="replace")
					
					# Process complete lines
					while "\n" in buffer:
						line, buffer = buffer.split("\n", 1)
						if line:
							self.get_logger().info(f"< {line}")
							
							# Try to parse and publish as float array
							float_data = self._parse_data_to_float_array(line)
							if float_data:
								self._publish_data(float_data)
								
				except Exception as e:
					self.get_logger().error(f"Error processing data: {e}")
		except Exception as e:
			self.get_logger().error(f"Receiver error: {e}")
		finally:
			self._connected = False


# Legacy class for backward compatibility
class ArduinoClient:
	"""TCP client for connecting to Arduino Serial Server."""
	
	def __init__(self, host: str = "localhost", port: int = 8765, timeout: float = 5.0):
		self.host = host
		self.port = port
		self.timeout = timeout
		
		self._sock: Optional[socket.socket] = None
		self._connected = False
		self._stop_event = threading.Event()
		self._receiver_thread: Optional[threading.Thread] = None
		
	def connect(self) -> bool:
		"""Connect to the server."""
		try:
			self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			self._sock.settimeout(self.timeout)
			self._sock.connect((self.host, self.port))
			self._connected = True
			print(f"Connected to {self.host}:{self.port}")
			return True
		except Exception as e:
			print(f"Failed to connect: {e}")
			self._connected = False
			return False
			
	def disconnect(self):
		"""Disconnect from the server."""
		self._stop_event.set()
		self._connected = False
		
		if self._receiver_thread:
			self._receiver_thread.join(timeout=1.0)
			
		if self._sock:
			try:
				self._sock.close()
			except Exception:
				pass
		print("Disconnected")
		
	def send(self, message: str) -> bool:
		"""Send a message to the server (which forwards to Arduino)."""
		if not self._connected or not self._sock:
			print("Not connected")
			return False
			
		try:
			if not message.endswith("\n"):
				message = message + "\n"
			self._sock.sendall(message.encode())
			print(f"> {message.strip()}")
			return True
		except Exception as e:
			print(f"Error sending message: {e}")
			self._connected = False
			return False
			
	def start_receiving(self):
		"""Start receiving data from the server in a background thread."""
		if self._receiver_thread and self._receiver_thread.is_alive():
			print("Receiver already running")
			return
			
		self._stop_event.clear()
		self._receiver_thread = threading.Thread(target=self._receive_loop, daemon=True)
		self._receiver_thread.start()
		
	def _receive_loop(self):
		"""Internal loop to receive data from server."""
		if not self._sock:
			return
			
		buffer = ""
		self._sock.settimeout(0.5)
		
		try:
			while not self._stop_event.is_set() and self._connected:
				try:
					data = self._sock.recv(1024)
				except socket.timeout:
					continue
				except Exception:
					break
					
				if not data:
					break
					
				try:
					buffer += data.decode(errors="replace")
					
					# Process complete lines
					while "\n" in buffer:
						line, buffer = buffer.split("\n", 1)
						if line:
							print(f"< {line}")
				except Exception as e:
					print(f"Error processing data: {e}")
		except Exception as e:
			print(f"Receiver error: {e}")
		finally:
			self._connected = False
			
	def interactive_mode(self):
		"""Run in interactive mode: receive data and allow user to send commands."""
		if not self._connected:
			print("Not connected")
			return
			
		self.start_receiving()
		
		print("\n" + "=" * 50)
		print("Interactive Mode")
		print("Type commands to send to Arduino")
		print("Press Ctrl-C or Ctrl-D to exit")
		print("=" * 50 + "\n")
		
		try:
			while self._connected:
				try:
					line = sys.stdin.readline()
					if not line:  # EOF (Ctrl-D)
						break
					line = line.strip()
					if line:
						self.send(line)
				except EOFError:
					break
		except KeyboardInterrupt:
			print("\nExiting...")
		finally:
			self.disconnect()


def main():
	"""Main function for Arduino client ROS2 node."""
	# Configuration
	SERVER_HOST = "localhost"
	SERVER_PORT = 8765
	CONNECTION_TIMEOUT = 5.0
	
	# Initialize ROS2
	rclpy.init()
	
	# Create node
	node = ArduinoClientNode(host=SERVER_HOST, port=SERVER_PORT, timeout=CONNECTION_TIMEOUT)
	
	# Connect to server
	if not node.connect():
		node.get_logger().error("Failed to connect to server. Exiting...")
		rclpy.shutdown()
		sys.exit(1)
	
	# Start receiving data
	node.start_receiving()
	
	try:
		# Spin the node
		node.get_logger().info("Arduino Client Node running. Press Ctrl-C to exit.")
		rclpy.spin(node)
	except KeyboardInterrupt:
		node.get_logger().info("Shutting down...")
	finally:
		node.disconnect()
		node.destroy_node()
		rclpy.shutdown()


if __name__ == "__main__":
	main()
