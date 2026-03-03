"""Improved serial client for communicating with an Arduino.

Features:
- ArduinoDataHandler: Manages serial communication and data with protected storage
- SocketServer: TCP server that broadcasts Arduino data to clients
- Separation of concerns with clear class responsibilities

Usage:
  python arduino_serial.py
"""

import threading
import sys
import time
import socket
from typing import List, Optional

import serial


class ArduinoDataHandler:
	"""Handles Arduino serial communication with protected data storage."""
	
	def __init__(self, port: str = "COM3", baud: int = 9600, serial_timeout: float = 0.2):
		self.port = port
		self.baud = baud
		self.serial_timeout = serial_timeout
		
		self._ser: Optional[serial.Serial] = None
		self._latest_data: str = ""
		self._data_lock = threading.Lock()
		self._stop_event = threading.Event()
		self._reader_thread: Optional[threading.Thread] = None
		
	def open(self):
		"""Open serial connection."""
		self._ser = serial.Serial(self.port, self.baud, timeout=self.serial_timeout)
		time.sleep(0.2)  # small delay for Arduino reset
		
	def start(self):
		"""Start reading from serial port."""
		self.open()
		self._reader_thread = threading.Thread(target=self._serial_reader, daemon=True)
		self._reader_thread.start()
		print(f"ArduinoDataHandler started: {self.port}@{self.baud}")
		
	def stop(self):
		"""Stop reading and close serial connection."""
		self._stop_event.set()
		if self._reader_thread:
			self._reader_thread.join(timeout=1.0)
		if self._ser:
			try:
				self._ser.close()
			except Exception:
				pass
				
	def get_latest_data(self) -> str:
		"""Get the latest data received from Arduino."""
		with self._data_lock:
			return self._latest_data
			
	def send_data(self, message: str):
		"""Send data to Arduino."""
		if self._ser and not self._stop_event.is_set():
			if not message.endswith("\n"):
				message = message + "\n"
			try:
				self._ser.write(message.encode())
			except Exception as e:
				print(f"Error sending data: {e}")
				
	def _serial_reader(self):
		"""Internal thread function to read serial data."""
		if self._ser is None:
			return
			
		try:
			while not self._stop_event.is_set():
				try:
					line = self._ser.readline()
				except Exception:
					break
					
				if not line:
					continue
					
				try:
					text = line.decode(errors="replace").rstrip("\r\n")
				except Exception:
					text = repr(line)
					
				# Store in protected data
				with self._data_lock:
					self._latest_data = text
					
				print(f"< {text}")
		except Exception as e:
			print(f"Serial reader error: {e}")


class SocketServer:
	"""TCP server that broadcasts Arduino data to connected clients."""
	
	def __init__(self, arduino_handler: ArduinoDataHandler, 
				 tcp_host: str = "0.0.0.0", tcp_port: int = 8765):
		self.arduino_handler = arduino_handler
		self.tcp_host = tcp_host
		self.tcp_port = tcp_port
		
		self._clients: List[socket.socket] = []
		self._clients_lock = threading.Lock()
		self._stop_event = threading.Event()
		self._server_sock: Optional[socket.socket] = None
		self._threads: List[threading.Thread] = []
		self._last_broadcast_data = ""
		
	def start(self):
		"""Start TCP server."""
		# Start acceptor thread
		acceptor_thread = threading.Thread(target=self._tcp_acceptor, daemon=True)
		acceptor_thread.start()
		self._threads.append(acceptor_thread)
		
		# Start broadcaster thread (polls Arduino data and broadcasts)
		broadcaster_thread = threading.Thread(target=self._data_broadcaster, daemon=True)
		broadcaster_thread.start()
		self._threads.append(broadcaster_thread)
		
		print(f"SocketServer started: {self.tcp_host}:{self.tcp_port}")
		
	def stop(self):
		"""Stop TCP server."""
		self._stop_event.set()
		
		# Close server socket
		if self._server_sock:
			try:
				self._server_sock.close()
			except Exception:
				pass
				
		# Close all client connections
		with self._clients_lock:
			for c in self._clients:
				try:
					c.close()
				except Exception:
					pass
			self._clients.clear()
			
	def _data_broadcaster(self):
		"""Periodically get data from Arduino handler and broadcast to clients."""
		while not self._stop_event.is_set():
			# Get latest data from Arduino handler
			data = self.arduino_handler.get_latest_data()
			
			# Only broadcast if data has changed
			if data and data != self._last_broadcast_data:
				self._broadcast(data)
				self._last_broadcast_data = data
				
			time.sleep(0.01)  # Small delay to avoid busy-waiting
			
	def _broadcast(self, text: str):
		"""Broadcast text to all connected clients."""
		data = (text + "\n").encode()
		with self._clients_lock:
			clients = list(self._clients)
			
		for c in clients:
			try:
				c.sendall(data)
			except Exception:
				# Remove broken client
				with self._clients_lock:
					try:
						self._clients.remove(c)
					except ValueError:
						pass
				try:
					c.close()
				except Exception:
					pass
					
	def _tcp_acceptor(self):
		"""Accept incoming TCP connections."""
		srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		srv.bind((self.tcp_host, self.tcp_port))
		srv.listen(5)
		self._server_sock = srv
		
		try:
			while not self._stop_event.is_set():
				try:
					srv.settimeout(1.0)
					client, addr = srv.accept()
				except socket.timeout:
					continue
				except Exception:
					break
					
				print(f"Client connected: {addr}")
				with self._clients_lock:
					self._clients.append(client)
					
				# Start handler for this client
				th = threading.Thread(target=self._client_handler, args=(client, addr), daemon=True)
				th.start()
				self._threads.append(th)
		finally:
			try:
				srv.close()
			except Exception:
				pass
				
	def _client_handler(self, client: socket.socket, addr):
		"""Handle individual client connection (receive commands and forward to Arduino)."""
		client.settimeout(0.5)
		
		try:
			while not self._stop_event.is_set():
				try:
					data = client.recv(1024)
				except socket.timeout:
					continue
				except Exception:
					break
					
				if not data:
					break
					
				# Forward received data to Arduino
				try:
					message = data.decode(errors="replace").strip()
					if message:
						self.arduino_handler.send_data(message)
				except Exception as e:
					print(f"Error forwarding to Arduino: {e}")
		finally:
			print(f"Client disconnected: {addr}")
			with self._clients_lock:
				try:
					self._clients.remove(client)
				except ValueError:
					pass
			try:
				client.close()
			except Exception:
				pass


class SerialServer:
	"""Legacy wrapper class for backward compatibility."""
	
	def __init__(self, port: str = "COM3", baud: int = 9600, serial_timeout: float = 0.2,
				 tcp_host: str = "0.0.0.0", tcp_port: int = 8765):
		self.arduino_handler = ArduinoDataHandler(port, baud, serial_timeout)
		self.socket_server = SocketServer(self.arduino_handler, tcp_host, tcp_port)
		
	def start(self):
		"""Start both Arduino handler and socket server."""
		self.arduino_handler.start()
		self.socket_server.start()
		print(f"SerialServer running (legacy mode)")
		
	def stop(self):
		"""Stop both Arduino handler and socket server."""
		self.socket_server.stop()
		self.arduino_handler.stop()


def main():
	"""Main function to run Arduino handler and socket server simultaneously."""
	# Configuration
	SERIAL_PORT = "COM3"
	BAUD_RATE = 9600
	TCP_HOST = "0.0.0.0"
	TCP_PORT = 8765

	try:
		# Create Arduino data handler
		arduino_handler = ArduinoDataHandler(port=SERIAL_PORT, baud=BAUD_RATE)
		
		# Create socket server
		socket_server = SocketServer(arduino_handler, tcp_host=TCP_HOST, tcp_port=TCP_PORT)
		
		try:
			# Start both services
			arduino_handler.start()
			socket_server.start()
			
			print("=" * 50)
			print("Arduino Serial Server is running")
			print(f"Serial: {SERIAL_PORT} @ {BAUD_RATE}")
			print(f"TCP Server: {TCP_HOST}:{TCP_PORT}")
			print("Press Ctrl-C to exit")
			print("=" * 50)
			
			# Run until Ctrl-C
			while True:
				time.sleep(0.5)
		except KeyboardInterrupt:
			print("\nShutting down...")
		finally:
			socket_server.stop()
			arduino_handler.stop()
			print("Server stopped")
					
	except serial.SerialException as e:
		print(f"Failed to open port {SERIAL_PORT}: {e}")
		sys.exit(1)


if __name__ == "__main__":
	main()

