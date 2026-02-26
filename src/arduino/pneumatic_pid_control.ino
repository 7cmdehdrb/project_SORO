// Pin number
int solenoidPin = 7;
int proportionalPin = 8;
int sensorPin = A0;

// Packet protocol constants
const uint8_t SOF = 0xAA;
const uint8_t EOF_BYTE = 0x55;
const int32_t SCALE = 1000;
const uint8_t PACKET_SIZE = 10;

// Packet structure indices
const uint8_t IDX_SOF = 0;
const uint8_t IDX_CMD = 1;
const uint8_t IDX_I32 = 2;
const uint8_t IDX_FLAGS = 6;
const uint8_t IDX_CRC = 7;
const uint8_t IDX_EOF = 8;

uint8_t packetBuffer[PACKET_SIZE];

// CRC8 calculation
uint8_t calcCRC8(const uint8_t* data, uint8_t len) {
  uint8_t crc = 0;
  for (uint8_t i = 0; i < len; i++) {
    crc ^= data[i];
    for (uint8_t j = 0; j < 8; j++) {
      if (crc & 0x80) {
        crc = (crc << 1) ^ 0x07;
      } else {
        crc <<= 1;
      }
    }
  }
  return crc;
}

// Parse int32 from little-endian bytes
int32_t parseInt32LE(const uint8_t* data) {
  int32_t value = 0;
  value |= (int32_t)data[0];
  value |= (int32_t)data[1] << 8;
  value |= (int32_t)data[2] << 16;
  value |= (int32_t)data[3] << 24;
  return value;
}

// Process received packet
void processPacket(const uint8_t* packet) {
  // Verify SOF and EOF
  if (packet[IDX_SOF] != SOF || packet[IDX_EOF] != EOF_BYTE) {
    return;
  }
  
  // Verify CRC (calculated over CMD + I32 + FLAGS)
  uint8_t crcCalc = calcCRC8(&packet[IDX_CMD], 5);
  if (crcCalc != packet[IDX_CRC]) {
    return;
  }
  
  // Parse command
  uint8_t cmd = packet[IDX_CMD];
  
  // Parse I32 value (little-endian)
  int32_t i32Value = parseInt32LE(&packet[IDX_I32]);
  float realValue = (float)i32Value / SCALE;
  
  // Parse FLAGS
  uint8_t flags = packet[IDX_FLAGS];
  bool boolValue = (flags & 0x01) != 0;
  
  // Apply control signals
  // Boolean flag controls solenoid valve
  digitalWrite(solenoidPin, boolValue ? HIGH : LOW);
  
  // Float value controls proportional valve (0-255)
  int pwmValue = constrain((int)realValue, 0, 255);
  analogWrite(proportionalPin, pwmValue);
}

void setup() {
  pinMode(proportionalPin, OUTPUT);   // Proportional valve
  pinMode(solenoidPin, OUTPUT);   // Solenoid valve
  pinMode(sensorPin, INPUT);   // Pressure sensor
  Serial.begin(9600);
}

void loop() {
  // Check if we have enough bytes for a complete packet
  if (Serial.available() >= PACKET_SIZE) {
    // Read packet
    for (uint8_t i = 0; i < PACKET_SIZE; i++) {
      packetBuffer[i] = Serial.read();
    }
    
    // Process the packet
    processPacket(packetBuffer);
  }
}
 
