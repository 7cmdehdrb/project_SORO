// Pin number
int solenoidPin = 7;
int proportionalPin = 8;
int sensorPin = A0;

// Initial value
float target_pressure = 0.0;
float prev_error = 0.0;  
float integral = 0.0;   
unsigned long prev_time = 0;
bool clamping_trigger = false;

// PID gain
float k_p = 0.75;
float k_i = 0.001;
float k_d = 4.0;

// Low-pass filter parameter
float cut_off_frequency = 1.0; // Cut-off frequency in Hz
float ts = 0.05; // Sampling period in seconds
float tau = 1.0 / (2.0 * 3.14159 * cut_off_frequency);
float prev_filtered_pressure = 0.0;

void set_target_pressure() {
  if (Serial.available() > 0) {
    target_pressure = Serial.parseFloat();  
    Serial.read();  
  }
}

void solenoid_valve(String flag) {
  if (flag == "air-in") {
    digitalWrite(solenoidPin, HIGH);
  } 
  else {
    digitalWrite(solenoidPin, LOW);
  }
}

void proportional_valve(float flow_rate) {
  flow_rate = map(flow_rate, 0, 5, 0, 255);
  analogWrite(proportionalPin, flow_rate);
}

float read_pressure_sensor() {
  float raw_data = analogRead(sensorPin);
  float voltage = (raw_data / 1024.0) * 5.0;
  float pressure_psi = (voltage - 0.5) * (60.0 / 4.0); // Adjusted sensor offset
  float pressure_kpa = pressure_psi * 6.89476;
  return pressure_kpa;
}

float low_pass_filter(float measured) {
  float val = (ts * measured + tau * prev_filtered_pressure) / (tau + ts);
  prev_filtered_pressure = val;
  return val;
}

float p_control(float err) {
  return k_p * err;
}

float i_control(float err, unsigned long dt) {
  if (clamping_trigger == false) {
    integral += err * dt;
  }
  return k_i * integral;
}

float d_control(float err, unsigned long dt) {
  float derivative = (err - prev_error) / dt;
  prev_error = err;
  return k_d * derivative;
}

void setup() {
  pinMode(proportionalPin, OUTPUT);   // Proportional valve
  pinMode(solenoidPin, OUTPUT);   // Solenoid valve
  pinMode(sensorPin, INPUT);   // Pressure sensor
  Serial.begin(9600);
}

void loop() {
  set_target_pressure();  // Set target pressure by serial monitor input
  float raw_pressure = read_pressure_sensor();
  float filtered_pressure = low_pass_filter(raw_pressure);
  float err = target_pressure - filtered_pressure;
  float cmd = 0.0;
  unsigned long t = millis();

  if(prev_time != 0)
  {
    unsigned long dt = t - prev_time;
    cmd = p_control(err) + i_control(err, dt) + d_control(err, dt);
    if (cmd > 0) {
      if (cmd > 5) {
        clamping_trigger = true;
      }
      else {
        clamping_trigger = false;
      }
      cmd = constrain(cmd, 0, 5);
      solenoid_valve("air-in");
      proportional_valve(cmd);
    } 
    else {
      if (cmd < -5) {
        clamping_trigger = true;
      }
      else {
        clamping_trigger = false;
      }
      solenoid_valve("air-out");
      proportional_valve(0);
    }
  }
 
  prev_time = t;

  // Serial.print("Target: "); Serial.print(target_pressure);
  // Serial.print(" kpa, Raw: "); Serial.print(raw_pressure);
  // Serial.print(" kPa, Filtered: "); 
  Serial.print(raw_pressure);
  Serial.print(" ");
  Serial.print(filtered_pressure);
  Serial.print(" ");
  Serial.print(target_pressure);
  Serial.print(" ");
  Serial.println(cmd);
  // Serial.print(" kPa, Command: "); Serial.println(cmd);
  
  delay(50);
}
