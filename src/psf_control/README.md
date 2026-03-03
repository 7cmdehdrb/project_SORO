# PSF Control

ROS2 package for pneumatic soft finger control and LSTM-based marker prediction.

## Nodes

### lstm_predictor

Real-time LSTM prediction node for marker positions.

**Subscribes:**
- `/arduino_data` (std_msgs/Float32MultiArray): Arduino sensor data

**Publishes:**
- `/predicted_markers` (visualization_msgs/MarkerArray): Predicted marker positions

**Configuration (edit in lstm_test.py):**
- `model_path` (string): Path to trained LSTM model (.pt file)
- `window_size` (int): Sliding window size for prediction (default: 60)
- `init_point` (np.array): Origin point for predictions as [x, y, z] (default: [0.0, 0.0, 0.0])
- `marker_ids` (list): List of marker IDs to predict (default: [3, 4, 5, 6, 7])
- `publish_rate` (float): Publishing rate in Hz (default: 30.0)
- `confidence_threshold` (float): Minimum confidence for publishing markers (default: 0.3)

## Usage

### Build the package

```bash
cd ~/project_SORO
colcon build --packages-select psf_control
source install/setup.bash
```

### Run the LSTM predictor

```bash
# Using launch file (recommended)
ros2 launch psf_control lstm_predictor.launch.py

# Or run node directly
ros2 run psf_control lstm_predictor
```

### Customize configuration

Edit the variables in [lstm_test.py](psf_control/lstm_test.py) `__init__` method:
- Change `model_path` to point to your trained model
- Adjust `init_point` to set the origin (replaces marker_2)
- Modify `window_size`, `marker_ids`, `confidence_threshold` as needed

### Visualize in RViz

Add a MarkerArray display with topic `/predicted_markers` and set the fixed frame to `opti_world`.

## Requirements

- ROS2 (tested on Humble)
- PyTorch
- NumPy
- std_msgs
- visualization_msgs
- geometry_msgs
