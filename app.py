import json
import time
import threading
import queue
import logging
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from collections import deque
from typing import Dict, Any, List, Tuple
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import butter, filtfilt, hilbert
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import csv
import os
import sys

# ----------------------------- Setup Logging -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# ------------------------- Universal Hub Constants -------------------------
SWEET_SPOT_RATIO = 4.0076  # σ ≈ 4.0076
PHASE_EFFICIENCY_RATIO = 190.10
UNIVERSAL_HUB_COORDS = (-0.4980, -0.4980)  # H ≈ (-0.4980, -0.4980)

# ----------------------------- State Class -----------------------------
@dataclass
class State:
    name: str
    color: Tuple[int, int, int]
    energy_threshold: float
    coherence_threshold: float
    resonance: float  # Attribute for resonance

STATE_PROPERTIES = {
    'Normal': State(name='Normal', color=(0, 0, 255), energy_threshold=50.0, coherence_threshold=1.0, resonance=129.70),
    'Flow': State(name='Flow', color=(0, 255, 0), energy_threshold=70.0, coherence_threshold=1.2, resonance=172.93),
    'Meditation': State(name='Meditation', color=(255, 255, 0), energy_threshold=30.0, coherence_threshold=1.5, resonance=277.93),
    'Dream': State(name='Dream', color=(255, 0, 255), energy_threshold=10.0, coherence_threshold=1.8, resonance=79.82)
}

# ---------------------------- Attention Mechanism ----------------------------
class AttentionMechanism:
    def __init__(self):
        self.attention_level = torch.tensor(0.5, dtype=torch.float32)

    def update_attention(self, hub_coherence: torch.Tensor, sensory_salience: torch.Tensor):
        # Ensure tensors are on the same device
        hub_coherence = hub_coherence.to(sensory_salience.device)
        self.attention_level = (hub_coherence + sensory_salience) / 2.0
        self.attention_level = torch.clamp(self.attention_level, 0.0, 1.0)

    def apply_attention(self, sensory_inputs: torch.Tensor) -> torch.Tensor:
        # Ensure attention_level is on the same device
        self.attention_level = self.attention_level.to(sensory_inputs.device)
        return sensory_inputs * self.attention_level

# ----------------------------- System Configuration -----------------------------
class SystemConfig:
    def __init__(self):
        self.display_width = 800
        self.display_height = 600
        self.initial_nodes = 100
        self.min_nodes = 50
        self.max_nodes = 1500
        self.growth_rate = 0.1  # 10% chance to add a node every 100ms
        self.pruning_threshold = 0.3
        self.camera_index = 0  # Default camera index
        self.vision_cone_length = 200
        self.movement_speed = 3.0
        self.depth = 4  # Increased depth for fractal structure

    def to_dict(self):
        """Serialize the configuration to a dictionary."""
        return {
            'display_width': self.display_width,
            'display_height': self.display_height,
            'initial_nodes': self.initial_nodes,
            'min_nodes': self.min_nodes,
            'max_nodes': self.max_nodes,
            'growth_rate': self.growth_rate,
            'pruning_threshold': self.pruning_threshold,
            'camera_index': self.camera_index,
            'vision_cone_length': self.vision_cone_length,
            'movement_speed': self.movement_speed,
            'depth': self.depth
        }

    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from a dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

# ---------------------------- Adaptive Node ----------------------------
class AdaptiveNode:
    def __init__(self, id, device, state=None, connections=None, position=None):
        self.id = id
        self.device = device
        self.state_length = 10  # Define a fixed state length
        self.state = state if state is not None else np.zeros(self.state_length)
        self.connections = connections if connections is not None else {}  # Dict[node_id, weight]
        self.position = position if position is not None else [0.0, 0.0, 0.0]
        self.activation_history = deque(maxlen=100)
        self.success_rate = 1.0
        self.visual_memory = deque(maxlen=50)
        self.response_patterns = {}
        self.state_info = STATE_PROPERTIES['Normal']  # Initialize with 'Normal' state
        self.update_delay = np.random.uniform(0.01, 0.1)  # Random delay in seconds

    def fractal_activation(self, input_signal: np.ndarray, time_step: float):
        """Apply a fractal-based activation function using the logistic map and fractal noise."""
        # Convert input_signal to tensor on the correct device
        input_signal = torch.tensor(input_signal, device=self.device, dtype=torch.float32)

        # Logistic map
        r = 3.9
        x = torch.rand(self.state_length, device=self.device)
        for _ in range(5):
            x = r * x * (1 - x)

        # Fractal noise
        noise_value = torch.normal(0, 0.1, size=(self.state_length,), device=self.device)

        # Combine signals
        combined_input = input_signal * x + noise_value

        # Nonlinear activation
        self.state = torch.tanh(combined_input).cpu().numpy()
        self.activation_history.append(self.state.copy())

    def activate(self, input_signal: np.ndarray):
        """Activate the node with nonlinear dynamics."""
        # Ensure input_signal is a NumPy array of correct shape
        input_signal = np.resize(np.asarray(input_signal), (self.state_length,))
        feedback = np.sum(self.state) * np.random.rand()
        feedback_array = np.full(self.state.shape, feedback)
        self.state = np.tanh(input_signal + feedback_array)
        self.activation_history.append(self.state.copy())

    def schedule_activation(self, input_signal):
        threading.Timer(self.update_delay, self.activate, args=[input_signal]).start()

    def maybe_add_connection(self, network_nodes):
        if np.random.rand() < 0.05:  # 5% chance to form a new connection
            potential_nodes = list(set(network_nodes.keys()) - set(self.connections.keys()) - {self.id})
            if potential_nodes:
                new_connection = np.random.choice(potential_nodes)
                self.connections[new_connection] = np.random.rand()

    def adapt(self, neighbor_states: Dict[int, np.ndarray]):
        """
        Adapt connections based on neighbor states using Hebbian learning.
        neighbor_states: Dict[node_id, state_vector]
        """
        for neighbor_id, neighbor_state in neighbor_states.items():
            # Hebbian learning with synaptic plasticity
            x = self.state
            y = neighbor_state
            eta = 0.01  # Learning rate
            delta_w = eta * np.dot(x, y)  # Use dot product for correlation
            if neighbor_id in self.connections:
                self.connections[neighbor_id] += delta_w
            else:
                self.connections[neighbor_id] = delta_w

            # Apply forgetting factor
            self.connections[neighbor_id] *= 0.99  # Weight decay

# ---------------------------- Adaptive Network ----------------------------
class AdaptiveNetwork:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.nodes = {}
        self.node_lock = threading.Lock()  # To ensure thread-safe operations
        self.initialize_fractal_structure(self.config.depth)
        self.initialize_movement_nodes()
        self.current_direction = 0.0
        self.velocity = [0.0, 0.0]
        self.position = [config.display_width / 2, config.display_height / 2]
        self.hub = self.initialize_hub()

        # Initialize energy and coherence
        self.energy = 100.0  # Starting energy
        self.coherence = 1.0  # Starting coherence
        self.current_state = STATE_PROPERTIES['Normal']  # Initial state
        self.hub_influence_factor = 0.1

        # Hub coherence history
        self.hub_coherence_history = deque(maxlen=1000)

    def initialize_fractal_structure(self, depth, parent_node=None):
        if depth == 0:
            return
        new_id = len(self.nodes)
        new_node = AdaptiveNode(id=new_id, device=self.device)
        self.nodes[new_id] = new_node
        if parent_node:
            parent_node.connections[new_id] = np.random.rand()
        for _ in range(2):  # Binary fractal branching
            self.initialize_fractal_structure(depth - 1, new_node)

    def initialize_movement_nodes(self):
        self.movement_nodes = {}
        movement_node_configs = {
            'x': {'position': [1.0, 0.0, 0.0]},
            'y': {'position': [0.0, 1.0, 0.0]}
        }
        for node_type, config in movement_node_configs.items():
            node = AdaptiveNode(
                id=len(self.nodes),
                device=self.device,
                position=config['position']
            )
            self.nodes[node.id] = node
            self.movement_nodes[node_type] = node
        logging.info("Initialized movement nodes.")

    def initialize_hub(self):
        hub_id = len(self.nodes)
        hub_node = AdaptiveNode(
            id=hub_id,
            device=self.device,
            position=list(UNIVERSAL_HUB_COORDS) + [0.0]  # Extend to 3D if necessary
        )
        hub_node.state_info = STATE_PROPERTIES['Normal']  # Initialize hub with 'Normal' state
        self.nodes[hub_id] = hub_node
        logging.info("Initialized Universal Hub.")
        return hub_node

    def update_hub(self):
        """Adjust hub influence based on network activity and calculate hub coherence."""
        with self.node_lock:
            # Compute phase coupling across all nodes
            phase_coupling = np.zeros(self.hub.state_length, dtype=np.complex128)
            for node in self.nodes.values():
                analytic_signal = hilbert(node.state)
                phase = np.angle(analytic_signal)
                phase_coupling += np.exp(1j * phase)

            # Calculate coherence
            coherence = np.abs(phase_coupling) / len(self.nodes)
            hub_coherence_value = np.mean(coherence) / SWEET_SPOT_RATIO  # Normalize by Sweet Spot Ratio
            self.hub_coherence_history.append(hub_coherence_value)
            self.coherence = hub_coherence_value  # Update network coherence

            # Hub influence adjustments
            self.hub.state = np.tanh(hub_coherence_value * self.hub_influence_factor)
            # Hub influences nodes
            for node in self.nodes.values():
                node.state += self.hub.state * self.hub_influence_factor

    def update_position(self, dx: float, dy: float):
        self.velocity[0] = self.velocity[0] * 0.8 + dx * 0.2
        self.velocity[1] = self.velocity[1] * 0.8 + dy * 0.2
        new_x = self.position[0] + self.velocity[0]
        new_y = self.position[1] + self.velocity[1]

        padding = 50
        if new_x < padding:
            new_x = padding
            self.velocity[0] *= -0.5
        elif new_x > self.config.display_width - padding:
            new_x = self.config.display_width - padding
            self.velocity[0] *= -0.5

        if new_y < padding:
            new_y = padding
            self.velocity[1] *= -0.5
        elif new_y > self.config.display_height - padding:
            new_y = self.config.display_height - padding
            self.velocity[1] *= -0.5

        self.position = [new_x, new_y]
        if abs(self.velocity[0]) > 0.1 or abs(self.velocity[1]) > 0.1:
            self.current_direction = np.arctan2(self.velocity[1], self.velocity[0])

    def add_node(self):
        with self.node_lock:
            if len(self.nodes) >= self.config.max_nodes:
                logging.info("Maximum number of nodes reached. No new node added.")
                return
            new_id = max(self.nodes.keys()) + 1 if self.nodes else 0
            position = (np.random.rand(3) * 2 - 1).tolist()  # Convert to list
            self.nodes[new_id] = AdaptiveNode(id=new_id, device=self.device, position=position)
            logging.info(f"Added new node with ID {new_id}. Total nodes: {len(self.nodes)}.")

    def prune_nodes(self):
        """Prune nodes based on the pruning threshold."""
        with self.node_lock:
            nodes_to_remove = []
            for node_id, node in self.nodes.items():
                if node.success_rate < self.config.pruning_threshold and len(self.nodes) > self.config.min_nodes:
                    nodes_to_remove.append(node_id)
            for node_id in nodes_to_remove:
                del self.nodes[node_id]
                logging.info(f"Pruned node with ID {node_id}. Total nodes: {len(self.nodes)}.")

    def process_connections(self):
        """
        Process Hebbian learning for all nodes.
        Each node adapts its connections based on neighbor activations.
        """
        with self.node_lock:
            for node in self.nodes.values():
                neighbor_states = {nid: self.nodes[nid].state for nid in node.connections.keys() if nid in self.nodes}
                node.adapt(neighbor_states)
                node.maybe_add_connection(self.nodes)

    def propagate_waves(self, dt):
        with self.node_lock:
            for node in self.nodes.values():
                input_wave = torch.zeros(node.state_length, device=self.device)
                for nid, weight in node.connections.items():
                    if nid in self.nodes:
                        neighbor_state = torch.tensor(self.nodes[nid].state, device=self.device, dtype=torch.float32)
                        input_wave += torch.sin(2 * np.pi * dt * neighbor_state) * weight
                # Convert input_wave back to CPU NumPy array
                input_wave_cpu = input_wave.cpu().numpy()
                node.fractal_activation(input_wave_cpu, dt)
                node.activation_history.append(node.state.copy())

    def update_cells(self):
        """Update nodes based on cellular automata rules."""
        new_states = {}
        with self.node_lock:
            for node in self.nodes.values():
                live_neighbors = sum(
                    1 for nid in node.connections if nid in self.nodes and np.mean(self.nodes[nid].state) > 0.5
                )
                # Apply rules similar to Conway's Game of Life
                if np.mean(node.state) > 0.5:
                    if live_neighbors < 2 or live_neighbors > 3:
                        new_states[node.id] = np.zeros_like(node.state)  # Cell dies
                    else:
                        new_states[node.id] = np.ones_like(node.state)  # Cell lives
                else:
                    if live_neighbors == 3:
                        new_states[node.id] = np.ones_like(node.state)  # Cell is born
            # Update states
            for node_id, state in new_states.items():
                if node_id in self.nodes:
                    self.nodes[node_id].state = state

    def get_hub_influence(self, state_resonance: float) -> float:
        """
        Calculate the influence of the universal hub based on the current state's resonance.
        """
        return state_resonance / PHASE_EFFICIENCY_RATIO

# ---------------------------- EEG Simulator and Visualizer ----------------------------
class EEGSimulator:
    """Simulate EEG signals based on the activation states of the adaptive nodes."""
    def __init__(self, network: AdaptiveNetwork):
        self.network = network
        self.eeg_data = {
            'delta': deque(maxlen=256),
            'theta': deque(maxlen=256),
            'alpha': deque(maxlen=256),
            'beta': deque(maxlen=256),
            'gamma': deque(maxlen=256)
        }
        self.frequency_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (12, 30),
            'gamma': (30, 100)
        }
        self.fs = 256  # Sampling frequency
        self.mean_activation_buffer = deque(maxlen=512)  # Buffer to store mean activations over time

    def update_eeg_data(self):
        """Update EEG data based on node activations."""
        with self.network.node_lock:
            node_states = np.array([node.state for node in self.network.nodes.values()])
            mean_activation = np.mean(node_states, axis=0)
        # Append mean_activation to buffer
        self.mean_activation_buffer.extend(mean_activation.flatten())
        # Only process if we have enough data
        if len(self.mean_activation_buffer) > 15:
            data = np.array(self.mean_activation_buffer)
            # Simulate EEG signal by filtering the mean activation through bandpass filters
            for band, (low, high) in self.frequency_bands.items():
                filtered_signal = self.bandpass_filter(data, low, high, self.fs)
                power = np.sum(filtered_signal ** 2)
                self.eeg_data[band].append(power)
        else:
            # Not enough data yet, append zero to eeg_data
            for band in self.eeg_data.keys():
                self.eeg_data[band].append(0)

    def bandpass_filter(self, data, lowcut, highcut, fs, order=2):
        """Bandpass filter the data."""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y

    def compute_coherence(self):
        """Compute hub coherence using phase synchronization."""
        with self.network.node_lock:
            phase_coupling = np.zeros(len(self.mean_activation_buffer), dtype=np.complex128)
            data = np.array(self.mean_activation_buffer)
            analytic_signal = hilbert(data)
            phase = np.angle(analytic_signal)
            phase_coupling += np.exp(1j * phase)
            coherence = np.abs(phase_coupling) / len(phase)
            hub_coherence_value = np.mean(coherence) / SWEET_SPOT_RATIO  # Normalize by Sweet Spot Ratio
            return hub_coherence_value

class EEGVisualizer:
    """Window for visualizing EEG signals."""
    def __init__(self, parent, eeg_simulator: EEGSimulator):
        self.parent = parent
        self.eeg_simulator = eeg_simulator
        self.window = tk.Toplevel(parent)
        self.window.title("EEG Visualization")
        self.window.geometry("800x600")
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.create_widgets()
        self.anim = animation.FuncAnimation(self.fig, self.update_plot, interval=1000 / self.eeg_simulator.fs)
        self.window.after(0, self.update_visualization)

    def create_widgets(self):
        # Create a matplotlib figure
        self.fig, self.axs = plt.subplots(5, 1, figsize=(8, 6), sharex=True)
        self.fig.tight_layout()
        self.lines = {}
        for ax, band in zip(self.axs, self.eeg_simulator.frequency_bands.keys()):
            ax.set_xlim(0, 256)
            ax.set_ylim(0, 1)
            ax.set_ylabel(band.capitalize())
            line, = ax.plot([], [], lw=1)
            self.lines[band] = line
        self.axs[-1].set_xlabel('Time (samples)')

        # Embed the figure in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_plot(self, frame):
        for band, line in self.lines.items():
            data = list(self.eeg_simulator.eeg_data[band])
            line.set_data(range(len(data)), data)
        return list(self.lines.values())

    def update_visualization(self):
        self.eeg_simulator.update_eeg_data()
        self.canvas.draw()
        self.window.after(1000 // self.eeg_simulator.fs, self.update_visualization)

    def on_close(self):
        self.window.destroy()

# ---------------------------- Conscious AI Model ----------------------------
class ConsciousAIModel(nn.Module):
    def __init__(self):
        super(ConsciousAIModel, self).__init__()
        # Simple CNN for demonstration
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(32 * 13 * 13, 128)
        self.fc2 = nn.Linear(128, 4)  # Outputs: velocity, rotation, energy change, state influence

        # Attention mechanism
        self.attention = AttentionMechanism()

    def forward(self, x, current_state_resonance):
        # Ensure current_state_resonance is a tensor on the same device as x
        if not isinstance(current_state_resonance, torch.Tensor):
            current_state_resonance = torch.tensor([current_state_resonance], device=x.device, dtype=torch.float32)
        else:
            current_state_resonance = current_state_resonance.to(x.device)

        # Update attention level based on hub coherence and sensory salience
        sensory_salience = torch.mean(x)
        self.attention.update_attention(current_state_resonance, sensory_salience)

        # Apply attention to inputs
        x = self.attention.apply_attention(x)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.contiguous().view(x.size(0), -1)  # Flatten the tensor

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Incorporate Universal Hub influence
        hub_influence = current_state_resonance / PHASE_EFFICIENCY_RATIO
        x[:, 3] = torch.tanh(x[:, 3] * hub_influence)

        return x

# ---------------------------- Sensory Processor ----------------------------
class SensoryProcessor:
    def __init__(self, config: SystemConfig, network: AdaptiveNetwork):
        self.config = config
        self.network = network
        self.webcam = cv2.VideoCapture(self.config.camera_index)
        if not self.webcam.isOpened():
            raise RuntimeError(f"Failed to open webcam with index {self.config.camera_index}")
        logging.info(f"Webcam with index {self.config.camera_index} opened.")
        self.prev_gray = None  # For optical flow

    def process_complex_frame(self, frame: np.ndarray) -> Dict[str, float]:
        """Extract advanced visual features."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255.0
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges) / 255.0

        # Optical flow
        flow_magnitude = 0.0
        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_magnitude = np.mean(mag)
        self.prev_gray = gray

        return {'brightness': brightness, 'edge_density': edge_density, 'flow_magnitude': flow_magnitude}

    def cleanup(self):
        if self.webcam:
            self.webcam.release()
            logging.info("Webcam released.")

# ----------------------------- Adaptive System -----------------------------
class AdaptiveSystem:
    def __init__(self, gui_queue: queue.Queue, vis_queue: queue.Queue, config: SystemConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")

        self.network = AdaptiveNetwork(self.config, device=self.device)
        try:
            self.sensory_processor = SensoryProcessor(self.config, self.network)
        except RuntimeError as e:
            messagebox.showerror("Webcam Error", str(e))
            logging.error(f"Failed to initialize SensoryProcessor: {e}")
            self.sensory_processor = None
        self.gui_queue = gui_queue
        self.vis_queue = vis_queue
        self.running = False
        self.capture_thread = None
        self.last_growth_time = time.time()
        self.stop_event = threading.Event()  # Event to signal stop

        # Initialize AI Model
        self.model = ConsciousAIModel()
        self.model.eval()
        self.model.to(self.device)

        # Initialize current state
        self.network.current_state = STATE_PROPERTIES['Normal']  # Set initial state

        # Initialize EEG Simulator
        self.eeg_simulator = EEGSimulator(self.network)

        # Initialize CSV Logging
        self.log_file_path = 'conscious_ai_log.csv'
        try:
            self.log_file = open(self.log_file_path, 'w', newline='')
            self.csv_writer = csv.writer(self.log_file)
            self.csv_writer.writerow(['Frame', 'Energy', 'Coherence', 'State', 'Resonance', 'Velocity', 'Rotation', 'Energy Change', 'State Influence'])
            logging.info(f"CSV log file '{self.log_file_path}' initialized.")
        except Exception as e:
            logging.error(f"Failed to initialize CSV log file: {e}")
            self.csv_writer = None

    def start(self):
        if not self.running and self.sensory_processor is not None:
            self.running = True
            self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
            self.capture_thread.start()
            logging.info("Adaptive system started.")

    def stop(self):
        if self.running:
            self.running = False
            self.stop_event.set()
            if self.capture_thread:
                self.capture_thread.join(timeout=2)
            if self.sensory_processor:
                self.sensory_processor.cleanup()
            try:
                if not self.log_file.closed:
                    self.log_file.close()
                    logging.info(f"CSV log file '{self.log_file_path}' closed.")
            except Exception as e:
                logging.error(f"Error closing log file: {e}")
            logging.info("Adaptive system stopped.")

    def capture_loop(self):
        frame_count = 0
        while self.running and self.sensory_processor is not None and not self.stop_event.is_set():
            try:
                ret, frame = self.sensory_processor.webcam.read()
                if ret:
                    features = self.sensory_processor.process_complex_frame(frame)
                    self.latest_features = features  # Store for attention calculation

                    dx = (features['brightness'] - 0.5) * 2 * self.config.movement_speed
                    dy = (features['edge_density'] - 0.5) * 2 * self.config.movement_speed
                    self.network.update_position(dx, dy)

                    # AI Model Processing
                    processed_frame = cv2.resize(frame, (64, 64))
                    img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    img_normalized = img.astype(np.float32) / 255.0
                    img_tensor = torch.tensor(img_normalized, device=self.device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

                    with torch.no_grad():
                        output = self.model(img_tensor, self.network.current_state.resonance)
                        output = output.cpu()  # Ensure output is on CPU if needed
                        velocity, rotation, energy_change, state_influence = output[0]

                    velocity = velocity.item()
                    rotation = rotation.item()
                    energy_change = energy_change.item()
                    state_influence = state_influence.item()

                    logging.debug(f"Model Output - Velocity: {velocity}, Rotation: {rotation}, Energy Change: {energy_change}, State Influence: {state_influence}")

                    # Update AI parameters
                    delta_energy = energy_change * SWEET_SPOT_RATIO
                    delta_coherence = (state_influence - 0.5) * 2 * (PHASE_EFFICIENCY_RATIO / 100.0)

                    delta_energy = np.clip(delta_energy, -2.0, 2.0)
                    delta_coherence = np.clip(delta_coherence, -5.0, 5.0)

                    if self.network.energy < 20.0:
                        delta_energy += 0.5

                    self.network.energy = np.clip(self.network.energy + delta_energy, 0.0, 100.0)
                    self.network.coherence = np.clip(self.network.coherence + delta_coherence, 0.0, 100.0)

                    logging.debug(f"Updated Energy: {self.network.energy}, Updated Coherence: {self.network.coherence}")

                    previous_state = self.network.current_state.name
                    self.determine_next_state()

                    if previous_state != self.network.current_state.name:
                        logging.info(f"State changed from {previous_state} to {self.network.current_state.name}")

                    attention_level = self.calculate_attention_level()

                    # Update EEG data
                    self.eeg_simulator.update_eeg_data()

                    # Log data to CSV
                    if self.csv_writer:
                        frame_count += 1
                        self.log_data(frame_count, velocity, rotation, energy_change, state_influence)

                    # Node Visualization Data
                    with self.network.node_lock:
                        positions = [node.position for node in self.network.nodes.values()]
                        states = [np.mean(node.state) for node in self.network.nodes.values()]
                    vis_data = {'positions': positions, 'states': states}
                    if not self.vis_queue.full():
                        self.vis_queue.put(vis_data)

                    # GUI Data
                    gui_data = {
                        'frame': frame,
                        'position': self.network.position,
                        'direction': self.network.current_direction,
                        'state': self.network.current_state.name,
                        'energy': self.network.energy,
                        'coherence': self.network.coherence,
                        'attention_level': attention_level
                    }
                    if not self.gui_queue.full():
                        self.gui_queue.put(gui_data)

                    # Handle node growth and pruning
                    current_time = time.time()
                    if (current_time - self.last_growth_time) > 0.1:  # Every 100ms
                        if np.random.rand() < self.config.growth_rate:
                            self.network.add_node()
                        self.network.prune_nodes()
                        self.network.process_connections()
                        self.network.update_hub()
                        self.network.propagate_waves(current_time)
                        self.network.update_cells()
                        self.last_growth_time = current_time

            except Exception as e:
                logging.error(f"Error in capture loop: {e}")
            time.sleep(0.01)  # Maintain loop rate

    def determine_next_state(self):
        """
        Determine the next state based on energy and resonance hierarchy.
        Incorporate sweet spot and phase efficiency ratios.
        Implement hysteresis to prevent rapid state flipping.
        """
        potential_states = sorted(STATE_PROPERTIES.values(), key=lambda s: s.resonance, reverse=True)

        for state in potential_states:
            if self.network.energy >= state.energy_threshold:
                # Calculate transition potential using sweet spot ratio
                delta_resonance = abs(state.resonance - self.network.current_state.resonance)
                transition_potential = np.exp(-delta_resonance / SWEET_SPOT_RATIO)

                # Modify coherence based on phase efficiency ratio
                modified_coherence = self.network.coherence * (PHASE_EFFICIENCY_RATIO / 1000.0)

                # Implement hysteresis: require a higher threshold for transitioning to a new state
                if transition_potential * modified_coherence > state.coherence_threshold + 0.1:
                    self.network.current_state = state
                    logging.info(f"State transitioned to {state.name} based on transition potential and coherence.")
                    break

    def calculate_attention_level(self) -> float:
        movement_intensity = self.latest_features.get('flow_magnitude', 0.0)
        flow_magnitude = self.latest_features.get('flow_magnitude', 0.0)

        distance = np.sqrt(
            (self.network.position[0] - (self.config.display_width / 2)) ** 2 +
            (self.network.position[1] - (self.config.display_height / 2)) ** 2
        )
        max_distance = np.sqrt(
            (self.config.display_width / 2) ** 2 +
            (self.config.display_height / 2) ** 2
        )
        proximity_factor = max(0.0, 1 - (distance / max_distance))

        attention_level = (movement_intensity + flow_magnitude + proximity_factor) / 3.0
        attention_level = min(max(attention_level, 0.0), 1.0)

        return attention_level

    def log_data(self, frame_num, velocity, rotation, energy_change, state_influence):
        """Log the AI's state and actions to a CSV file."""
        try:
            self.csv_writer.writerow([
                frame_num,
                f"{self.network.energy:.2f}",
                f"{self.network.coherence:.2f}",
                self.network.current_state.name,
                f"{self.network.current_state.resonance:.2f}",
                f"{velocity:.2f}",
                f"{rotation:.2f}",
                f"{energy_change:.2f}",
                f"{state_influence:.2f}"
            ])
            logging.debug(f"Logged data for frame {frame_num}.")
        except (ValueError, IOError) as e:
            logging.error(f"Failed to write to log file: {e}")

    def load_system(self, filepath: str):
        """Load the system's configuration and node states from a JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            # Update configuration
            self.config.update_from_dict(data['config'])
            # Update nodes
            with self.network.node_lock:
                self.network.nodes = {}
                for node_id, node_info in data['nodes'].items():
                    state_name = node_info.get('state_info', 'Normal')
                    state = STATE_PROPERTIES.get(state_name, STATE_PROPERTIES['Normal'])
                    self.network.nodes[int(node_id)] = AdaptiveNode(
                        id=int(node_id),
                        device=self.device,
                        position=node_info['position'],
                        connections={int(k): np.array(v) for k, v in node_info['connections'].items()}
                    )
                    self.network.nodes[int(node_id)].state_info = state
            logging.info(f"System loaded from {filepath}.")
            messagebox.showinfo("Load System", f"System successfully loaded from {filepath}.")
        except Exception as e:
            logging.error(f"Failed to load system: {e}")
            messagebox.showerror("Load System", f"Failed to load system: {e}")

    def save_system(self, filepath: str):
        """Save the system's configuration and node states to a JSON file."""
        try:
            with open(filepath, 'w') as f:
                data = {
                    'config': self.config.to_dict(),
                    'nodes': {
                        node_id: {
                            'position': node.position,
                            'connections': node.connections,
                            'state_info': node.state_info.name
                        }
                        for node_id, node in self.network.nodes.items()
                    }
                }
                json.dump(data, f, indent=4)
            logging.info(f"System saved to {filepath}.")
            messagebox.showinfo("Save System", f"System successfully saved to {filepath}.")
        except Exception as e:
            logging.error(f"Failed to save system: {e}")
            messagebox.showerror("Save System", f"Failed to save system: {e}")

# ------------------------------- Node Visualizer -------------------------------
class NodeVisualizer:
    """Separate window for 3D node visualization."""
    def __init__(self, parent, vis_queue: queue.Queue):
        self.parent = parent
        self.vis_queue = vis_queue
        self.window = tk.Toplevel(parent)
        self.window.title("3D Node Visualization")
        self.window.geometry("800x600")
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.create_widgets()
        self.nodes_positions = []
        self.nodes_states = []
        self.colorbar = None  # Initialize the colorbar reference
        self.update_visualization()

    def create_widgets(self):
        # Create a matplotlib figure
        self.fig = plt.Figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([-2, 2])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title("Adaptive Network Nodes")

        # Embed the figure in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_visualization(self):
        try:
            while not self.vis_queue.empty():
                data = self.vis_queue.get_nowait()
                if 'positions' in data and 'states' in data:
                    self.nodes_positions = data['positions']
                    self.nodes_states = data['states']
                    logging.debug(f"NodeVisualizer received {len(self.nodes_positions)} nodes.")

            self.ax.cla()  # Clear the current axes
            self.ax.set_xlim([-2, 2])
            self.ax.set_ylim([-2, 2])
            self.ax.set_zlim([-2, 2])
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_title("Adaptive Network Nodes")

            # Extract positions and states
            if self.nodes_positions:
                xs, ys, zs = zip(*self.nodes_positions)
                states = self.nodes_states

                # Normalize positions for visualization
                xs_norm = [(x - min(xs)) / (max(xs) - min(xs) + 1e-5) * 4 - 2 for x in xs]
                ys_norm = [(y - min(ys)) / (max(ys) - min(ys) + 1e-5) * 4 - 2 for y in ys]
                zs_norm = [(z - min(zs)) / (max(zs) - min(zs) + 1e-5) * 4 - 2 for z in zs]

                # Normalize states for coloring
                states_norm = [(s - min(states)) / (max(states) - min(states) + 1e-5) for s in states]

                # Plot nodes with colors based on state
                scatter = self.ax.scatter(
                    xs_norm, ys_norm, zs_norm,
                    c=states_norm, cmap='plasma', marker='o', s=20, alpha=0.6
                )

                # Remove the previous colorbar if it exists
                if self.colorbar:
                    self.colorbar.remove()

                # Add a new colorbar
                self.colorbar = self.fig.colorbar(scatter, ax=self.ax, shrink=0.5, aspect=5)
                logging.debug(f"Plotted {len(xs_norm)} nodes.")
            else:
                logging.debug("No nodes to plot.")

            self.canvas.draw()
        except Exception as e:
            logging.error(f"Error in node visualization update: {e}")
        finally:
            self.window.after(100, self.update_visualization)  # Update every 100 ms

    def on_close(self):
        self.window.destroy()

# ----------------------------- Configuration Window -----------------------------
class ConfigWindow:
    """Configuration window for adjusting system parameters."""
    def __init__(self, parent, config: SystemConfig, adaptive_system: AdaptiveSystem):
        self.parent = parent
        self.config = config
        self.adaptive_system = adaptive_system
        self.window = tk.Toplevel(parent)
        self.window.title("Configuration")
        self.window.geometry("400x400")
        self.window.resizable(False, False)
        self.window.grab_set()  # Make the config window modal
        self.create_widgets()

    def create_widgets(self):
        padding = {'padx': 10, 'pady': 5}

        # Depth
        ttk.Label(self.window, text="Depth:").grid(row=0, column=0, sticky=tk.W, **padding)
        self.depth_var = tk.IntVar(value=self.config.depth)
        self.depth_spinbox = ttk.Spinbox(self.window, from_=1, to=10, textvariable=self.depth_var, width=10)
        self.depth_spinbox.grid(row=0, column=1, **padding)

        # Pruning Rate
        ttk.Label(self.window, text="Pruning Rate:").grid(row=1, column=0, sticky=tk.W, **padding)
        self.pruning_rate_var = tk.DoubleVar(value=self.config.pruning_threshold)
        self.pruning_rate_entry = ttk.Entry(self.window, textvariable=self.pruning_rate_var, width=12)
        self.pruning_rate_entry.grid(row=1, column=1, **padding)

        # Growth Rate
        ttk.Label(self.window, text="Growth Rate:").grid(row=2, column=0, sticky=tk.W, **padding)
        self.growth_rate_var = tk.DoubleVar(value=self.config.growth_rate)
        self.growth_rate_entry = ttk.Entry(self.window, textvariable=self.growth_rate_var, width=12)
        self.growth_rate_entry.grid(row=2, column=1, **padding)

        # Minimum Nodes
        ttk.Label(self.window, text="Minimum Nodes:").grid(row=3, column=0, sticky=tk.W, **padding)
        self.min_nodes_var = tk.IntVar(value=self.config.min_nodes)
        self.min_nodes_spinbox = ttk.Spinbox(self.window, from_=1, to=self.config.max_nodes, textvariable=self.min_nodes_var, width=10)
        self.min_nodes_spinbox.grid(row=3, column=1, **padding)

        # Maximum Nodes
        ttk.Label(self.window, text="Maximum Nodes:").grid(row=4, column=0, sticky=tk.W, **padding)
        self.max_nodes_var = tk.IntVar(value=self.config.max_nodes)
        self.max_nodes_spinbox = ttk.Spinbox(self.window, from_=self.config.min_nodes, to=10000, textvariable=self.max_nodes_var, width=10)
        self.max_nodes_spinbox.grid(row=4, column=1, **padding)

        # Webcam Selection
        ttk.Label(self.window, text="Webcam:").grid(row=5, column=0, sticky=tk.W, **padding)
        self.webcam_var = tk.IntVar(value=self.config.camera_index)
        self.webcam_combobox = ttk.Combobox(self.window, textvariable=self.webcam_var, state='readonly', width=8)
        self.webcam_combobox['values'] = self.detect_webcams()
        # Set current selection based on camera_index
        camera_str = str(self.config.camera_index)
        if camera_str in self.webcam_combobox['values']:
            self.webcam_combobox.current(self.webcam_combobox['values'].index(camera_str))
        else:
            self.webcam_combobox.current(0)
        self.webcam_combobox.grid(row=5, column=1, **padding)

        # Save and Load Buttons
        self.save_button = ttk.Button(self.window, text="Save Configuration", command=self.save_configuration)
        self.save_button.grid(row=6, column=0, **padding)

        self.load_button = ttk.Button(self.window, text="Load Configuration", command=self.load_configuration)
        self.load_button.grid(row=6, column=1, **padding)

        # Apply Button
        self.apply_button = ttk.Button(self.window, text="Apply", command=self.apply_changes)
        self.apply_button.grid(row=7, column=0, columnspan=2, pady=20)

    def detect_webcams(self, max_tested=5) -> List[str]:
        """Detect available webcams and return their indices as strings."""
        available_cameras = []
        for i in range(max_tested):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(str(i))
                cap.release()
        if not available_cameras:
            available_cameras.append("0")  # Default to 0 if no cameras found
        return available_cameras

    def save_configuration(self):
        """Save the current configuration and node states to a JSON file."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")],
            title="Save System Configuration"
        )
        if filepath:
            self.adaptive_system.save_system(filepath)

    def load_configuration(self):
        """Load configuration and node states from a JSON file."""
        filepath = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")],
            title="Load System Configuration"
        )
        if filepath:
            self.adaptive_system.load_system(filepath)
            # Update GUI elements with loaded configuration
            self.depth_var.set(self.config.depth)
            self.pruning_rate_var.set(self.config.pruning_threshold)
            self.growth_rate_var.set(self.config.growth_rate)
            self.min_nodes_var.set(self.config.min_nodes)
            self.max_nodes_var.set(self.config.max_nodes)
            camera_str = str(self.config.camera_index)
            if camera_str in self.webcam_combobox['values']:
                self.webcam_combobox.current(self.webcam_combobox['values'].index(camera_str))
            else:
                self.webcam_combobox.current(0)

    def apply_changes(self):
        """Apply the changes made in the configuration window."""
        try:
            # Retrieve values from the GUI
            new_depth = self.depth_var.get()
            new_pruning_rate = float(self.pruning_rate_var.get())
            new_growth_rate = float(self.growth_rate_var.get())
            new_min_nodes = self.min_nodes_var.get()
            new_max_nodes = self.max_nodes_var.get()
            new_camera_index = int(self.webcam_var.get())

            # Validate values
            if new_min_nodes > new_max_nodes:
                messagebox.showerror("Configuration Error", "Minimum nodes cannot exceed maximum nodes.")
                return

            # Update configuration
            self.config.depth = new_depth
            self.config.pruning_threshold = new_pruning_rate
            self.config.growth_rate = new_growth_rate
            self.config.min_nodes = new_min_nodes
            self.config.max_nodes = new_max_nodes
            self.config.camera_index = new_camera_index

            # Apply webcam change
            was_running = self.adaptive_system.running
            self.adaptive_system.stop()
            try:
                # Update webcam in sensory processor
                self.adaptive_system.config.camera_index = new_camera_index
                self.adaptive_system.sensory_processor = SensoryProcessor(self.adaptive_system.config, self.adaptive_system.network)
                if was_running:
                    self.adaptive_system.start()
            except RuntimeError as e:
                messagebox.showerror("Webcam Error", str(e))
                logging.error(f"Failed to change webcam: {e}")
                return

            messagebox.showinfo("Configuration", "Configuration applied successfully.")
            self.window.destroy()
        except Exception as e:
            logging.error(f"Error applying configuration: {e}")
            messagebox.showerror("Configuration Error", f"Failed to apply configuration: {e}")

# ----------------------------- GUI Application -----------------------------
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Adaptive System with EEG Visualization")
        self.root.geometry("1200x800")
        self.gui_queue = queue.Queue(maxsize=50)
        self.vis_queue = queue.Queue(maxsize=50)
        self.config = SystemConfig()
        self.system = AdaptiveSystem(self.gui_queue, self.vis_queue, self.config)
        self.node_visualizer = None  # Will hold the NodeVisualizer instance
        self.eeg_visualizer = None  # Will hold the EEGVisualizer instance
        self.create_widgets()
        self.update_gui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        # Create menu bar without camera selection to avoid conflicts
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Control frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.start_button = ttk.Button(control_frame, text="Start", command=self.start)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.stop_button.config(state=tk.DISABLED)

        # Add Node Visualization Button
        self.visualize_button = ttk.Button(control_frame, text="Visualize Nodes", command=self.open_node_visualizer)
        self.visualize_button.pack(side=tk.LEFT, padx=5)

        # Add EEG Visualization Button
        self.eeg_button = ttk.Button(control_frame, text="Show EEG", command=self.open_eeg_visualizer)
        self.eeg_button.pack(side=tk.LEFT, padx=5)

        # Add Config Button
        self.config_button = ttk.Button(control_frame, text="Config", command=self.open_config_window)
        self.config_button.pack(side=tk.LEFT, padx=5)

        # Add Save and Load Buttons
        self.save_button = ttk.Button(control_frame, text="Save", command=self.save_system)
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.load_button = ttk.Button(control_frame, text="Load", command=self.load_system)
        self.load_button.pack(side=tk.LEFT, padx=5)

        # Canvas for video feed
        self.canvas = tk.Canvas(self.root, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Optional: Add Attention Level Progress Bar
        self.attention_progress = ttk.Progressbar(self.root, orient='horizontal', length=200, mode='determinate')
        self.attention_progress.pack(side=tk.TOP, pady=10)

        # Optional: Add Attention Label
        self.attention_label = ttk.Label(self.root, text="Attention: 0%", font=("Helvetica", 12))
        self.attention_label.pack(side=tk.TOP)

        # State Display Label
        self.state_label = ttk.Label(self.root, text="State: Normal", font=("Helvetica", 16))
        self.state_label.pack(side=tk.BOTTOM, pady=10)

        # Energy and Coherence Display Labels
        self.energy_label = ttk.Label(self.root, text="Energy: 100.00%", font=("Helvetica", 12))
        self.energy_label.pack(side=tk.BOTTOM)
        self.coherence_label = ttk.Label(self.root, text="Coherence: 1.00", font=("Helvetica", 12))
        self.coherence_label.pack(side=tk.BOTTOM)

    def save_system(self):
        """Save the system's configuration and node states."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")],
            title="Save System"
        )
        if filepath:
            self.system.save_system(filepath)

    def load_system(self):
        """Load the system's configuration and node states."""
        filepath = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")],
            title="Load System"
        )
        if filepath:
            self.system.load_system(filepath)

    def open_config_window(self):
        """Open the configuration window."""
        ConfigWindow(self.root, self.config, self.system)

    def _on_canvas_resize(self, event):
        self.system.config.display_width = event.width
        self.system.config.display_height = event.height

    def update_gui(self):
        try:
            while not self.gui_queue.empty():
                data = self.gui_queue.get_nowait()
                if 'frame' in data and data['frame'] is not None:
                    # Process frame for display
                    frame = cv2.cvtColor(data['frame'], cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (self.canvas.winfo_width(), self.canvas.winfo_height()))
                    image = Image.fromarray(frame)
                    photo = ImageTk.PhotoImage(image=image)
                    self.canvas.delete("all")
                    self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                    self.canvas._photo = photo  # Keep a reference to prevent garbage collection

                    if 'position' in data:
                        x, y = data['position']
                        direction = data.get('direction', 0)
                        cone_length = self.system.config.vision_cone_length
                        cone_angle = np.pi / 4
                        p1 = (x, y)
                        p2 = (x + cone_length * np.cos(direction - cone_angle),
                              y + cone_length * np.sin(direction - cone_angle))
                        p3 = (x + cone_length * np.cos(direction + cone_angle),
                              y + cone_length * np.sin(direction + cone_angle))
                        self.canvas.create_polygon(
                            p1[0], p1[1], p2[0], p2[1], p3[0], p3[1],
                            fill='#00ff00', stipple='gray50', outline='#00ff00', width=2
                        )
                        radius = 10
                        self.canvas.create_oval(
                            x - radius, y - radius, x + radius, y + radius,
                            fill='#00ff00', outline='white', width=2
                        )

                        # Change canvas background based on state
                        state_color_map = {
                            'Normal': '#000000',        # Black
                            'Flow': '#1E90FF',          # Dodger Blue
                            'Meditation': '#32CD32',    # Lime Green
                            'Dream': '#FF69B4'          # Hot Pink
                        }
                        current_state = data.get('state', 'Normal')
                        canvas_color = state_color_map.get(current_state, '#000000')
                        self.canvas.config(bg=canvas_color)

            # Update State, Energy, and Coherence Labels
            if 'state' in data:
                current_state = data['state']
                self.state_label.config(text=f"State: {current_state}")

            if 'energy' in data:
                current_energy = data['energy']
                self.energy_label.config(text=f"Energy: {current_energy:.2f}%")

            if 'coherence' in data:
                current_coherence = data['coherence']
                self.coherence_label.config(text=f"Coherence: {current_coherence:.2f}")

            # Update Attention Level Progress Bar and Label
            if 'attention_level' in data:
                attention = data['attention_level']
                # Update attention progress bar
                self.attention_progress['value'] = attention * 100  # Progressbar expects a value between 0 and 100
                # Update attention label
                self.attention_label.config(text=f"Attention: {int(attention * 100)}%")

        except Exception as e:
            logging.error(f"Error updating GUI: {e}")

        self.root.after(33, self.update_gui)  # Approximately 30 FPS

    def start(self):
        self.system.start()
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        logging.info("System started via GUI.")

    def stop(self):
        self.system.stop()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        logging.info("System stopped via GUI.")

    def open_node_visualizer(self):
        if self.node_visualizer is None or not tk.Toplevel.winfo_exists(self.node_visualizer.window):
            self.node_visualizer = NodeVisualizer(self.root, self.vis_queue)
            logging.info("Node visualization window opened.")
        else:
            self.node_visualizer.window.lift()  # Bring to front if already open

    def open_eeg_visualizer(self):
        if self.eeg_visualizer is None or not tk.Toplevel.winfo_exists(self.eeg_visualizer.window):
            self.eeg_visualizer = EEGVisualizer(self.root, self.system.eeg_simulator)
            logging.info("EEG visualization window opened.")
        else:
            self.eeg_visualizer.window.lift()  # Bring to front if already open

    def on_close(self):
        if self.system.running:
            self.stop()
        if self.node_visualizer and tk.Toplevel.winfo_exists(self.node_visualizer.window):
            self.node_visualizer.window.destroy()
        if self.eeg_visualizer and tk.Toplevel.winfo_exists(self.eeg_visualizer.window):
            self.eeg_visualizer.window.destroy()
        self.root.destroy()

# ----------------------------- Main Function -----------------------------
def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
