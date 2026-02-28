# ===================================================================
# Bio ANNa - Loihi 2 Hardware Interface
#
# Author: chandan sheikder
# Date: 07 oct 2025
#
# Description:
# This module provides a simulated Hardware Abstraction Layer (HAL) for
# the Intel Loihi 2 neuromorphic chip. Its purpose is to hide the
# complexity of the underlying neuromorphic SDK (e.g., Intel Lava)
# from the high-level ROS nodes.
#
# The ROS nodes interact with this simple, clean interface, which in turn
# would contain the hardware-specific code to compile, load, and run
# Spiking Neural Networks (SNNs) on the physical chip.
#
# This simulated version uses a Leaky Integrate-and-Fire (LIF) neuron
# model to provide a realistic, stateful, and non-linear response to
# input spikes, mimicking the behavior of a real SNN.
# ===================================================================

import numpy as np
import time

class Loihi2Interface:
    """
    A simulated hardware abstraction layer for the Intel Loihi 2 chip.
    In a real implementation, this class would be a wrapper around the
    Intel Lava SDK functions.
    """
    def __init__(self, network_config_path: str, num_neurons: int, leak_rate: float = 0.1, threshold: float = 1.0):
        """
        Initializes the Loihi interface and simulates loading a network.
        
        Args:
            network_config_path (str): Path to the compiled network configuration file.
            num_neurons (int): The number of neurons in the simulated SNN.
            leak_rate (float): The rate at which membrane potential decays to zero.
            threshold (float): The membrane potential threshold for a neuron to spike.
        """
        self.get_logger().info(f"[Loihi Interface] Initializing with network from '{network_config_path}'...")
        
        # --- Hardware-Specific Setup (Simulated) ---
        # In a real scenario, this would involve using the Lava SDK to:
        # 1. Read the network configuration file.
        # 2. Establish a connection to the Loihi 2 board.
        # 3. Compile the network for the specific neurocores.
        # 4. Deploy the network to the hardware.
        time.sleep(0.5) # Simulate hardware handshake and loading time
        self._network_loaded = True
        self.get_logger().info("[Loihi Interface] Network successfully loaded onto simulated hardware.")

        # --- Internal LIF Neuron Model State ---
        self._num_neurons = num_neurons
        self._leak_rate = leak_rate
        self._threshold = threshold
        self._reset_potential = 0.0
        
        # Membrane potential for each neuron in the network
        self._membrane_potential = np.zeros(self._num_neurons)

    def run_step(self, input_spikes: np.ndarray) -> np.ndarray:
        """
        Simulates one computational step of the SNN on the chip.

        This method implements the core logic of Leaky Integrate-and-Fire neurons.
        
        Args:
            input_spikes (np.ndarray): A numpy array of shape (num_neurons,)
                                      representing the weighted input current/spikes
                                      to each neuron for this time step.
            
        Returns:
            np.ndarray: A binary numpy array of shape (num_neurons,) where a 1
                        indicates that the neuron has fired a spike in this step.
        """
        if not self._network_loaded:
            raise RuntimeError("Loihi network is not loaded or has been stopped.")
            
        if input_spikes.shape[0] != self._num_neurons:
            raise ValueError(f"Input spike vector size ({input_spikes.shape[0]}) does not match "
                             f"network neuron count ({self._num_neurons}).")

        # --- Leaky Integrate-and-Fire (LIF) Simulation ---
        
        # 1. Apply leak: Membrane potential decays over time.
        self._membrane_potential *= (1.0 - self._leak_rate)
        
        # 2. Integrate inputs: Add the new input currents/spikes.
        self._membrane_potential += input_spikes
        
        # 3. Check for firing: Find which neurons have crossed the threshold.
        output_spikes = (self._membrane_potential >= self._threshold).astype(np.uint8)
        
        # 4. Reset neurons that fired: Their potential is reset to the resting state.
        self._membrane_potential[output_spikes == 1] = self._reset_potential
        
        return output_spikes

    def stop(self):
        """Simulates stopping execution and releasing hardware resources."""
        self.get_logger().info("[Loihi Interface] Stopping execution and releasing hardware resources.")
        # In a real scenario, this would call SDK functions to stop the chip
        # and close the hardware connection gracefully.
        self._network_loaded = False
        
    def get_logger(self):
        # Simple logger for standalone use. In ROS, this would use the node's logger.
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger("Loihi2Interface")

# ===================================================================
# Self-Contained Test Block
# ===================================================================
if __name__ == '__main__':
    """
    This block allows the file to be run directly for testing the interface.
    It demonstrates how to initialize and use the Loihi2Interface class.
    """
    print("--- Running Loihi2Interface Standalone Test ---")
    
    NUM_TEST_NEURONS = 10
    loihi_sim = Loihi2Interface(
        network_config_path="test_net.net",
        num_neurons=NUM_TEST_NEURONS,
        leak_rate=0.2,
        threshold=1.0
    )
    
    # Create a strong, sustained input to a few neurons
    test_input_spikes = np.zeros(NUM_TEST_NEURONS)
    test_input_spikes[2] = 0.6
    test_input_spikes[3] = 0.8
    
    print(f"\nRunning simulation for 10 steps with constant input:\n{test_input_spikes}\n")
    print("Step | Membrane Potentials (Rounded)      | Output Spikes")
    print("----------------------------------------------------------------")
    
    for i in range(10):
        # Get the membrane potential *before* the step for visualization
        potentials_before = loihi_sim._membrane_potential.copy()
        
        # Run one step
        output = loihi_sim.run_step(test_input_spikes)
        
        # Print the state
        potential_str = np.round(potentials_before, 2)
        print(f"{i:4} | {str(potential_str):35} | {output}")

    loihi_sim.stop()
    print("\n--- Test Complete ---")