import snntorch as snn
import torch.nn as nn
from snntorch import surrogate
import torch

# Global Parameters
v_thr = 1.0
placeholder_val = 0.5

# Define the surrogate gradient function to propagate spikes through the network
spike_grad = surrogate.fast_sigmoid()   # surrogate.atan()   
# Parameters for Dense Layers
inputDataDim = 2       # max_channel_idx - min_channel_idx + 1    # Number of input channels

input_to_hidden = (inputDataDim, 24) # 16 # TODO: Increase the size of this layer # (inputDataDim, 100) # (inputDataDim, 500)  # Number of neurons in the first Fully-Connected Layer

hiddenL2Dim = (input_to_hidden[1], input_to_hidden[1])  # Number of neurons in the Recurrent Fully-Connected Layer (L2)

hiddenL3Dim = (input_to_hidden[1], 16)  # Number of neurons in the Fully-Connected Layer (L3)

hiddenL4Dim = (hiddenL3Dim[1], input_to_hidden[1])  # Number of neurons in the Recurrent Fully-Connected Layer (L4)

hidden_to_out = (hiddenL3Dim[1], 1)  # Number of neurons in the Output Fully-Connected Layer
# In this case, we only need 1 output neuron -> Fires when HFO is detected

    # Define Network
class Net_P(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        
        # Create a Linear Layer to serve input to LIF1
        self.fc_in = nn.Linear(input_to_hidden[0], input_to_hidden[1],
                bias=False,
                dtype=torch.float32     # Set the data type of the weights to float32
        )

        self.lif1 = snn.Synaptic(
            alpha=torch.full(size=(input_to_hidden[1],), fill_value=placeholder_val), 
            beta=torch.full(size=(input_to_hidden[1],), fill_value=placeholder_val),
            threshold=v_thr,
            reset_mechanism="zero", reset_delay=False,
            # init_hidden=True,   # enables the methods in snntorch.backprop to automatically clear the hidden states and detach them from the comp. graph
            spike_grad=spike_grad,
            learn_alpha=True,   # Learn the alpha parameter
            learn_beta=True,    # Learn the beta parameter
            learn_threshold=False,   # Learn the threshold parameter
            
        )      

        """ self.fc2 = nn.Linear(
            hiddenL2Dim[0], hiddenL2Dim[1],
            bias=False,
            dtype=torch.float32     # Set the data type of the weights to float32
        ) """

        self.fc3 = nn.Linear(
            hiddenL3Dim[0], hiddenL3Dim[1],
            bias=False,
            dtype=torch.float32     # Set the data type of the weights to float32
        )

        self.lif2 = snn.Synaptic(
            alpha=torch.full(size=(hiddenL3Dim[1],), fill_value=placeholder_val), 
            beta=torch.full(size=(hiddenL3Dim[1],), fill_value=placeholder_val),
            threshold=v_thr,
            reset_mechanism="zero", reset_delay=False,
            # TODO: How to add Refractory Period?
            # init_hidden=True,   # enables the methods in snntorch.backprop to automatically clear the hidden states and detach them from the comp. graph
            spike_grad=spike_grad,
            learn_alpha=True,   # Learn the alpha parameter
            learn_beta=True,    # Learn the beta parameter
            learn_threshold=False,   # Learn the threshold parameter
        )   

        """ self.fc4 = nn.Linear(
            hiddenL4Dim[0], hiddenL4Dim[1],
            bias=False,
            dtype=torch.float32     # Set the data type of the weights to float32
        ) """

        self.fc_out = nn.Linear(
            hidden_to_out[0], hidden_to_out[1],
            bias=False,
            dtype=torch.float32     # Set the data type of the weights to float32
        )

        self.lif_out = snn.Synaptic(
            alpha=placeholder_val, 
            beta=placeholder_val,
            threshold=v_thr,
            reset_mechanism="zero", reset_delay=False,
            # init_hidden=True,   # enables the methods in snntorch.backprop to automatically clear the hidden states and detach them from the comp. graph
            spike_grad=spike_grad,
            learn_alpha=True,   # Learn the alpha parameter
            learn_beta=True,    # Learn the beta parameter
            learn_threshold=False,   # Learn the threshold parameter
        )

        # Initialize the membrane potential of each LIF neuron
        self.syn1, self.mem1, self.spk1 = None, None, None
        self.syn2, self.mem2, self.spk2 = None, None, None
        self.syn_out, self.mem_out, self.spk_out = None, None, None

    def reset_state(self):
        self.syn1, self.mem1, self.spk1 = None, None, None
        self.syn2, self.mem2, self.spk2 = None, None, None
        self.syn_out, self.mem_out, self.spk_out = None, None, None
        
    """
    Function called during the forward pass of the network
    """
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Forward Pass of the Network (Single Step Update)

        Parameters:
        - x: input tensor. Shape: (batch_size, num_features)

        Returns:
        - spk_vals: tuple of tensors containing the spikes of the neurons. Shape: (batch_size, num_neurons)
        - mem_vals: tuple of tensors containing the membrane potentials of the neurons. Shape: (batch_size, num_neurons)
        - syn_vals: tuple of tensors containing the currents of the neurons. Shape: (batch_size, num_neurons)
        '''
        
        
        cur_batch_size, duration, cur_num_channels = x.shape
        
        # --- Lazy State Initialization
        if self.mem1 is None:
            device = x.device   # Get the device of the input tensor

            # Initialize the membrane potential of each LIF neuron
            self.syn1, self.mem1 = self.lif1.reset_mem()
            self.syn2, self.mem2 = self.lif2.reset_mem()
            self.syn_out, self.mem_out = self.lif_out.reset_mem()

            # # Define small residual for spk1
            # spk1_factor = 0.01
            # self.spk1 = torch.rand(size=(cur_batch_size, input_to_hidden[1]), dtype=torch.float32, device=device) * spk1_factor
            # self.spk2 = torch.zeros(size=(cur_batch_size, hiddenL3Dim[1]), dtype=torch.float32, device=device) * spk1_factor
            # self.spk_out = torch.zeros(size=(cur_batch_size, hidden_to_out[1]), dtype=torch.float32, device=device)

        
        spk_rec: torch.Tensor[float] =[]

        if len(x.shape) == 1:
            # If the input is 1D, it means we have only one feature (one channel)
            # Unsqueeze the input to add the num_features dimension
            x = x.unsqueeze(1)
            
        for t in range(duration):
            input_t= x[:, t, :]
            ############# State Update #############
            # Calculate Input Current for LIF1 from the Input Layer (FC1) Input -> LIF1
            cur_fc1 = self.fc_in(input_t) 
        
            # Calculate Input Current from Recurrent Layer (FC2) LIF1 -> LIF1
            # cur_fc2 = self.fc2(spk1)   # Connect LIF1 to itself using FC Layer 2 (Recurrent Layer)

            # Join the input currents for LIF1 (FC1 + FC2)
            cur1 = cur_fc1 # + cur_fc2  # TODO: Not feeding Recurent Layer to LIF1 for now

            # Feed the joined input current to LIF1
            self.spk1, self.syn1, self.mem1 = self.lif1(cur1, self.syn1, self.mem1)  # Feed input to LIF1

            # Calculate Input Current for LIF2 from LIF1 (FC3) LIF1 -> LIF2
            cur2 = self.fc3(self.spk1)   # Connect LIF1 to LIF2 using FC Layer 3
            # Feed the input current to LIF2 and get the spikes, synaptic currents and membrane potentials
            self.spk2, self.syn2, self.mem2 = self.lif2(cur2, self.syn2, self.mem2)  # Feed input to LIF2

            # Calculate Input Current for LIF_OUT from LIF2 (FC4) LIF2 -> LIF_OUT
            cur_out = self.fc_out(self.spk2)
            # Feed the input current to LIF_OUT and get the spikes, synaptic currents and membrane potentials
            self.spk_out, self.syn_out, self.mem_out = self.lif_out(cur_out, self.syn_out, self.mem_out)  # Feed input to LIF_OUT

            # Return the currents, membrane potentials and spikes of the current timestep
            syn_val = (self.syn1, self.syn2, self.syn_out)
            mem_vals = (self.mem1, self.mem2, self.mem_out)
            spk_vals = (self.spk1, self.spk2, self.spk_out)

            spk_rec.append(self.spk_out)
        return spk_rec