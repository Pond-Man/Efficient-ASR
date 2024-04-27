
from torch import nn
import torch
from typing import List
import speechbrain as sb

from collections import OrderedDict


# Could not import this using speechbrain.lobes.models.EnhanceResnet.SEblock so I copied it here
# Credit to Peter Platinga and Speechbrain for writing this part
# https://speechbrain.readthedocs.io/en/latest/API/speechbrain.lobes.models.EnhanceResnet.html#speechbrain.lobes.models.EnhanceResnet.SEblock
class SEblock(torch.nn.Module):
    """Squeeze-and-excitation block.

    Defined: https://arxiv.org/abs/1709.01507

    Arguments
    ---------
    input_size : tuple of ints
        Expected size of the input tensor

    Example
    -------
    >>> inputs = torch.rand([10, 20, 30, 256])
    >>> se_block = SEblock(input_size=inputs.shape[-1])
    >>> outputs = se_block(inputs)
    >>> outputs.shape
    torch.Size([10, 1, 1, 256])
    """

    def __init__(self, input_size):
        super().__init__()
        self.linear1 = sb.nnet.linear.Linear(
            input_size=input_size, n_neurons=input_size
        )
        self.linear2 = sb.nnet.linear.Linear(
            input_size=input_size, n_neurons=input_size
        )

    def forward(self, x):
        """Processes the input tensor with a squeeze-and-excite block."""
        # torch.mean causes weird inplace error
        # x = torch.mean(x, dim=(1, 2), keepdim=True)
        count = x.size(1) * x.size(2)
        x = torch.sum(x, dim=(1, 2), keepdim=True) / count
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        return torch.sigmoid(x)

class CitrinetBlock(nn.Module):
    """Citrinet Block
    
    Defined: https://arxiv.org/pdf/2104.01721
    
    Arguments
    _________
    R : Number of Citrinet Blocks to create
    in_channels: Number of input channels for convolutions
    out_channels: Number of output channels for convolutions
    kernel_size: kernel size for the depthwise convolution
    dropout: dropout probability
    
    Description
    ___________
    From https://arxiv.org/pdf/2104.01721: 
    Each residual block consists of basic QuartzNet blocks, repeated R times,
    plus a Squeeze and Excitation module in the end. A QuartzNet block is composed
    of 1D time-channel separable convolution with kernel K, batchnorm, ReLU, and dropout layers.
    
    """
    def __init__(self, R, in_channels, out_channels, dropout, kernel_size):
        super().__init__()
        
        # Activation + Normalization used in the paper
        self.activation_function = nn.ReLU()
        normalization = nn.BatchNorm1d
        
        # Initialize a citrinet_block
        self.citrinet_blocks = nn.ModuleList()

        # Create R blocks
        for i in range(R-1):               
            citrinet_module = nn.ModuleList([
                nn.Conv1d(in_channels=in_channels if i == 0 else out_channels,  # 1D Depthwise Conv
                          out_channels=out_channels,
                          groups=in_channels if i == 0 else out_channels,
                          kernel_size=kernel_size,
                          padding="same"),
                nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1), # Pointwise Conv
                normalization(out_channels), # BN
                self.activation_function, # ReLU
                nn.Dropout(dropout)]) # Dropout
            self.citrinet_blocks.extend(citrinet_module)
        
        # Last R block with SE
        self.last_citrinet_block = nn.Sequential(
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, # Depthwise Conv
                      groups=out_channels, kernel_size=kernel_size, padding="same"), 
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1), # Pointwise Conv
            normalization(out_channels)) # BN
        
        # Create residual connection
        self.skip_connection = nn.Sequential(nn.Conv1d(in_channels=out_channels, # Pointwise Conv
                                                       out_channels=out_channels, 
                                                       kernel_size=1),
                                             normalization(out_channels)) # BN
        
        # Final dropout layer
        self.drop = nn.Dropout(dropout) # Dropout

        
    def forward(self, x):
        output = x
        # Pass through R modules
        for module in self.citrinet_blocks:
            output = module(output)
        output = self.last_citrinet_block(output)
        
        # Squeeze and excitation
        output = output.unsqueeze(-1)
        se_block = SEblock(input_size=output.shape[-1]).to(torch.device("cuda"))
        output = se_block(output)
        output = output.squeeze(1)
        
        # Residual connection
        output = self.activation_function(output + self.skip_connection(x))
        output = self.drop(output)
        return output


class Citrinet(nn.Module):
    """
    Citrinet’s design implementation based on the proposed paper.
    """

    def __init__(self, 
                 n_feats=80, 
                 n_class=31,
                 hidden_channels=256,
                 B=5,
                 S=1,
                 R=5,
                 prolog_kernel=5,
                 kernels_B1=[11,13,15,17,19,21],
                 kernels_B2=[13,15,17,19,21,23,25],
                 kernels_B3=[25,27,29,31,33,35,37,39],
                 epilog_kernel=41,
                 dropout=0.2
                ):
        super().__init__()

        self.n_feats = n_feats

        # Activation + Normalization used in the paper
        activation_function = nn.ReLU
        normalization = nn.BatchNorm1d

        ###################### Prolog ######################
        self.B0 = nn.Sequential(nn.Conv1d(in_channels=n_feats,  # 1D Depthwise Conv
                                           out_channels=hidden_channels, 
                                           groups=n_feats,  # The depthwise convolution is 
                                                            # applied independently for
                                                            # each channel
                                           kernel_size=prolog_kernel, 
                                           padding=(prolog_kernel-1)//2),
                                    normalization(hidden_channels), # BN
                                    activation_function() # ReLU
                                   )
                                    
        ###################### Mega block 1 ######################
        self.B1_B6 = nn.Sequential()
        
        # 1D time-channel separable convolutional layer with stride 2
        self.B1_B6.add_module(
            'stride2_depthwise',
            nn.Conv1d(in_channels=hidden_channels,  # 1D Depthwise Conv
                      out_channels=hidden_channels,
                      groups=hidden_channels,
                      kernel_size=kernels_B1[0],
                      padding="same")
        )
        self.B1_B6.add_module(
            'stride2_pointwise',
            nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=1) # Pointwise Conv
        )
        
        # Get rid of the first kernel, it is only for the stride 2 convolution
        kernels_B1.pop(0)    
        
        # Create mega block
        for i, kernel in enumerate(kernels_B1):
            self.B1_B6.add_module(
            f'citrinet_block_{i}',
                CitrinetBlock(R=R, 
                             in_channels=hidden_channels,
                             out_channels=hidden_channels,
                             kernel_size=kernel,
                             dropout=dropout)
            )
        
        ###################### Mega block 2 ######################
        self.B7_B13 = nn.Sequential()
        
        # 1D time-channel separable convolutional layer with stride 2
        self.B7_B13.add_module(
            'stride2_depthwise',
            nn.Conv1d(in_channels=hidden_channels,  # 1D Depthwise Conv
                      out_channels=hidden_channels,
                      groups=hidden_channels,
                      kernel_size=kernels_B2[0],
                      padding="same")
        )
        self.B7_B13.add_module(
            'stride2_pointwise',
            nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=1) # Pointwise Conv
        )
        
        # Get rid of the first kernel, it is only for the stride 2 convolution
        kernels_B2.pop(0)    
        
        # Create mega block
        for i, kernel in enumerate(kernels_B2):
            self.B7_B13.add_module(
            f'citrinet_block_{i}',
                CitrinetBlock(R=R, 
                             in_channels=hidden_channels,
                             out_channels=hidden_channels,
                             kernel_size=kernel,
                             dropout=dropout)
            )
        
        ###################### Mega block 3 ######################
        self.B14_B21 = nn.Sequential()
        
        # 1D time-channel separable convolutional layer with stride 2
        self.B14_B21.add_module(
            'stride2_depthwise',
            nn.Conv1d(in_channels=hidden_channels,  # 1D Depthwise Conv
                      out_channels=hidden_channels,
                      groups=hidden_channels,
                      kernel_size=kernels_B3[0],
                      padding="same")
        )
        self.B14_B21.add_module(
            'stride2_pointwise',
            nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=1) # Pointwise Conv
        )
        
        # Get rid of the first kernel, it is only for the stride 2 convolution
        kernels_B3.pop(0)    
        
        # Create mega block
        for i, kernel in enumerate(kernels_B3):
            self.B14_B21.add_module(
            f'citrinet_block_{i}',
                CitrinetBlock(R=R, 
                             in_channels=hidden_channels,
                             out_channels=hidden_channels,
                             kernel_size=kernel,
                             dropout=dropout)
            )
        
        ###################### Epilog ######################
        self.B22 = nn.Sequential(nn.Conv1d(in_channels=hidden_channels,  # 1D Depthwise Conv
                                           out_channels=hidden_channels, 
                                           groups=hidden_channels,
                                           kernel_size=epilog_kernel, 
                                           padding=(epilog_kernel-1)//2),
                                    normalization(hidden_channels), # BN
                                    activation_function() # ReLU
                                   )
            
        ###################### Out ######################
        self.out = nn.Conv1d(in_channels=hidden_channels, out_channels=n_class, kernel_size=1) # Pointwise Conv
    
    def forward(self, inputs):
        # Use permute, so we have (batch_size, mel_scale, mel_length)
        output = self.B0(inputs.permute(0, 2, 1))
        output = self.B1_B6(output)
        output = self.B7_B13(output)
        output = self.B14_B21(output)
        output = self.B22(output)
        output = self.out(output)
        return output.permute(0, 2, 1)
