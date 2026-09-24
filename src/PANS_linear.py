import torch
import torch.nn as nn
import torch.nn.functional as F
from PhotonActivation import PhotonActivation, PhotonActivationCoh

class PANS_active(nn.Module):
    """Active PANS with structured illumination encoding.
    
    Args:
        n_input: Input dimension (d_obj in paper, e.g., 784 for 28x28)
        n_output: Number of output classes
        n_feature: Number of features/illumination patterns (d_\text{f} in paper)
        in_channels: Input channels for conv layers (default 1)
        conv_channels: Channel config for conv blocks ([] to disable)
        conv_kernel: Kernel size for conv layers
        pool_size: Pooling window size
        pool_type: Pooling type
        fc_units: Hidden units in fully connected layers
        use_feature_batchnorm: Use batch normalization on the feature vector
        use_logsoft: Use log softmax in the output layer (used in training)
    """
    
    def __init__(self, n_input, n_output, n_feature, in_channels=1,
                 conv_channels=[[64,64], [128,128], [256,256,256]],
                 conv_kernel=15, pool_size=2, pool_type='avg', fc_units=[512],
                 use_feature_batchnorm=True, use_logsoft=True):
        super().__init__()

        self.n_feature = n_feature
        self.in_channels = in_channels
        self.use_feature_batchnorm = use_feature_batchnorm
        self.use_logsoft = use_logsoft
        
        # Optical front-end: trainable illumination patterns
        self.input_weights = nn.Parameter(torch.empty(n_feature, n_input))
        nn.init.kaiming_uniform_(self.input_weights, a=1.2)
        
        # Photon detection layer (defined in Section 1)
        self.photon_act = PhotonActivation()
        
        # Digital back-end: Batch normalization
        if use_feature_batchnorm:
            self.feature_batchnorm = nn.BatchNorm1d(n_feature)
        
        # Digital back-end: Conv blocks
        conv_modules = []
        conv_output_size = n_feature
        for block in conv_channels:
            for out_ch in block:
                conv_modules.extend([
                    nn.Conv1d(in_channels, out_ch, conv_kernel,
                              padding=conv_kernel // 2),
                    nn.BatchNorm1d(out_ch), nn.ReLU()])
                in_channels = out_ch

            if pool_type == 'max':
                conv_modules.append(nn.MaxPool1d(pool_size))
            elif pool_type == 'avg':
                conv_modules.append(nn.AvgPool1d(pool_size))
            else:
                raise ValueError('Invalid pooling type')

            conv_output_size //= pool_size
        self.conv_layers = nn.Sequential(*conv_modules)
        
        # Digital back-end: Fully connected layers
        if len(conv_channels) > 0:
            fc_input_size = conv_output_size * conv_channels[-1][-1]
        else:
            fc_input_size = n_feature
        fc_modules = []
        for units in fc_units:
            fc_modules.extend([nn.Linear(fc_input_size, units), nn.ReLU()])
            fc_input_size = units
        fc_modules.append(nn.Linear(fc_input_size, n_output))
        self.fc_layers = nn.Sequential(*fc_modules)

    def forward(self, x, n_rep=1, slope=1.0):
        x = x.view(x.size(0), -1)
        # Optical encoding: squared weights ensure non-negativity
        x = F.linear(x, self.input_weights ** 2)
        # Stochastic photon detection
        x = self.photon_act(x, n_rep=n_rep, slope=slope)
        # Digital processing
        if self.use_feature_batchnorm:
            x = self.feature_batchnorm(x)
        x = self.conv_layers(x.unsqueeze(1))
        x = self.fc_layers(x.view(x.size(0), -1))
        if self.use_logsoft:
            x = F.log_softmax(x, dim=1)
        return x

    def robust_test(self, x, dcr=0.001, atten=1.0, n_rep=1, slope=1.):
        x = x.view(x.size(0), -1)*atten
        x = F.linear(x, self.input_weights**2)
        x = self.photon_act(x+dcr, n_rep=n_rep, slope=slope)
        if self.use_feature_batchnorm:
            x = self.feature_batchnorm(x)
        # Pass through convolutional layers
        x = x.view(x.size(0), 1, -1)
        x = self.conv_layers(x)
        # Flatten and pass through fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    def get_acts(self, x, n_rep=1, slope=1.0):
        x = x.view(x.size(0), -1)
        # Optical encoding: squared weights ensure non-negativity
        x = F.linear(x, self.input_weights ** 2)
        # Stochastic photon detection
        x = self.photon_act(x, n_rep=n_rep, slope=slope)
        return x
    
    def acts_out(self, x):
        shape = x.shape[:-1]
        x = x.view(-1,x.shape[-1])
        if self.use_feature_batchnorm:
            x = self.feature_batchnorm(x)
        x = x.unsqueeze(-2)
        x = self.conv_layers(x)
        x = x.view(x.shape[0],-1)
        x = self.fc_layers(x)
        return x.view(shape.__add__((x.shape[-1],)))

class PANS_passive(nn.Module):
    """Passive PANS with structured illumination encoding.
    
    Args:
        n_input: Input dimension (d_obj in paper, e.g., 784 for 28x28)
        n_output: Number of output classes
        n_feature: Number of features (d_\text{f} in paper)
        in_channels: Input channels for conv layers (default 1)
        conv_channels: Channel config for conv blocks ([] to disable)
        conv_kernel: Kernel size for conv layers
        pool_size: Pooling window size
        pool_type: Pooling type
        fc_units: Hidden units in fully connected layers
        use_feature_batchnorm: Use batch normalization on the feature vector
        use_logsoft: Use log softmax in the output layer (used in training)
    """
    
    def __init__(self, n_input, n_output, n_feature, in_channels=1,
                 conv_channels=[[64,64], [128,128], [256,256,256]],
                 conv_kernel=15, pool_size=2, pool_type='avg', fc_units=[512],
                 use_feature_batchnorm=True, use_logsoft=True):
        super().__init__()

        self.n_feature = n_feature
        self.in_channels = in_channels
        self.use_feature_batchnorm = use_feature_batchnorm
        self.use_logsoft = use_logsoft
        
        # Optical front-end: trainable optical linear processor
        self.input_layer = nn.Linear(n_input,n_feature,bias=False)
        
        # Photon detection layer (defined in Section 1)
        self.photon_act = PhotonActivationCoh()
        
        # Digital back-end: Batch normalization
        if use_feature_batchnorm:
            self.feature_batchnorm = nn.BatchNorm1d(n_feature)
        
        # Digital back-end: Conv blocks
        conv_modules = []
        conv_output_size = n_feature
        for block in conv_channels:
            for out_ch in block:
                conv_modules.extend([
                    nn.Conv1d(in_channels, out_ch, conv_kernel,
                              padding=conv_kernel // 2),
                    nn.BatchNorm1d(out_ch), nn.ReLU()])
                in_channels = out_ch

            if pool_type == 'max':
                conv_modules.append(nn.MaxPool1d(pool_size))
            elif pool_type == 'avg':
                conv_modules.append(nn.AvgPool1d(pool_size))
            else:
                raise ValueError('Invalid pooling type')

            conv_output_size //= pool_size
        self.conv_layers = nn.Sequential(*conv_modules)
        
        # Digital back-end: Fully connected layers
        if len(conv_channels) > 0:
            fc_input_size = conv_output_size * conv_channels[-1][-1]
        else:
            fc_input_size = n_feature
        fc_modules = []
        for units in fc_units:
            fc_modules.extend([nn.Linear(fc_input_size, units), nn.ReLU()])
            fc_input_size = units
        fc_modules.append(nn.Linear(fc_input_size, n_output))
        self.fc_layers = nn.Sequential(*fc_modules)

    def forward(self, x, n_rep=1, slope=1.0):
        x = x.view(x.size(0), -1)
        # Optical encoding: passive linear operation
        x = self.input_layer(x)
        # Stochastic photon detection
        x = self.photon_act(x, n_rep=n_rep, slope=slope)
        # Digital processing
        if self.use_feature_batchnorm:
            x = self.feature_batchnorm(x)
        x = self.conv_layers(x.unsqueeze(1))
        x = self.fc_layers(x.view(x.size(0), -1))
        if self.use_logsoft:
            x = F.log_softmax(x, dim=1)
        return x

    def robust_test(self, x, dcr=0.001, atten=1.0, n_rep=1, slope=1.):
        x = x.view(x.size(0), -1)*atten
        x = self.input_layer(x)
        x = self.photon_act(x+dcr, n_rep=n_rep, slope=slope)
        x += torch.poisson(torch.ones_like(x)*dcr)
        x[x>1] = 1.
        if self.use_feature_batchnorm:
            x = self.feature_batchnorm(x)
        # Pass through convolutional layers
        x = x.view(x.size(0), 1, -1)
        x = self.conv_layers(x)
        # Flatten and pass through fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    def get_acts(self, x, n_rep=1, slope=1.0):
        x = x.view(x.size(0), -1)
        # Optical encoding: passive linear operation
        x = self.input_layer(x)
        # Stochastic photon detection
        x = self.photon_act(x, n_rep=n_rep, slope=slope)
        return x
    
    def acts_out(self, x):
        shape = x.shape[:-1]
        x = x.view(-1,x.shape[-1])
        if self.use_feature_batchnorm:
            x = self.feature_batchnorm(x)
        x = x.unsqueeze(-2)
        x = self.conv_layers(x)
        x = x.view(x.shape[0],-1)
        x = self.fc_layers(x)
        return x.view(shape.__add__((x.shape[-1],)))