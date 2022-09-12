import torch
import csv
import numpy as np

[training_channel_instances, cv_channel_instances, test_channel_instances] = torch.utils.data.DataLoader(
        torch.load('channel_instances_N50_length1e4.pt'), pin_memory=False)

torch.save([test_channel_instances], 'channel_instances_N50_fd_length1e4.pt')