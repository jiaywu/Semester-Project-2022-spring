import torch
import csv
import numpy as np
from scipy import special
from scipy.linalg import dft

Order = 5 # AR model order
# channel sequence length + AR order
pilot_length = 3 # length of pilot signal
# SNR = 10 #SNR in dB

v_mph = 60 # velocity of either TX or RX, in miles per hour
center_freq = 200e6 # RF carrier frequency in Hz
Fs = 1e4 # sample rate of simulation
N = 50 # number of sinusoids to sum

v = v_mph * 0.44704 # convert to m/s
fd = v*center_freq/3e8
print("max Doppler shift:", fd)
t = np.arange(0, 1e0, 1/Fs)
# t_t = np.arange(0, 1 * 1e-2, 1/Fs)
Seq_length = len(t)
# Seq_length_t_c = len(t_t)

train_num = 500
cv_num = 100
test_num = 200

'''
training_input_imag = torch.empty((train_num, pilot_length, Seq_length_t_c - Order), dtype=torch.complex128)
training_target_imag = torch.empty((train_num, Order, Seq_length_t_c - Order), dtype=torch.complex128)
training_input = torch.empty((train_num, 2 * pilot_length, Seq_length_t_c - Order))
training_target = torch.empty((train_num, 2 * Order, Seq_length_t_c - Order))

cv_input_imag = torch.empty((cv_num, pilot_length, Seq_length_t_c - Order), dtype=torch.complex128)
cv_target_imag = torch.empty((cv_num, Order, Seq_length_t_c- Order), dtype=torch.complex128)
cv_input = torch.empty((cv_num, 2 * pilot_length, Seq_length_t_c - Order))
cv_target = torch.empty((cv_num, 2 * Order, Seq_length_t_c - Order))

test_input_imag = torch.empty((test_num, pilot_length, Seq_length - Order), dtype=torch.complex128)
test_target_imag = torch.empty((test_num, Order, Seq_length - Order), dtype=torch.complex128)
test_input = torch.empty((test_num, 2 * pilot_length, Seq_length - Order))
test_target = torch.empty((test_num, 2 * Order, Seq_length - Order))
'''

# generating channel instances
# training_channel_instances = torch.empty((train_num, len(t)), dtype=torch.complex128)
# cv_channel_instances = torch.empty((cv_num, len(t)), dtype=torch.complex128)
test_channel_instances = torch.empty((test_num, len(t)), dtype=torch.complex128)

'''
# training data
for it in range(train_num):
    hi_t = np.zeros(len(t))
    hq_t = np.zeros(len(t))
    phi = (np.random.rand()) * 2 * np.pi
    for i in range(N):
        # alpha = (np.random.rand() - 0.5) * 2 * np.pi
        # phi = (np.random.rand() - 0.5) * 2 * np.pi

        # x = x + np.random.randn() * np.cos(2 * np.pi * fd * t * np.cos(alpha) + phi)
        # y = y + np.random.randn() * np.sin(2 * np.pi * fd * t * np.cos(alpha) + phi)

        alpha = (np.random.rand()) * 2 * np.pi
        beta = (np.random.rand()) * 2 * np.pi

        hi_t = hi_t + np.cos(2 * np.pi * fd * t * np.cos((((2 * (i + 1) - 1) * np.pi + phi) / (4 * N))) + alpha)
        hq_t = hq_t + np.sin(2 * np.pi * fd * t * np.cos((((2 * (i + 1) - 1) * np.pi + phi) / (4 * N))) + beta)

    training_channel_instances[it, :] = torch.from_numpy((1 / np.sqrt(N)) * (hi_t + 1j * hq_t))  # this is what you would actually use when simulating the channel

# cv data
for it in range(cv_num):
    hi_c = np.zeros(len(t))
    hq_c = np.zeros(len(t))
    phi = (np.random.rand()) * 2 * np.pi
    for i in range(N):
        # alpha = (np.random.rand() - 0.5) * 2 * np.pi
        # phi = (np.random.rand() - 0.5) * 2 * np.pi

        # x = x + np.random.randn() * np.cos(2 * np.pi * fd * t * np.cos(alpha) + phi)
        # y = y + np.random.randn() * np.sin(2 * np.pi * fd * t * np.cos(alpha) + phi)

        alpha = (np.random.rand()) * 2 * np.pi
        beta = (np.random.rand()) * 2 * np.pi

        hi_c = hi_c + np.cos(2 * np.pi * fd * t * np.cos((((2 * (i + 1) - 1) * np.pi + phi) / (4 * N))) + alpha)
        hq_c = hq_c + np.sin(2 * np.pi * fd * t * np.cos((((2 * (i + 1) - 1) * np.pi + phi) / (4 * N))) + beta)

    cv_channel_instances[it, :] = torch.from_numpy((1 / np.sqrt(N)) * (hi_c + 1j * hq_c))  # this is what you would actually use when simulating the channel
'''

# test data
for it in range(test_num):
    hi_te = np.zeros(len(t))
    hq_te = np.zeros(len(t))
    phi = (np.random.rand()) * 2 * np.pi
    for i in range(N):
        # alpha = (np.random.rand() - 0.5) * 2 * np.pi
        # phi = (np.random.rand() - 0.5) * 2 * np.pi

        # x = x + np.random.randn() * np.cos(2 * np.pi * fd * t * np.cos(alpha) + phi)
        # y = y + np.random.randn() * np.sin(2 * np.pi * fd * t * np.cos(alpha) + phi)

        alpha = (np.random.rand()) * 2 * np.pi
        beta = (np.random.rand()) * 2 * np.pi

        hi_te = hi_te + np.cos(2 * np.pi * fd * t * np.cos((((2 * (i + 1) - 1) * np.pi + phi) / (4 * N))) + alpha)
        hq_te = hq_te + np.sin(2 * np.pi * fd * t * np.cos((((2 * (i + 1) - 1) * np.pi + phi) / (4 * N))) + beta)

    test_channel_instances[it, :] = torch.from_numpy((1 / np.sqrt(N)) * (hi_te + 1j * hq_te))  # this is what you would actually use when simulating the channel
    
#print(training_channel_instances.unsqueeze(0).size())
torch.save([test_channel_instances], 'channel_instances_N50_fd_length1e4.pt')


