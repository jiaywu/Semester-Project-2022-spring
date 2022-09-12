import numpy as np
import torch
import torch.nn as nn
from scipy import special
from scipy.linalg import dft
from matplotlib import pyplot as plt

# set parameters
Order = 5 #AR model order
#channel sequence length + AR order
pilot_length = 3 #length of pilot signal
# SNR = 10 #SNR in dB
SNR = [-10, 0, 10, 20]

mse = np.zeros_like(SNR)
mse2 = np.zeros_like(SNR)
channel_num = 50

v_mph = 60 # velocity of either TX or RX, in miles per hour
center_freq = 200e6 # RF carrier frequency in Hz
Fs = 1e4 # sample rate of simulation
N = 50 # number of sinusoids to sum

v = v_mph * 0.44704 # convert to m/s
fd = v*center_freq/3e8
print("max Doppler shift:", fd)
t = np.arange(0, 1e0, 1/Fs) # time vector. (start, stop, step)
print(len(t))
Seq_length = len(t)

# generate AR parameter d from Yule-Wlaker equation with Clarke's model
v = np.zeros(Order)
R = np.zeros((Order, Order))

for i in range(Order):
    v[i] = 2 * 0.5 * special.jv(0, 2 * np.pi * (i + 1) * (fd / Fs))  # Bessel function
    for j in range(Order):
        R[i][j] = 2 * 0.5 * special.jv(0, 2 * np.pi * abs(i - j) * (fd / Fs))

R = R + 1e-6 * np.eye(Order)

d = -1 * np.dot(np.linalg.inv(R), v)  # from Yule-Walker equation
sigma_w_2 = np.real(special.jv(0, 0) + np.dot(d, v))  # R_0 variance of w_t

# r_tensor = torch.from_numpy(np.asarray(10**(-SNR/10)))

print("sigma_w_2:", sigma_w_2)

# generate modal transition matrix F
F = np.zeros((Order, Order))
for i in range(Order - 1):
    for j in range(i, Order):
        if j == (i + 1):
            F[i][j] = 1
        else:
            pass

for i in range(Order):
    F[Order - 1][i] = -1 * d[Order - 1 - i]
# print(F)
F_real = np.zeros((2 * Order, 2 * Order))
F_real[0:Order, 0:Order] = F
F_real[Order:, Order:] = F
F_real_ten = torch.from_numpy(F_real).float()

# generate Nosie Corvariance of e
# E = np.zeros((Order, Order))
# E[Order - 1][Order - 1] = sigma_w_2
# E = 10000 * np.eye(Order)

E_real = np.zeros((2 * Order, 2 * Order))
E_real[Order - 1][Order - 1] = 0.5 * sigma_w_2
E_real[2 * Order - 1][2 * Order - 1] = 0.5 * sigma_w_2
E_real_2 = 100000 * np.eye(2 * Order)

# generate observation matrix H
H = np.zeros((pilot_length, Order), dtype=np.cdouble)

for i in range(pilot_length):
    H[i][Order - 1] = dft(pilot_length, scale=None)[2, i]
# print(H)

H_real = np.zeros((2 * pilot_length, 2 * Order))
H_real[0:pilot_length, 0: Order] = np.real(H)
H_real[0:pilot_length, Order:] = -1 * np.imag(H)
H_real[pilot_length:, 0: Order] = np.imag(H)
H_real[pilot_length:, Order:] = np.real(H)

[test_channel_instances] = torch.utils.data.DataLoader(
    torch.load('channel_instances_N50_fd_length1e4.pt'), pin_memory=False)


for i_SNR in range(0, len(SNR)):
    LOSS1 = 0.0
    LOSS2 = 0.0
    loss_fn = nn.MSELoss(reduction='mean')

    # generate Nosie Corvariance of v
    V = 10 ** (-SNR[i_SNR] / 10) * np.eye(pilot_length)
    V_real = 0.5 * 10 ** (-SNR[i_SNR] / 10) * np.eye(2 * pilot_length)

    for index in range(channel_num):
        h = test_channel_instances.squeeze()[index, :]
        h = h.numpy()

        test_target_imag = torch.empty((Order, len(t) - Order), dtype=torch.complex128)
        test_target = torch.empty((2 * Order, len(t) - Order))

        xh = np.zeros((Order, len(t) - Order), dtype=np.cdouble)
        for i in range(len(t) - Order):

            for j in range(Order):
                xh[j][i] = h[j + i]

        test_target_imag[::] = torch.from_numpy(xh)

        test_target[0:Order, :] = torch.from_numpy(np.real(xh)).float()
        test_target[Order:, :] = torch.from_numpy(np.imag(xh)).float()

        h_var = np.var(np.real(h))
        h_var2 = np.var(np.imag(h))
        print(h_var + h_var2)


        # generate observation with sending pilots(2nd. row in the DFT matrix)
        y = np.zeros((pilot_length, Seq_length - Order), dtype=np.cdouble)
        for i in range(Seq_length - Order):
            for j in range(pilot_length):
                y[j][i] = dft(pilot_length, scale=None)[2, j] * h[Order + i]
            # y[:,i] += np.transpose(np.random.multivariate_normal(np.zeros(pilot_length), 0.5*10**(-SNR/20)*np.eye(pilot_length)).view(np.complex128))
            y[:, i] += np.transpose(
                10 ** (-SNR[i_SNR] / 20) * (1 / np.sqrt(2)) * (np.random.randn(pilot_length) + 1j * np.random.randn(pilot_length)))
        # print(y)
        y_real = np.zeros((2 * pilot_length, Seq_length - Order))
        y_real[0:pilot_length, :] = np.real(y)
        y_real[pilot_length:, :] = np.imag(y)


        m1x_posterior = np.transpose(np.zeros(2 * Order))
        m2x_posterior = np.zeros((2 * Order, 2 * Order))
        m2x_posterior[0: Order, 0: Order] = 0.5 * R
        m2x_posterior[Order:, Order:] = 0.5 * R

        x = np.zeros((Order, Seq_length - Order), dtype=np.cdouble)
        x_real = np.zeros((2 * Order, Seq_length - Order), dtype=np.double)
        x_real_noisy = np.zeros((2 * Order, Seq_length - Order), dtype=np.double)

        for i in range(Seq_length - Order):
            #print(i)
            m1x_prior = np.dot(F_real, m1x_posterior)
            # print(m1x_prior)
            # m2x_prior = F * m2x_posterior * np.transpose(F) + E
            m2x_prior = np.dot(F_real, m2x_posterior)
            m2x_prior = np.dot(m2x_prior, np.transpose(F_real)) + E_real

            # KG = m2x_prior * np.transpose(np.conj(H)) * np.linalg.inv(H * m2x_prior * np.transpose(np.conj(H)) + V)
            KG = np.dot(H_real, m2x_prior)
            KG = np.dot(KG, np.transpose(H_real)) + V_real
            #print(KG)
            KG = np.dot(np.transpose(H_real), np.linalg.inv(KG))
            KG = np.dot(m2x_prior, KG)

            m1x_posterior = m1x_prior + np.dot(KG, (y_real[:, i] - np.dot(H_real, m1x_prior)))
            m2x_posterior = m2x_prior - np.dot(KG, np.dot(H_real, m2x_prior))

            for j in range(2 * Order):
                x_real[j][i] = m1x_posterior[j]

        for i in range(Seq_length - Order):
            #print(i)
            m1x_prior = np.dot(F_real, m1x_posterior)
            # print(m1x_prior)
            # m2x_prior = F * m2x_posterior * np.transpose(F) + E
            m2x_prior = np.dot(F_real, m2x_posterior)
            m2x_prior = np.dot(m2x_prior, np.transpose(F_real)) + E_real_2

            # KG = m2x_prior * np.transpose(np.conj(H)) * np.linalg.inv(H * m2x_prior * np.transpose(np.conj(H)) + V)
            KG = np.dot(H_real, m2x_prior)
            KG = np.dot(KG, np.transpose(H_real)) + V_real
            #print(KG)
            KG = np.dot(np.transpose(H_real), np.linalg.inv(KG))
            KG = np.dot(m2x_prior, KG)

            m1x_posterior = m1x_prior + np.dot(KG, (y_real[:, i] - np.dot(H_real, m1x_prior)))
            m2x_posterior = m2x_prior - np.dot(KG, np.dot(H_real, m2x_prior))

            for j in range(2 * Order):
                x_real_noisy[j][i] = m1x_posterior[j]

        # print(np.dot(KG, H))
        # print(np.shape(KG))

        x_real = torch.from_numpy(x_real).float()
        x_real_noisy = torch.from_numpy(x_real_noisy).float()

        h_estimated = x_real[Order - 1, :] + 1j * x_real[2 * Order - 1, :]
        h_noisy = x_real_noisy[Order - 1, :] + 1j * x_real_noisy[2 * Order - 1, :]
        # h_model = xm[Order - 1, :] + 1j * xm[2 * Order - 1, :]
        # h_model = h_model.numpy()

        h_mag = np.abs(h)  # take magnitude for the sake of plotting
        h_mag_dB = 10 * np.log10(h_mag)  # convert to dB
        h_est_mag = np.abs(h_estimated)  # take magnitude for the sake of plotting
        h_est_mag_dB = 10 * np.log10(h_est_mag)  # convert to dB
        h_est_noisy_mag = np.abs(h_noisy)  # take magnitude for the sake of plotting
        h_est_noisy_mag_dB = 10 * np.log10(h_est_noisy_mag)
        # h_est_model_mag = np.abs(h_model)  # take magnitude for the sake of plotting
        # h_est_model_mag_dB = 10 * np.log10(h_est_model_mag)

        #h_generated = torch.from_numpy(h)
        h_gen_real = torch.empty((1, 2 * (Seq_length - Order)))
        h_gen_real[0, 0:(Seq_length - Order)] = torch.from_numpy(np.real(h[Order:]))
        h_gen_real[0, (Seq_length - Order):] = torch.from_numpy(np.imag(h[Order:]))

        LOSS1 += loss_fn(torch.cat((x_real[Order - 1, :].unsqueeze(0), x_real[2 * Order - 1, :].unsqueeze(0)), 1),
                                       h_gen_real).item()
        LOSS2 += loss_fn(torch.cat((x_real_noisy[Order - 1, :].unsqueeze(0), x_real_noisy[2 * Order - 1, :].unsqueeze(0)), 1),
                                       h_gen_real).item()

        # mse = (np.abs(h_generated - h_estimated)**2).mean(axis='None')
        # mse = 10 * np.log10(np.mean(np.abs(h[Order:] - h_estimated) ** 2))

    mse[i_SNR] = 10 * np.log10(LOSS1 / channel_num)
    mse2[i_SNR] = 10 * np.log10(LOSS2 / channel_num)
    print("MSE_noisy = %.3f dB" % mse2[i_SNR])
    print("MSE_KF = %.3f dB" % mse[i_SNR])

noise_floor = np.zeros(len(SNR))
for i in range(len(SNR)):
    noise_floor[i] = - SNR[i]

p1, = plt.plot(SNR, mse, 'g')
p2, = plt.plot(SNR, mse, 'g^')
p3, = plt.plot(SNR, mse2, 'b')
p4, = plt.plot(SNR, mse2, 'bs')
p5, = plt.plot(SNR, noise_floor, 'r--')
plt.xlabel('SNR / dB')
plt.ylabel('NMSE / dB')
plt.legend([p2, p4, p5], ['Estimated with KF / AR order 5', 'Estimated without model knowledge', 'Noise floor'])
# plt.legend(['Estimated without model knowledge', 'Estimated with KF / AR order 5'])
plt.show()

'''
# Plot fading over time
plt.plot(t[Order:], h_est_noisy_mag_dB, 'pink')
plt.plot(t[Order:], h_est_mag_dB, 'r')
# plt.plot(t[Order:], h_est_mag_dB, 'limegreen')
# plt.plot(t[Order:], h_est_model_mag_dB, 'r')
plt.plot(t[Order:], h_mag_dB[Order:])
plt.xlabel("time/s")
plt.ylabel("Channel Magnitude")
plt.legend(['Estimated without model knowledge SNR=20dB', 'Estimated with KF / AR order 5 SNR=20dB', 'Generated'])
plt.axis([0, 1e0, -15, 5])
plt.show()
'''