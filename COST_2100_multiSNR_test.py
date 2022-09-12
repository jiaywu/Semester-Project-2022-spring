import numpy as np
import torch
import torch.nn as nn
from scipy import special
from scipy.linalg import dft
from matplotlib import pyplot as plt
import scipy.io
import os
from sklearn.linear_model import LinearRegression

path = 'cost2100/matlab'

h_total = np.empty([1000, 50], dtype=np.complex)
h_total[::] = scipy.io.loadmat(os.path.join(path, f'channel_50.mat'))['h_need']

print(np.shape(h_total))

# set parameters
Order = 10  # AR model order
Seq_length = 1000  # channel sequence length + AR order
pilot_length = 3  # length of pilot signal
t = np.arange(0, 1, 1 / Seq_length)
SNR = [-10, 0, 10, 20]  # SNR in dB

mse = np.zeros_like(SNR)
mse2 = np.zeros_like(SNR)
mse3 = np.zeros_like(SNR)
channel_num = 50

ACF = np.zeros(Order + 1, dtype=np.complex)
ACF1 = np.zeros(Order + 1, dtype=np.complex)
ACF2 = np.zeros(Order + 1, dtype=np.complex)
ACF3 = np.zeros(Order + 1, dtype=np.complex)
ACF4 = np.zeros(Order + 1, dtype=np.complex)

for i_avg in range(channel_num):
    h = h_total[:, i_avg]
    for i_auto in range(Order + 1):
        for j_it in range(Seq_length - i_auto):
            ACF[i_auto] += h[j_it + i_auto] * np.conj(h[j_it])
        # ACF[i_auto] = ACF[i_auto] / (Seq_length - i_auto)

    # real case ACF:
    h1 = np.real(h)
    h2 = np.imag(h)

    for i_auto in range(Order + 1):
        for j_it in range(Seq_length - i_auto):
            ACF1[i_auto] += h1[j_it + i_auto] * h1[j_it]
        # ACF1[i_auto] = ACF1[i_auto] / (Seq_length - i_auto)

    for i_auto in range(Order + 1):
        for j_it in range(Seq_length - i_auto):
            ACF2[i_auto] += h1[j_it + i_auto] * h2[j_it]
        # ACF2[i_auto] = ACF2[i_auto] / (Seq_length - i_auto)

    for i_auto in range(Order + 1):
        for j_it in range(Seq_length - i_auto):
            ACF3[i_auto] += h2[j_it + i_auto] * h1[j_it]
        # ACF3[i_auto] = ACF3[i_auto] / (Seq_length - i_auto)

    for i_auto in range(Order + 1):
        for j_it in range(Seq_length - i_auto):
            ACF4[i_auto] += h2[j_it + i_auto] * h2[j_it]
        # ACF4[i_auto] = ACF4[i_auto] / (Seq_length - i_auto)

for i_auto in range(Order + 1):
    ACF[i_auto] = ACF[i_auto] / (channel_num * (Seq_length - i_auto))
    ACF1[i_auto] = ACF1[i_auto] / (channel_num * (Seq_length - i_auto))
    ACF2[i_auto] = ACF2[i_auto] / (channel_num * (Seq_length - i_auto))
    ACF3[i_auto] = ACF3[i_auto] / (channel_num * (Seq_length - i_auto))
    ACF4[i_auto] = ACF4[i_auto] / (channel_num * (Seq_length - i_auto))

# generate AR parameter d from Yule-Wlaker equation
v = np.zeros(Order, dtype=np.complex)
R = np.zeros((Order, Order), dtype=np.complex)

for i in range(Order):
    v[i] = ACF[i + 1]  # Bessel function
    for j in range(Order):
        if i < j:
            R[i][j] = np.conj(ACF[j - i])
        else:
            R[i][j] = ACF[i - j]

# print(np.linalg.det(R))
R = R + 1e-10 * np.eye(Order)

d = -1 * np.dot(np.linalg.inv(R), v)  # from Yule-Walker equation
sigma_w_2 = np.real(ACF[0] + np.dot(d, np.conj(v)))  # R_0 variance of w_t

R1 = np.zeros((Order, Order), dtype=np.complex)
for i in range(Order):
    for j in range(Order):
        if i < j:
            R1[i][j] = np.conj(ACF1[j - i])
        else:
            R1[i][j] = ACF1[i - j]

R1 = R1 + 1e-10 * np.eye(Order)

R2 = np.zeros((Order, Order), dtype=np.complex)
for i in range(Order):
    for j in range(Order):
        if i < j:
            R2[i][j] = np.conj(ACF2[j - i])
        else:
            R2[i][j] = ACF2[i - j]

R2 = R2 + 1e-10 * np.eye(Order)

R3 = np.zeros((Order, Order), dtype=np.complex)
for i in range(Order):
    for j in range(Order):
        if i < j:
            R3[i][j] = np.conj(ACF3[j - i])
        else:
            R3[i][j] = ACF2[i - j]

R3 = R3 + 1e-10 * np.eye(Order)

R4 = np.zeros((Order, Order), dtype=np.complex)
for i in range(Order):
    for j in range(Order):
        if i < j:
            R4[i][j] = np.conj(ACF4[j - i])
        else:
            R4[i][j] = ACF4[i - j]

R4 = R4 + 1e-10 * np.eye(Order)

print("sigma_w_2:", sigma_w_2)

A = np.empty([50 * (Seq_length - Order), Order], dtype=np.complex)
for a in range(50):
    for i in range(Seq_length - Order):
        for j in range(Order):
            A[a * (Seq_length - Order) + i, j] = h_total[i + j, a]

b = np.empty(50 * (Seq_length - Order), dtype=np.complex)
for a in range(50):
    for i in range(Seq_length - Order):
        b[a * (Seq_length - Order) + i] = h_total[i + Order, a]

d2 = np.linalg.inv(np.dot(np.transpose(np.conj(A)), A))
d2 = np.dot(d2, np.transpose(np.conj(A)))
d2 = np.dot(d2, b)

print(np.shape(A))
print(np.shape(d2))
print(d2)

diff = np.dot(A, d2) - b
diff_real = np.real(diff)
diff_imag = np.imag(diff)

var_real = np.var(diff_real)
var_imag = np.var(diff_imag)
var = 0.5 * (var_real + var_imag)

P_h = np.real(np.dot(np.transpose(np.conj(np.reshape(h_total, (1000 * 50), order='F'))), np.reshape(h_total, (1000 * 50), order='F')) / (Seq_length * 50))
print(P_h)

# generate modal transition matrix F
F = np.zeros((Order, Order), dtype=np.complex)
for i in range(Order - 1):
    for j in range(i, Order):
        if j == (i + 1):
            F[i][j] = 1
        else:
            pass

for i in range(Order):
    F[Order - 1][i] = -1 * d[Order - 1 - i]

# convert F into real model
F_real = np.zeros((2 * Order, 2 * Order))
F_real[0:Order, 0:Order] = np.real(F)
F_real[0:Order, Order:] = -1 * np.imag(F)
F_real[Order:, 0:Order] = np.imag(F)
F_real[Order:, Order:] = np.real(F)
F_real_ten = torch.from_numpy(F_real).float()

# generate modal transition matrix F
F_lg = np.zeros((Order, Order), dtype=np.complex)
for i in range(Order - 1):
    for j in range(i, Order):
        if j == (i + 1):
            F_lg[i][j] = 1
        else:
            pass

for i in range(Order):
    F_lg[Order - 1][i] = 1 * d2[i]

# convert F into real model
F_lg_real = np.zeros((2 * Order, 2 * Order))
F_lg_real[0:Order, 0:Order] = np.real(F_lg)
F_lg_real[0:Order, Order:] = -1 * np.imag(F_lg)
F_lg_real[Order:, 0:Order] = np.imag(F_lg)
F_lg_real[Order:, Order:] = np.real(F_lg)
F_lg_real_ten = torch.from_numpy(F_lg_real).float()

# generate Nosie Corvariance of e
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

for i_SNR in range(0, len(SNR)):
    LOSS1 = 0.0
    LOSS2 = 0.0
    LOSS3 = 0.0
    loss_fn = nn.MSELoss(reduction='mean')

    # generate Nosie Corvariance of v
    V = 10 ** (-SNR[i_SNR] / 10) * np.eye(pilot_length)
    V_real = 0.5 * float(P_h) * 10 ** (-SNR[i_SNR] / 10) * np.eye(2 * pilot_length)

    for index in range(50):
        h = h_total[:, index]
        # h = h.numpy()

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
                np.sqrt(float(P_h)) * (10 ** (-SNR[i_SNR] / 20)) * (1 / np.sqrt(2)) * (np.random.randn(pilot_length) + 1j * np.random.randn(pilot_length)))
        # print(y)
        y_real = np.zeros((2 * pilot_length, Seq_length - Order))
        y_real[0:pilot_length, :] = np.real(y)
        y_real[pilot_length:, :] = np.imag(y)


        m1x_posterior = np.transpose(np.zeros(2 * Order))
        m2x_posterior = np.zeros((2 * Order, 2 * Order))
        m2x_posterior[0: Order, 0: Order] = R1
        m2x_posterior[0: Order, Order:] = R2
        m2x_posterior[Order:, 0: Order] = R3
        m2x_posterior[Order:, Order:] = R4

        x = np.zeros((Order, Seq_length - Order), dtype=np.cdouble)
        x_real = np.zeros((2 * Order, Seq_length - Order), dtype=np.double)
        x_lg_real = np.zeros((2 * Order, Seq_length - Order), dtype=np.double)
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

        m1x_posterior = np.transpose(np.zeros(2 * Order))
        m2x_posterior = np.zeros((2 * Order, 2 * Order))
        m2x_posterior[0: Order, 0: Order] = R1
        m2x_posterior[0: Order, Order:] = R2
        m2x_posterior[Order:, 0: Order] = R3
        m2x_posterior[Order:, Order:] = R4

        for i in range(Seq_length - Order):
            #print(i)
            m1x_prior = np.dot(F_lg_real, m1x_posterior)
            # print(m1x_prior)
            # m2x_prior = F * m2x_posterior * np.transpose(F) + E
            m2x_prior = np.dot(F_lg_real, m2x_posterior)
            m2x_prior = np.dot(m2x_prior, np.transpose(F_lg_real)) + E_real

            # KG = m2x_prior * np.transpose(np.conj(H)) * np.linalg.inv(H * m2x_prior * np.transpose(np.conj(H)) + V)
            KG = np.dot(H_real, m2x_prior)
            KG = np.dot(KG, np.transpose(H_real)) + V_real
            #print(KG)
            KG = np.dot(np.transpose(H_real), np.linalg.inv(KG))
            KG = np.dot(m2x_prior, KG)

            m1x_posterior = m1x_prior + np.dot(KG, (y_real[:, i] - np.dot(H_real, m1x_prior)))
            m2x_posterior = m2x_prior - np.dot(KG, np.dot(H_real, m2x_prior))

            for j in range(2 * Order):
                x_lg_real[j][i] = m1x_posterior[j]

        m1x_posterior = np.transpose(np.zeros(2 * Order))
        m2x_posterior = np.zeros((2 * Order, 2 * Order))
        m2x_posterior[0: Order, 0: Order] = R1
        m2x_posterior[0: Order, Order:] = R2
        m2x_posterior[Order:, 0: Order] = R3
        m2x_posterior[Order:, Order:] = R4

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

        x_real = torch.from_numpy(x_real).float()
        x_lg_real = torch.from_numpy(x_lg_real).float()
        x_real_noisy = torch.from_numpy(x_real_noisy).float()

        h_estimated = x_real[Order - 1, :] + 1j * x_real[2 * Order - 1, :]
        h_lg_estimated = x_lg_real[Order - 1, :] + 1j * x_lg_real[2 * Order - 1, :]
        h_noisy = x_real_noisy[Order - 1, :] + 1j * x_real_noisy[2 * Order - 1, :]
        # h_model = xm[Order - 1, :] + 1j * xm[2 * Order - 1, :]
        # h_model = h_model.numpy()

        h_mag = np.abs(h)  # take magnitude for the sake of plotting
        h_mag_dB = 10 * np.log10(h_mag)  # convert to dB
        h_est_mag = np.abs(h_estimated)  # take magnitude for the sake of plotting
        h_est_mag_dB = 10 * np.log10(h_est_mag)  # convert to dB
        h_est_lg_mag = np.abs(h_lg_estimated)  # take magnitude for the sake of plotting
        h_est_lg_mag_dB = 10 * np.log10(h_est_lg_mag)  # convert to dB
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
        LOSS3 += loss_fn(torch.cat((x_lg_real[Order - 1, :].unsqueeze(0), x_lg_real[2 * Order - 1, :].unsqueeze(0)), 1),
                                       h_gen_real).item()

        # mse = (np.abs(h_generated - h_estimated)**2).mean(axis='None')
        # mse = 10 * np.log10(np.mean(np.abs(h[Order:] - h_estimated) ** 2))

    mse[i_SNR] = 10 * np.log10(LOSS1 / (P_h * channel_num))
    mse2[i_SNR] = 10 * np.log10(LOSS2 / (P_h * channel_num))
    mse3[i_SNR] = 10 * np.log10(LOSS3 / (P_h * channel_num))
    print("MSE_noisy = %.3f dB" % mse2[i_SNR])
    print("MSE_KF_ACF = %.3f dB" % mse[i_SNR])
    print("MSE_KF_LR = %.3f dB" % mse3[i_SNR])

noise_floor = np.zeros(len(SNR))
for i in range(len(SNR)):
    noise_floor[i] = - SNR[i]

p1, = plt.plot(SNR, mse, 'g')
p2, = plt.plot(SNR, mse, 'g^')
# p3, = plt.plot(SNR, mse3, 'm')
# p4, = plt.plot(SNR, mse3, 'mo')
p5, = plt.plot(SNR, mse2, 'b')
p6, = plt.plot(SNR, mse2, 'bs')
p7, = plt.plot(SNR, noise_floor, 'r--')
plt.xlabel('SNR / dB')
plt.ylabel('NMSE / dB')
plt.legend([p2, p6, p7], ['Estimated with KF Yule-Walker / AR order 10', 'Estimated without model knowledge', 'Noise floor'])
# plt.legend(['Estimated without model knowledge', 'Estimated with KF / AR order 5'])
plt.show()
