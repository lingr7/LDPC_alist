"""
================================================
Coding - Decoding simulation of a random message
================================================

This example shows a simulation of the transmission of a binary message
through a gaussian white noise channel with an LDPC coding and decoding system.
"""

import datetime
import numpy as np
from code_ldpc import make_ldpc_tmp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from decoder import decode

def encode_LDPC(x_bits, G):#要调用这个方法，根据生成矩阵再对2取模，导致经过编码后序列里只有0或者1
        u_coded_bits = np.mod(np.matmul(x_bits, G), 2)#这里用的编码方法是用生成矩阵编码
#        check = np.mod(np.matmul(u_coded_bits, np.transpose(self.H_matrix)),2)
        return u_coded_bits
        
def BPSK(u_coded_bits):
    #s_mod = u_coded_bits * (-2) + 1#这个将序列转换成2种相位，要么正，要么负啊。
    s_mod = (-1) ** u_coded_bits #理论上是一样的效果
    return s_mod

        
n = 128
k = 64
d_v = 1
d_c = 2
channel = 'AWGN'
decoding_data_folder = "./DecodingData/"
results_folder = "./results/"
function = 'BP'
#dothing = 'figure'
dothing = 'simu'
seed = np.random.RandomState(42)
##################################################################
# First we create an LDPC code i.e a pair of decoding and coding matrices
# H and G. H is a regular parity-check matrix with d_v ones per row
# and d_c ones per column

#H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
H, G = make_ldpc_tmp()

print(H.shape)
print(G.shape)
k, n = G.shape
print("Number of coded bits:", k)

##################################################################
# Now we simulate transmission for different levels of noise and
# compute the percentage of errors using the bit-error-rate score
# The coding and decoding can be done in parallel by column stacking.

errors = []
#snrs = np.array([1, 1.5, 2.0, 2.5, 3, 3.5], np.float32)#np.linspace(-2, 10, 20)
#snrs = np.array([1, 1.5], np.float32)
#snrs = np.array([1.0,1.5,2.0,2.5,3.0], np.float32)
#snrs = np.array([0, 0.2, 0.4, 0.6, 0.65, 0.7, 0.75, 0.8], np.float32)
#snrs = np.array([2.5, 2.6, 2.7, 2.8, 2.9, 3.0], np.float32)
snrs = np.array([2.0,2.5,3.0,3.5,4.0], np.float32)
#[0, 0.2, 0.4, 0.6, 0.65, 0.7, 0.75, 0.8]
batch_size = 1000
total_samples = 4e6#4e5不行，4e8又太多了。4e6是个比较好的办法。有什么办法加速计算？比特数大了，多了，就会多花时间，40分钟的仿真是可以考虑的。量化
target_err_bits_num = 1000#spa现在不提了。我直接考虑加上量化吧。
target_err_fers_num = 50
BP_iter_num = np.array([10])
rng = np.random.RandomState(None)
total_batches = int(total_samples // (batch_size*k))

bp_str = np.array2string(BP_iter_num, separator='_', formatter={'int': lambda d: "%d" % d})
bp_str = bp_str[1:(len(bp_str) - 1)]
ber_file = format('%sBER(%d_%d)_BP(%s)' % (results_folder, n, k, bp_str))
ber_file = format('%s_%s' % (ber_file, function))
ber_file = format('%s.txt' % ber_file)
if (dothing is 'simu'):
    fout_ber = open(ber_file, 'wt')   
    start = datetime.datetime.now()
    for snr in snrs:
        real_batch_size = batch_size
        actual_simutimes = 0
        bit_errs_iter = np.zeros(1, dtype=np.int32)
        fer_errs_iter = np.zeros(1, dtype=np.int32)
        fer_cdc_iter= np.zeros(1, dtype=np.int32)
        for ik in range(0, total_batches):
            x_bits = np.random.randint(0, 2, size=(batch_size, k))#5000*64 0,1序列
            u_coded_bits = encode_LDPC(x_bits,G)#LDPC编码 0,1序列
            s_mod = BPSK(u_coded_bits) #1，-1序列
            noise_awgn = rng.randn(batch_size, n)
            ch_noise_normalize = noise_awgn.astype(np.float64)
            ch_noise_sigma = np.sqrt(1 / np.power(10, snr / 10.0) / 2.0)
            ch_noise = ch_noise_normalize * ch_noise_sigma
            y_receive = s_mod + ch_noise#5000*128 这个译码算法好像也没有考虑解调啊。。。
            u_BP_decoded = decode(H, y_receive.T, snr, maxiter=BP_iter_num[0]).T#得到一个batch_size*n的矩阵 这里得到的是128*128根本不对啊。所谓并行就是大矩阵传入速度快，避免for循环。
            #u_BP_decoded = spa(H, y_receive.T, snr, maxiter=BP_iter_num[0])
            output_x =  u_BP_decoded[:, 0:k]#128*64 这里的译码结果是根据y_receive解出的x=(-1)的v次方，那么，这里的结果应该是一个元素是1 和 -1的序列才对
            bit_errs_iter[0] += np.sum(output_x != x_bits)
            fer_errs_iter[0] += np.sum(np.sign(np.sum(output_x != x_bits, axis=1)))#误帧率正确。
            actual_simutimes += real_batch_size
            if bit_errs_iter[0] >= target_err_bits_num :#现在的情况是误码数太少，根本不足以触发这个提前结束仿真的条件
                    break
            if fer_errs_iter[0] >= target_err_fers_num :#现在的情况是误码数太少，根本不足以触发这个提前结束仿真的条件
                    break
        print('%d bits are simulated!' % (actual_simutimes * k))# K是信息位数，信息位与实际仿真次数就是所有仿真位数。
        # load to files
        ber_iter = np.zeros(1, dtype=np.float64)
        fer_iter = np.zeros(1, dtype=np.float64)
        #ber
        fout_ber.write(str(snr) + '\t')
        ber_iter[0] = bit_errs_iter[0] / float(k * actual_simutimes)
        fer_iter[0] = fer_errs_iter[0] / float(actual_simutimes)
        fout_ber.write(str(ber_iter[0])  + '\t')
        fout_ber.write(str(fer_iter[0]) + '\t')
        fout_ber.write('\n')
#simulation finished
    fout_ber.close()#这仿真的结果只有两个信噪比下3.5 3.25
        #y_receive = y_receive.astype(np.float32)
        #y_receive.tofile(fout_yrecieve)
        #x_bits = x_bits.astype(np.float32)
        #x_bits.tofile(fout_xtransmit)
        
    end = datetime.datetime.now()
    print('Time: %ds' % (end-start).seconds)#4973s这个程序验证一下真的挺花时间的。我感觉是用了太多的tofile 造成效率很低。2020-05-03 9:56:42再跑一次，用了7363s为什么这么慢。
    print("end\n")#这里应该都是不涉及tf的。   
        
else :
    data = np.loadtxt(ber_file, dtype=np.float64)
    plt.figure()
    #ax.semilogy(data[:,0], data[:,1])
    plt.plot(data[:,0], data[:,1], color="indianred")
    plt.ylabel("Bit error rate")
    plt.xlabel("SNR")
    plt.show()
    
    fig, ax = plt.subplots(figsize=(5, 4))
    plt.rcParams['savefig.dpi'] = 1000       # 图片像素
    plt.rcParams['figure.dpi'] = 1000        # 分辨率
    ax.semilogy(data[:,0], data[:,1], label='decode', color='r', linestyle='solid', linewidth=2)
   # ax.semilogx(x_data, y_data_2, label='decay 2', color='g', linestyle='--', linewidth=2)
    ax.set_xlabel('SNR (dB)', fontsize=13)
    ax.set_ylabel('Bit Error Rate', fontsize=13)
    ax.tick_params(labelsize=13)            # 刻度字体大小
    fig.legend(loc="upper right", fontsize=16, bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes, ncol=1,
               columnspacing=0.1, labelspacing=0.2, markerscale=1, shadow=True, borderpad=0.2, handletextpad=0.2)
    fig.set_tight_layout(tight='rect')
    # plt.savefig("exponential_decay.png")
    plt.show()