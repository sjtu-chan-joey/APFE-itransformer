import argparse
parser = argparse.ArgumentParser(description='APEC_iTransformer')
#模型配置

parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]') #任务名称
parser.add_argument('--model', type=str, default='Autoformer',
                        help='model') #任务名称
# forecasting task
parser.add_argument('--seq_len', type=int, default=32, help='input sequence length') #序列长度 用前seq_len的数据预测
parser.add_argument('--label_len', type=int, default=16, help='start token length') #目标长度 希望模型预测的未来时间步长的数量
parser.add_argument('--pred_len', type=int, default=32, help='prediction sequence length') #预测长度 希望模型预测未来 pred_len 天的数据
parser.add_argument('--freq', type=str, default='d',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
#freq 用于时间特征编码，选项：[s:秒，t:分钟，h:小时，d:日，b:工作日，w:周，m:月]，您也可以使用更详细的 freq，如 15 分钟或 3 小时

parser.add_argument('--output_attention',default=False, action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--factor', type=int, default=2, help='attn factor')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--activation', type=str, default='gelu', help='activation')

parser.add_argument('--down_sampling_layers', type=int, default=1, help='num of down sampling layers')
parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
parser.add_argument('--down_sampling_method', type=str, default='avg',
                    help='down sampling method, only support avg, max, conv')
parser.add_argument('--channel_independence', type=int, default=1,
                    help='0: channel dependence 1: channel independence for FreTS model')
parser.add_argument('--decomp_method', type=str, default='moving_avg',
                    help='method of series decompsition, only support moving_avg or dft_decomp')
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')

#参数设置
parser.add_argument('--train_epochs', type=int, default=30)  #训练轮数
parser.add_argument('--learning_rate', type=float, default=1e-4)  #学习率
parser.add_argument('--scale', type=bool, default=True,help='Whether to perform feature scaling')  #是否进行特征缩放
parser.add_argument('--batch_size', type=int, default=4)  #批量
parser.add_argument('--num_workers', type=int, default=0)  #
parser.add_argument('--patience', type=int, default=10)  #提前终止轮数
parser.add_argument('--itr', type=int, default=1)  #实验次数
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# model define
parser.add_argument('--top_k', type=int, default=2, help='for TimesBlock')  #取傅里叶相值的前 top_k数据
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception') #Inception模型的参数
parser.add_argument('--enc_in', type=int, default=32, help='encoder input size') #编码器输入尺寸 模型输入的特征数(维度)
parser.add_argument('--dec_in', type=int, default=32, help='decoder input size') #解码器输入尺寸
parser.add_argument('--c_out', type=int, default=32, help='output size') #输出尺寸
parser.add_argument('--d_model', type=int, default=32, help='dimension of model') #模型维度
parser.add_argument('--e_layers', type=int, default=6, help='num of encoder layers') #编码层个数
parser.add_argument('--d_layers', type=int, default=6, help='num of decoder layers') #解码层个数
parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn') # FCN 网络中的隐藏层神经元的数量
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]') #时间特征编码, 选项:[timeF, 固定, 学习]


args = parser.parse_args()
