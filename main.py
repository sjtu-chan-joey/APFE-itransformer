from model.AFPE_iTransformer import Model
from configs import *
from Warehouse import *

# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':

    batch_x = torch.randn(args.batch_size,args.seq_len,args.enc_in).to(device)
    batch_y = torch.randn(args.batch_size,args.label_len + args.pred_len,args.enc_in).to(device)
    model = Model(args).float().to(device)
    output=model(batch_x, batch_y)
    print(output.shape)