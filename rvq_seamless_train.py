import pynvml
from utils import rotation_conversions as rc

import os
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import logging
import sys
from dataloaders import data_tools
from utils import config, logger_tools, other_tools, metric
import numpy as np

import warnings
warnings.filterwarnings('ignore')
from models.vq.model import RVQVAE

# 防止被当作配置文件解析
if __name__ != "__main__":
    print("This script should be executed directly, not imported.")
    print("Usage: python rvq_seamless_train.py [options]")
    sys.exit(1)

def get_logger(out_dir):
    logger = logging.getLogger('Exp')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_path = os.path.join(out_dir, "run.log")
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger

class ReConsLoss(nn.Module):
    def __init__(self, recons_loss):
        super(ReConsLoss, self).__init__()

        if recons_loss == 'l1':
            self.Loss = torch.nn.L1Loss()
        elif recons_loss == 'l2' :
            self.Loss = torch.nn.MSELoss()
        elif recons_loss == 'l1_smooth' :
            self.Loss = torch.nn.SmoothL1Loss()

    def my_forward(self,motion_pred,motion_gt,mask) :
        loss = self.Loss(motion_pred[..., mask], motion_gt[..., mask])
        return loss

    def velocity_loss(self, motion_pred, motion_gt, mask):
        """计算速度损失，惩罚相邻帧间的不连续变化"""
        if len(motion_pred.shape) < 3:
            return torch.tensor(0.0, device=motion_pred.device)

        # 计算预测速度和真实速度（时间差分）
        if motion_pred.shape[1] > 1:  # 确保序列长度足够计算差分
            pred_vel = torch.diff(motion_pred, dim=1)  # [batch, seq_len-1, dim]
            gt_vel = torch.diff(motion_gt, dim=1)      # [batch, seq_len-1, dim]

            # 应用掩码 (mask是list或array，需要转换为tensor)
            if isinstance(mask, (list, np.ndarray)):
                mask_tensor = torch.tensor(mask, device=motion_pred.device, dtype=torch.long)
                pred_vel = pred_vel[..., mask_tensor]
                gt_vel = gt_vel[..., mask_tensor]

            # 计算速度损失
            vel_loss = self.Loss(pred_vel, gt_vel)
        else:
            vel_loss = torch.tensor(0.0, device=motion_pred.device)

        return vel_loss

    def jitter_loss(self, motion_pred, motion_gt, mask):
        """计算jitter损失，直接优化抖动问题

        相比速度损失，jitter损失更直接地惩罚：
        1. 加速度的二阶差分（减少突兀的变化）
        2. 高频成分（减少快速抖动）
        3. 连续性约束（鼓励平滑过渡）
        """
        if len(motion_pred.shape) < 3:
            return torch.tensor(0.0, device=motion_pred.device)

        if motion_pred.shape[1] <= 2:  # 需要至少3帧计算二阶差分
            return torch.tensor(0.0, device=motion_pred.device)

        # 应用掩码
        if isinstance(mask, (list, np.ndarray)):
            mask_tensor = torch.tensor(mask, device=motion_pred.device, dtype=torch.long)
            motion_pred_masked = motion_pred[..., mask_tensor]
            motion_gt_masked = motion_gt[..., mask_tensor]
        else:
            motion_pred_masked = motion_pred
            motion_gt_masked = motion_gt

        # 1. 速度匹配损失 - 与GT速度保持一致
        pred_vel = torch.diff(motion_pred_masked, dim=1)  # [batch, seq_len-1, dim]
        gt_vel = torch.diff(motion_gt_masked, dim=1)      # [batch, seq_len-1, dim]
        vel_match = torch.mean((pred_vel - gt_vel)**2)

        # 2. 加速度匹配损失 - 与GT加速度保持一致
        pred_accel = torch.diff(motion_pred_masked, n=2, dim=1)  # [batch, seq_len-2, dim]
        gt_accel = torch.diff(motion_gt_masked, n=2, dim=1)      # [batch, seq_len-2, dim]
        accel_match = torch.mean((pred_accel - gt_accel)**2)

        # 综合jitter损失 - 速度和加速度都与真值比较
        jitter_loss = (
            0.6 * vel_match +        # 速度匹配 (40%)
            0.4 * accel_match      # 加速度匹配 (40%)
        )

        return jitter_loss

import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='RVQ-VAE training for Seamless dataset',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader
    parser.add_argument('--dataname', type=str, default='seamless', help='dataset directory')
    parser.add_argument('--batch-size', default=4096, type=int, help='batch size')
    parser.add_argument('--window-size', type=int, default=64, help='training motion length')
    parser.add_argument('--body_part',type=str,default='whole', help='body part: whole, upper, lower, hands')
    ## optimization
    parser.add_argument('--total-iter', default=50000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=1e-5, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[30000, 40000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")

    parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')
    parser.add_argument("--commit", type=float, default=0.2, help="hyper-parameter for the commitment loss")
    # parser.add_argument('--loss-vel', type=float, default=0.1, help='hyper-parameter for the velocity loss')
    # parser.add_argument('--use-jitter-loss', action='store_true', help='use jitter loss instead of velocity loss for better smoothness')
    parser.add_argument('--recons-loss', type=str, default='l1_smooth', help='reconstruction loss')

    ## vqvae arch
    parser.add_argument("--code-dim", type=int, default=128, help="embedding dimension")
    parser.add_argument("--nb-code", type=int, default=1024, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output-emb-width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq-act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')
    parser.add_argument('--vq-norm', type=str, default=None, help='dataset directory')

    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset'], help="eps for optimal transport")
    parser.add_argument('--beta', type=float, default=1.0, help='commitment loss in standard VQ')

    ## resume
    parser.add_argument("--resume-pth", type=str, default=None, help='resume pth for VQ')
    parser.add_argument("--resume-gpt", type=str, default=None, help='resume pth for GPT')


    ## output directory
    parser.add_argument('--out-dir', type=str, default='outputs/rvq_seamless', help='output directory')
    parser.add_argument('--results-dir', type=str, default='visual_results/', help='output directory')
    parser.add_argument('--visual-name', type=str, default='seamless_baseline', help='output directory')
    parser.add_argument('--exp-name', type=str, default='RVQVAE_Seamless', help='name of the experiment, will create a file inside out-dir')
    ## other
    parser.add_argument('--print-iter', default=100, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=1000, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=123, type=int, help='123 seed for initializing training.')

    parser.add_argument('--vis-gt', action='store_true', help='whether visualize GT motions')
    parser.add_argument('--nb-vis', default=20, type=int, help='nb of visualizations')
    parser.add_argument('--mode', type=str, default='train', help='train or eval')
    parser.add_argument('--rebuild-cache', action='store_true', help='force rebuild cache even if exists')
    parser.add_argument('--cache-only', action='store_true', help='only build cache, do not train')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID to use for training')


    return parser.parse_args()

def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):

    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

##### ---- Exp dirs ---- #####
args = get_args_parser()

# 设置GPU设备
device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(args.gpu_id)
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}_{args.body_part}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(f"使用GPU设备: {device}")
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

##### ---- Dataloader ---- #####
from dataloaders.seamless_sep import CustomDataset
from omegaconf import OmegaConf

# 直接使用OmegaConf加载配置，避免utils.config的参数解析问题
cfg = OmegaConf.load("configs/seamless_rvqvae_train.yaml")
dataset_args = cfg

# 设置缓存选项
if args.rebuild_cache:
    dataset_args.new_cache = True
    logger.info("🔄 强制重建缓存模式已启用")
else:
    dataset_args.new_cache = False

logger.info("="*60)
logger.info("开始构建Seamless数据集缓存...")
logger.info(f"数据集路径: {dataset_args.data_path}")
logger.info(f"缓存路径: {dataset_args.cache_path}")
logger.info(f"多长度训练: {dataset_args.multi_length_training}")
logger.info(f"强制重建缓存: {args.rebuild_cache}")
logger.info("="*60)

# 构建训练集缓存
logger.info("🔄 正在构建训练集缓存...")
trainSet = CustomDataset(dataset_args,"train",build_cache = True)
logger.info(f"✅ 训练集缓存构建完成！样本数量: {len(trainSet.selected_files)}")

# 构建测试集缓存
logger.info("🔄 正在构建测试集缓存...")
testSet = CustomDataset(dataset_args,"val",build_cache = True)
logger.info(f"✅ 测试集缓存构建完成！样本数量: {len(testSet.selected_files)}")

# 如果只是构建缓存，则直接退出
if args.cache_only:
    logger.info("🎯 缓存构建完成，程序退出（--cache-only 模式）")
    sys.exit(0)

logger.info("="*60)
logger.info("缓存构建完成，开始创建数据加载器...")
logger.info("="*60)

train_loader = torch.utils.data.DataLoader(trainSet,
                                              args.batch_size,
                                              shuffle=True,
                                              num_workers=8,
                                              drop_last = True)
test_loader = torch.utils.data.DataLoader(testSet,
                                          1,
                                            shuffle=False,
                                            num_workers=8,
                                            drop_last = True)

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

train_loader_iter = cycle(train_loader)
test_loader_iter = cycle(test_loader)

##### ---- Seamless Joint Masks ---- #####
if args.body_part == "upper":
    # 上半身：13个关节点
    joints = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    upper_body_mask = []
    for i in joints:
        upper_body_mask.extend([i*6, i*6+1, i*6+2, i*6+3, i*6+4, i*6+5])
    mask = upper_body_mask
    rec_mask = list(range(len(mask)))

elif args.body_part == "hands":
    # 手部：30个关节点 (22-51)
    joints = list(range(22, 52))
    hands_body_mask = []
    for i in joints:
        hands_body_mask.extend([i*6, i*6+1, i*6+2, i*6+3, i*6+4, i*6+5])
    mask = hands_body_mask
    rec_mask = list(range(len(mask)))

elif args.body_part == "lower":
    # 下半身：9个关节点
    joints = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    lower_body_mask = []
    for i in joints:
        lower_body_mask.extend([i*6, i*6+1, i*6+2, i*6+3, i*6+4, i*6+5])
    mask = lower_body_mask
    rec_mask = list(range(len(mask)))

elif args.body_part == "whole":
    # 全部52个关节点
    joints = list(range(52))
    whole_body_mask = []
    for i in joints:
        whole_body_mask.extend([i*6, i*6+1, i*6+2, i*6+3, i*6+4, i*6+5])
    mask = whole_body_mask
    rec_mask = list(range(len(mask)))

logger.info(f"Body part: {args.body_part}, Joint count: {len(joints)}, Mask dimensions: {len(mask)}")

##### ---- Model Configuration ---- #####
if args.body_part == "upper":
    dim_pose = len(joints) * 6   # 13 * 6 = 78
elif args.body_part == "hands":
    dim_pose = len(joints) * 6   # 30 * 6 = 180
elif args.body_part == "lower":
    dim_pose = len(joints) * 6   # 9 * 6 = 54
elif args.body_part == "whole":
    dim_pose = len(joints) * 6   # 52 * 6 = 312

args.num_quantizers = 6
args.shared_codebook = False
args.quantize_dropout_prob = 0.2

net = RVQVAE(args,
            dim_pose,
            args.nb_code,
            args.code_dim,
            args.code_dim,
            args.down_t,
            args.stride_t,
            args.width,
            args.depth,
            args.dilation_growth_rate,
            args.vq_act,
            args.vq_norm)

if args.resume_pth :
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)

##### ---- Seamless Normalization ---- #####
# 使用seamless数据集的归一化文件
mean_pose_path = './mean_std_seamless/seamless_2_312_mean.npy'
std_pose_path = './mean_std_seamless/seamless_2_312_std.npy'

mean_pose = np.load(mean_pose_path)
std_pose = np.load(std_pose_path)

# 裁剪到对应的身体部位
mean_pose = torch.from_numpy(mean_pose[mask]).to(device)
std_pose = torch.from_numpy(std_pose[mask]).to(device)

logger.info(f"Loaded normalization: mean_pose.shape={mean_pose.shape}, std_pose.shape={std_pose.shape}")

if args.mode == 'train':
    net.train()
    net.to(device)

    ##### ---- Optimizer & Scheduler ---- #####
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

    Loss = ReConsLoss(args.recons_loss)

    ##### ------ warm-up ------- #####
    avg_recons, avg_perplexity, avg_commit, avg_velocity = 0., 0., 0., 0.

    logger.info(f"Starting warm-up for {args.warm_up_iter} iterations...")

    for nb_iter in range(1, args.warm_up_iter):

        optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)

        gt_motion = next(train_loader_iter)

        gt_motion = gt_motion[...,mask].to(device).float() # (bs, 64, dim)

        pred_motion, loss_commit, perplexity = net(gt_motion).values()
        loss_motion = Loss.my_forward(pred_motion, gt_motion,rec_mask)
        # 选择损失函数类型 - 暂时注释掉jitter和velocity损失
        # if args.use_jitter_loss:
        #     loss_vel = Loss.jitter_loss(pred_motion, gt_motion, rec_mask)
        # else:
        #     loss_vel = Loss.velocity_loss(pred_motion, gt_motion, rec_mask)
        loss_vel = torch.tensor(0.0, device=loss_motion.device)

        loss = loss_motion + args.commit * loss_commit

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_recons += loss_motion.item()
        avg_perplexity += perplexity.item()
        avg_commit += loss_commit.item()
        avg_velocity += loss_vel.item()

        if nb_iter % args.print_iter ==  0 :
            avg_recons /= args.print_iter
            avg_perplexity /= args.print_iter
            avg_commit /= args.print_iter
            avg_velocity /= args.print_iter

            # loss_type_name = "Jitter" if args.use_jitter_loss else "Velocity" - 暂时注释掉jitter和velocity损失相关的日志
            logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.7f} 	 Commit. {avg_commit:.5f} 	 PPL. {avg_perplexity:.2f} 	 Recons.  {avg_recons:.5f}")

            avg_recons, avg_perplexity, avg_commit, avg_velocity = 0., 0., 0., 0.

    ##### ---- Training ---- #####
    avg_recons, avg_perplexity, avg_commit, avg_velocity = 0., 0., 0., 0.

    logger.info(f"Starting main training for {args.total_iter} iterations...")

    best_l2 = 10000
    l2_history = []
    early_stop_patience = 5
    early_stop_counter = 0

    for nb_iter in range(1, args.total_iter + 1):

        gt_motion = next(train_loader_iter)
        gt_motion = gt_motion[...,mask].to(device).float() # bs, nb_joints, joints_dim, seq_len

        pred_motion, loss_commit, perplexity = net(gt_motion).values()
        loss_motion = Loss.my_forward(pred_motion, gt_motion,rec_mask)
        # 选择损失函数类型 - 暂时注释掉jitter和velocity损失
        # if args.use_jitter_loss:
        #     loss_vel = Loss.jitter_loss(pred_motion, gt_motion, rec_mask)
        # else:
        #     loss_vel = Loss.velocity_loss(pred_motion, gt_motion, rec_mask)
        loss_vel = torch.tensor(0.0, device=loss_motion.device)

        loss = loss_motion + args.commit * loss_commit

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        avg_recons += loss_motion.item()
        avg_perplexity += perplexity.item()
        avg_commit += loss_commit.item()
        avg_velocity += loss_vel.item()

        if nb_iter % args.print_iter ==  0 :
            avg_recons /= args.print_iter
            avg_perplexity /= args.print_iter
            avg_commit /= args.print_iter
            avg_velocity /= args.print_iter

            writer.add_scalar('./Train/L1', avg_recons, nb_iter)
            writer.add_scalar('./Train/PPL', avg_perplexity, nb_iter)
            writer.add_scalar('./Train/Commit', avg_commit, nb_iter)

            # loss_type_name = "Jitter" if args.use_jitter_loss else "Velocity" - 暂时注释掉jitter和velocity损失相关的日志
            logger.info(f"Train. Iter {nb_iter} : 	 Commit. {avg_commit:.5f} 	 PPL. {avg_perplexity:.2f} 	 Recons.  {avg_recons:.5f}")

            avg_recons, avg_perplexity, avg_commit, avg_velocity = 0., 0., 0., 0.,

        # 简化评估 - 只使用L2距离
        if nb_iter % args.eval_iter==0 :
            torch.save({'net' : net.state_dict()}, os.path.join(args.out_dir, f'net_{nb_iter}.pth'))

            net.eval()

            total_length = 0
            l2_all = 0
            lvel = 0

            logger.info(f"Evaluating at iteration {nb_iter}...")

            with torch.no_grad():
                for its, batch_data in enumerate(test_loader):
                    gt_motion = batch_data.to(device).float()
                    n = gt_motion.shape[1]
                    remain = n%8

                    if remain != 0:
                        gt_motion = gt_motion[:, :-remain, :]

                    gt_ori = gt_motion
                    gt_motion = gt_motion[...,mask] # (bs, 64, dim)
                    bs = gt_motion.shape[0]
                    pred_motion, loss_commit, perplexity = net(gt_motion).values()
                    diff = pred_motion - gt_motion

                    pred_motion = pred_motion
                    rec_motion = gt_ori.clone()

                    rec_motion[..., mask] = pred_motion # it is already a 6d tensor

                    n = rec_motion.shape[1]

                    # 简化重建 - 只应用反归一化到掩码部分
                    rec_motion_part = rec_motion[..., mask] * std_pose + mean_pose
                    gt_motion_part = gt_ori[..., mask] * std_pose + mean_pose

                    # 计算L2距离
                    l2_batch = torch.sqrt(torch.sum(diff ** 2, dim=[1, 2])).mean().item()
                    l2_all += l2_batch

                avg_l2 = l2_all / len(test_loader)
                logger.info(f"Iteration {nb_iter}: L2 distance: {avg_l2:.6f}")

                # Add current L2 to history
                l2_history.append(avg_l2)

                # Check for early stopping
                if len(l2_history) > early_stop_patience:
                    if best_l2 <= min(l2_history[-early_stop_patience:]):
                        early_stop_counter += 1
                        logger.info(f"Early stopping counter: {early_stop_counter}/{early_stop_patience}")
                        if early_stop_counter >= early_stop_patience:
                            logger.info(f"Early stopping triggered at iteration {nb_iter}")
                            break
                    else:
                        early_stop_counter = 0

                if avg_l2 < best_l2:
                    best_l2 = avg_l2
                    torch.save({'net' : net.state_dict()}, os.path.join(args.out_dir, 'net_best.pth'))
                    logger.info(f"New best L2: {best_l2:.6f} - saved best model")

                # 记录L2历史
                import matplotlib.pyplot as plt
                plt.figure()
                iterations = list(range(len(l2_history)))
                iterations = [x * args.eval_iter for x in iterations]
                plt.plot(iterations, l2_history, label='L2 Distance')
                plt.xlabel('Iteration')
                plt.ylabel('L2 Distance')
                plt.title('L2 Distance over Iterations')
                plt.legend()
                plt.savefig(os.path.join(args.out_dir, 'l2_plot.png'))
                plt.close()

            net.train()

    logger.info("Training completed!")
    logger.info(f"Best L2 distance achieved: {best_l2:.6f}")

else:
    logger.info("Evaluation mode - loading model...")
    net.eval()
    net.to(device)

    if args.resume_pth:
        logger.info('loading checkpoint from {}'.format(args.resume_pth))
        ckpt = torch.load(args.resume_pth, map_location='cpu')
        net.load_state_dict(ckpt['net'], strict=True)

    total_length = 0
    l2_all = 0
    lvel = 0

    logger.info("Running evaluation...")

    with torch.no_grad():
        for its, batch_data in enumerate(test_loader):
            gt_motion = batch_data.to(device).float()
            n = gt_motion.shape[1]
            remain = n%8

            if remain != 0:
                gt_motion = gt_motion[:, :-remain, :]

            gt_ori = gt_motion
            gt_motion = gt_motion[...,mask] # (bs, 64, dim)
            pred_motion, loss_commit, perplexity = net(gt_motion).values()
            diff = pred_motion - gt_motion

            pred_motion = pred_motion
            rec_motion = gt_ori.clone()

            rec_motion[..., mask] = pred_motion # it is already a 6d tensor

            n = rec_motion.shape[1]

            # 简化重建 - 只应用反归一化到掩码部分
            rec_motion_part = rec_motion[..., mask] * std_pose + mean_pose
            gt_motion_part = gt_ori[..., mask] * std_pose + mean_pose

            l2_batch = torch.sqrt(torch.sum(diff ** 2, dim=[1, 2])).mean().item()
            l2_all += l2_batch

        avg_l2 = l2_all / len(test_loader)
        logger.info(f"Evaluation L2 distance: {avg_l2:.6f}")

logger.info("Script completed successfully!")