import numpy as np
import time
from matplotlib import pyplot as plt
import os, cv2
import sys
# pytorch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import argparse
import random
from dataset_processing.batch_accessing import data_provider
from skimage.metrics import structural_similarity as compare_ssim
from VAE_model import VanillaVAE
import lpips
from models import *
loss_fn_alex = lpips.LPIPS(net='alex')
import sys
import gc
from tqdm import tqdm
torch.cuda.empty_cache()
gc.collect()
import torchvision.models as models
import torch
# from torchmetrics.image import StructuralSimilarityIndexMeasure
from pytorch_msssim import ssim
torch.autograd.set_detect_anomaly(True)
MSE = nn.MSELoss()
CE = nn.CrossEntropyLoss()
BCE= nn.BCELoss()

seed = 0
random.seed(seed)

# Combo: BCEWithLogits + Dice Loss
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

def combo_loss(pred, target):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dsc = dice_loss(pred, target)
    return 0.5 * bce + 0.5 * dsc


class SSTA_Net(nn.Module):
    def __init__(self, input_dim, h_units, act, args):
        super(SSTA_Net, self).__init__()
        # [10, 128, 128, 5]
        self.filter_size = args.filter_size
        self.padding = self.filter_size // 2

        self.frame_predictor = DeterministicConvLSTM(input_dim, h_units[-1], h_units[0], len(h_units), args)
        self.l3 = nn.Conv3d(h_units[-1], args.ssta_output_channels, kernel_size=self.filter_size, stride=1, padding=self.padding, bias=False)

        if act == "relu":
            self.act = F.relu
        elif act == "sig":
            self.act = F.sigmoid
        elif act == "tanh":
            self.act= F.tanh

    def __call__(self, x, m_t, m_t_others, memory):
        pred_x_tp1, message, memory\
            = self.forward(x, m_t, m_t_others, memory)
        return pred_x_tp1, message, memory

    def forward(self, x_t, m_t, m_t_others, frame_predictor_hidden):
        # print(frame_predictor_hidden)
        x = torch.cat([x_t, m_t, *m_t_others] , -1)
        x = x.permute(0, 4, 1, 2, 3)
        h, frame_predictor_hidden = self.frame_predictor(x, frame_predictor_hidden) 
        pred_x_tp1 = self.l3(h)
        message = m_t
        message = None
        pred_x_tp1 = pred_x_tp1.permute(0, 2, 3, 4, 1)
        B, T, H, W, C = pred_x_tp1.shape  # C should be 200
        if args.loss_fn=="ce":
            pred_x_tp1 = pred_x_tp1.view(B, T, H, W, 2, C // 2 )
        pred_x_tp1 = self.act(pred_x_tp1)
        return pred_x_tp1, message, frame_predictor_hidden

    def predict(self, x, m_t, m_t_others, memory):
        pred_x_tp1, message, memory\
            = self.forward(x, m_t, m_t_others, memory)
        return pred_x_tp1.data, message.data, memory
    
def run_steps(x_batch, models, optimizers, connections, vae, inference = True, args = None):
    

    memory = [None for _ in range(args.num_views)]

    x_t = torch.split(x_batch, x_batch.shape[-1] // args.num_views, dim=-1)

    # print(x_t[0].shape)


    # for i in range(14):
    #     # view_1 = x_t[0].squeeze().detach().cpu().numpy()[0, i, :, :,0:3]
    #     # view_2 = x_t[1].squeeze().detach().cpu().numpy()[0, i, :,:, 0:3]

    #     view_1 = x_batch.squeeze().detach().cpu().numpy()[0, i, :, :,0:3]
    #     view_2 = x_batch.squeeze().detach().cpu().numpy()[0, i, :,:, 5:8]

    #     # view1-

    # #     # Convert from float (0-1) to uint8 (0-255) if needed
    # #     if view_1.max() <= 1.0:
    # #         view_1 = np.uint8(view_1 * 255)
    # #         view_2 = np.uint8(view_2 * 255)

    # #     # Make sure shape is (H, W, 3)
    # #     if view_1.shape[0] == 3:
    # #         view_1 = np.transpose(view_1, (1, 2, 0))
    # #         view_2 = np.transpose(view_2, (1, 2, 0))

    # #     # Concatenate horizontally
    #     print(view_1.shape,view_2.shape)
    #     views = np.concatenate([view_1, view_2], axis=1)

    #     # Show the image
    #     cv2.imshow('view', views)
    #     cv2.waitKey(100)

    # x_0_t, x_1_t = torch.split(x_batch, x_batch.shape[-1] // args.num_views, dim=-1)
    pred_batch_list = [[] for _ in range(args.num_views)]
    train_list = [[] for _ in range(args.num_views)]
    message_list = [[] for _ in range(args.num_views)]


    message_0 = x_t[0][:, 0:0 + 1,:,:, 0:3]
    # print(message_0.shape)

    messages = {}
    if args.message_type == 'raw_data':
        for view, ssta_name in enumerate(models.keys()):
            messages[ssta_name] = x_t[view][:, 0:0 + 1,:,:, 0:3]

    elif args.message_type == 'vae':
        with torch.no_grad():
            for view, ssta_name in enumerate(models.keys()):
                messages[ssta_name] = vae.get_message(x_t[view][:, 0:0 + 1,:,:, 0:3].detach())

    else:
        for view, ssta_name in enumerate(models.keys()):
            messages[ssta_name] = torch.zeros((x_batch.shape[0], 1, x_batch.shape[2], x_batch.shape[3], 1)).to(args.device)



    #above messages
    #############need to revamp this whole section
    if args.eval_mode == 'multi_step_eval' and inference == True:

        x_t_prev_preds = []
        for view in range(args.num_views):
            x_t_prev_preds.append(x_t[view][:, 0:0 + 1,:,:, 0:3])
 
        use_gt_flag = False
        for t in range(args.train_sequence - 1):
           

            for view, (ssta_key,model) in enumerate(models.items()):
                message_others = get_relevant_msgs(ssta_key, messages, connections)
                
                x_t_pred, messages[ssta_key], memory[view] = model(x_t_prev_preds[view], messages[ssta_key], message_others, memory[view])
                # print(x_t_pred.shape)

                
                if args.message_type in ['vae']:
                    with torch.no_grad():
                        if t < args.num_past or np.random.uniform(0, 1) > (1-1/args.mask_per_step):  # t % args.mask_per_step == 0:
                            messages[ssta_key] = vae.get_message(x_t[view][:, t:t + 1,:,:,0:3].detach())
                        else:
                            messages[ssta_key] = vae.get_message(x_t_prev_preds[view].detach())

                elif args.message_type in ['raw_data']:
                    messages[ssta_key] = x_t[view][:, t:t + 1,:,:,0:3]

                elif args.message_type == 'zeros':
                    messages[ssta_key] = torch.zeros_like(messages[ssta_key])

                elif args.message_type == 'randn':
                    messages[ssta_key] = torch.randn_like(messages[ssta_key])

                x_t_prev_preds[view] = x_t[view][:, t+1:t+2,:,:, 0:3]

                pred_batch_list[view].append(x_t_pred)
                message_list[view].append(messages[ssta_key])
                loss=0

        pred_batch_before = [torch.cat(first,1) for first in pred_batch_list]
        pred_batch = torch.cat(pred_batch_before, -1)

        message_list_before = [torch.cat(first,1) for first in message_list]
        message_batch = torch.cat(message_list_before, -1)

        
    else:

        loss=0.0
        x_t_prev_preds = []
        for view in range(args.num_views):
            x_t_prev_preds.append(x_t[view][:, 0:0 + 1,:,:, 0:3])
        
        for t in range(args.train_sequence-1):
            for view, (ssta_key,model) in enumerate(models.items()):
                model.train()
                optimizers[ssta_key].zero_grad()

                message_others = get_relevant_msgs(ssta_key, messages, connections)
                # print(x_t_prev_preds[view].shape, messages[ssta_key].shape, len(message_others),print(memory[view]))
                # print(messages)
                x_t_pred, _, memory_temp = model(x_t_prev_preds[view], messages[ssta_key], message_others, memory[view])
                # print("modelout",x_t_pred.shape)
                # print(messages)
                
                memory[view] = [(mem1.detach(), mem2.detach()) for mem1,mem2 in memory_temp]

                if args.message_type in ['vae']:
                    with torch.no_grad():
                        if t < args.num_past or np.random.uniform(0, 1) > (1-1/args.mask_per_step):  # t % args.mask_per_step == 0:
                            messages[ssta_key] = vae.get_message(x_t[view][:, t:t + 1,:,:,0:3].detach())
                            # message_0 = vae.get_message(x_t[:, t:t + 1])
                        else:
                            messages[ssta_key] = vae.get_message(x_t_prev_preds[view].detach())

                elif args.message_type in ['raw_data']:
                    messages[ssta_key] = x_t[view][:, t:t + 1,:,:,0:3].detach()

                elif args.message_type == 'zeros':
                    # print(messages)
                    # print(ssta_key)
                    # print(messages[ssta_key])
                    messages[ssta_key] = torch.zeros_like(messages[ssta_key])

                elif args.message_type == 'randn':
                    messages[ssta_key] = torch.randn_like(messages[ssta_key])

                
                gt_train =  x_t[view][:, t:t + 1,:,:, 3:]
                # print(x_t_pred.shape, x_t_prev_preds[view].shape, gt_train.shape)

                if args.loss_fn == "detr":

                    bce_loss, mse_t2no_loss, mse_t2nd_loss=0.0 ,0.0 , 0.0

                    gt_mask=(gt_train[...,0]!=1).float()
                    
                    # np.set_printoptions(threshold=np.inf)

                    # if view==0:

                    #     # print(gt_train[...,0])
                    #     # Example: save the 0th image in the batch
                    #     pred_img = x_t_pred[...,1][0, 0].detach().cpu().numpy()
                    #     gt_img = gt_train[...,1][0, 0].detach().cpu().numpy()

                    #     # print(pred_img)
                    #     # print("-----")
                    #     # print(gt_img)

                    #     # Optional: normalize if values aren't already in [0, 255]
                    #     pred_img = (pred_img * 255).astype('uint8')  # if in [0,1]
                    #     gt_img = (gt_img * 255).astype('uint8')

                    #     # Save with OpenCV
                    #     cv2.imwrite('pred_img.png', pred_img)
                    #     cv2.imwrite('gt_img.png', gt_img)
                    # sys.exit(0)


                    ##
                    bce_loss=F.binary_cross_entropy_with_logits(x_t_pred[...,2],gt_mask)

                    mask = gt_mask.bool() 

                    if mask.sum() !=0:
                        masked_t2no_pred, masked_t2nd_pred= x_t_pred[...,0][mask],x_t_pred[...,1][mask]

                        masked_gtno, masked_gtnd= gt_train[...,0][mask],gt_train[...,1][mask]

                        mse_t2no_loss = F.mse_loss(masked_t2no_pred,masked_gtno)

                        mse_t2nd_loss = F.mse_loss(masked_t2nd_pred,masked_gtnd)
                    else:
                        mse_t2no_loss = torch.tensor(0.0, device=x_t_pred.device)
 
                        mse_t2nd_loss = torch.tensor(0.0, device=x_t_pred.device)

                    loss = bce_loss + mse_t2no_loss + mse_t2nd_loss

                    gt_mask_expanded = gt_mask.unsqueeze(-1)
                    # Step 2: Concatenate along the last dimension
                    gt_train_combined = torch.cat([gt_train, gt_mask_expanded], dim=-1)



                if args.loss_fn == "ssim":
                    # Compute binary mask for BCE loss
                    gt_mask = (gt_train[..., 0] != 1).float()

                    # BCE loss on third channel (assuming logits)
                    # bce_loss = F.binary_cross_entropy_with_logits(x_t_pred[..., 2], gt_mask)
                    bce_loss = combo_loss(x_t_pred[...,2],gt_mask)

                    # Ensure tensors have channel dimension: [B, 1, H, W]
                    pred_t2no = x_t_pred[..., 0].unsqueeze(1)
                    gt_t2no   = gt_train[..., 0].unsqueeze(1)

                    pred_t2nd = x_t_pred[..., 1].unsqueeze(1)
                    gt_t2nd   = gt_train[..., 1].unsqueeze(1)

                    # SSIM losses (mean SSIM values)
                    ssim_loss_t2no = 1 - ssim(pred_t2no, gt_t2no, data_range=1.0, size_average=True)
                    ssim_loss_t2nd = 1 - ssim(pred_t2nd, gt_t2nd, data_range=1.0, size_average=True)

                    # Combine all losses
                    loss = bce_loss + ssim_loss_t2no + ssim_loss_t2nd

                    # print(f"BCE: {bce_loss.item():.4f}, SSIM T2NO: {ssim_loss_t2no.item():.4f}, SSIM T2ND: {ssim_loss_t2nd.item():.4f}")

                    gt_mask_expanded = gt_mask.unsqueeze(-1)
                    # Step 2: Concatenate along the last dimension
                    gt_train_combined = torch.cat([gt_train, gt_mask_expanded], dim=-1)







                if args.loss_fn == "mse":
                    loss = MSE(x_t_pred, gt_train)

                if args.loss_fn == "bce":
                    loss = BCE(x_t_pred, gt_train)

                if args.loss_fn == "ce":
                    pred_cngd = x_t_pred.squeeze(1)   # New shape: [3, 128, 128, 2, 100]
                    gt_cngd = gt_train.squeeze(1) 
                    gt_cngd = gt_cngd.long()

                    pred_cngd = pred_cngd.permute(0, 4, 1, 2, 3)  # New shape: [3, 100, 128, 128, 2]

                    loss = CE(pred_cngd, gt_cngd)
                

                loss.backward()
                optimizers[ssta_key].step()

                #softmax apply

                if args.loss_fn == "ce":
                    # Apply softmax along the last dimension to get probabilities
                    x_t_pred = torch.softmax(x_t_pred, dim=-1)
                    # Then, take argmax to obtain the predicted class indices
                    x_t_pred = x_t_pred.argmax(dim=-1)

                pred_batch_list[view].append(x_t_pred)
                train_list[view].append(gt_train_combined)
                message_list[view].append(messages[ssta_key])

                x_t_prev_preds[view] = x_t[view][:, t+1:t+2,:,:, 0:3]

        pred_batch_before = [torch.cat(first,1) for first in pred_batch_list]
        pred_batch = torch.cat(pred_batch_before, -1)

        message_list_before = [torch.cat(first,1) for first in message_list]
        message_batch = torch.cat(message_list_before, -1)

        train_list_before = [torch.cat(first,1) for first in train_list]
        train_batch = torch.cat(train_list_before, -1)

    return train_batch, pred_batch, message_batch , loss  



def training(n_epoch, act,args):

    loss_train = []
    loss_val = []
   
    #DATALOADER for training and test set
    train_input_handle, test_input_handle = data_provider(
        args.data_name, args.train_data_paths, args.valid_data_paths, args.bs, args.img_width,
        seq_length=args.train_sequence, is_training=True, num_views=args.num_views, img_channel=args.img_channel,
         eval_batch_size=args.vis_bs, n_epoch=n_epoch, args=args)
    
    if args.message_type in ['raw_data']:
        input_dim = 3 + 3 * args.num_views
    elif args.message_type in ['vae']:
        input_dim = 3 + (args.vae_latent_dim * args.num_views)
    else:
        input_dim = 3+1 * args.num_views

    h_units = [int(x) for x in args.num_hidden.split(',')]
    if (args.mode == 'eval' or args.mode == 'transfer_learning') and args.ckpt_dir is not None:
        models = {}
        
        paths = [os.path.join(args.ckpt_dir, "ssta_0.pt"), os.path.join(args.ckpt_dir, "ssta_1.pt")]

        models, optimizers, connections = load_sstas(args.num_views, paths)
        for i in range(len(paths)):    
            print('Loaded model_{} from {}'.format(i, paths[i]))
    else:
        models, optimizers, connections = create_sstas(args.num_views, input_dim, h_units, act, args)
            
    parameters = []
    for name, _ in models.items():
        models[name] = models[name].to(args.device)
        parameters += list(models[name].parameters())

    # vae = vae.to(args.device)
    # print('Loaded VAE model_0 from {}'.format(vae_path))
    # vae = VanillaVAE(input_dim, h_units, act, args)
    vae_path = os.path.join(args.vae_ckpt_dir, 'vae.pt')
    vae = torch.load(vae_path,weights_only=False)
    vae = vae.to(args.device)
    vae.eval()
    print('Loaded VAE model_0 from {}'.format(vae_path))

    # optimizer = optim.Adam(parameters,lr = 0.0001)

    # ssim_loss = pytorch_ssim.SSIM(window_size = 11)

    root_res_path = os.path.join(args.gen_frm_dir)
    os.makedirs(root_res_path, exist_ok=True)

    print("START")
    best_eval_loss = np.inf
    total_final_loss=[]
    continue_epoch=0
    
    if args.mode=="transfer_learning":
        continue_epoch=args.continue_epoch

    progress_bar_total=0
    if args.mode=="train" or args.mode=="transfer_learning":
        for epoch in range(1+continue_epoch, n_epoch + 1):
            print("-----------------------",epoch,"------------------")
            if args.mode == 'train' or args.mode=="transfer_learning":
                for name, _ in models.items():
                    models[name].train()
                
                sum_loss = 0
                N,iter=0,0
                loss=0.0
                print('Training ... {}'.format(epoch))
                train_input_handle.begin(do_shuffle=False)
                progress_bar = tqdm(total=333, desc='Epoch Completion')
                
                while (train_input_handle.no_batch_left() == False):
                    if epoch==1:progress_bar_total+=1
                     
                    ims = train_input_handle.get_batch()
                    # print(ims.shape)
                    train_input_handle.next()
                    x_batch = ims[:, :]


                    
                    gt_batch = ims[:, 1:]
                    x_batch = torch.from_numpy(x_batch.astype(np.float32)).to(args.device)  # .reshape(x.shape[0], 1))
                    gt_batch = torch.from_numpy(gt_batch.astype(np.float32)).to(args.device)  # .reshape(gt.shape[0], 1))

                    gt_channel_split= torch.split(gt_batch, gt_batch.shape[-1] // args.num_views, dim=-1)

                    gt_batch = torch.cat([t[..., -2:] for t in gt_channel_split], dim=-1)

                    train_batch, pred_batch, _ , loss = run_steps(x_batch, models, optimizers, connections, vae,
                                                        inference=False, args=args)
                    
                    sum_loss += loss.data * args.bs
                    
                    # N+=pred_batch.shape[1]* args.bs
                    N+=1
                    progress_bar.update(1)

                progress_bar.close()

                ave_loss = sum_loss / N 
                total_final_loss.append(ave_loss)
                loss_train.append(ave_loss)

                print("Total images computed with sequence:",N)
                print("averageloss ",epoch,":",ave_loss.data)


                pred_batch = pred_batch.detach().cpu().numpy()

                train_batch = train_batch.detach().cpu().numpy()

                # print("pred",pred_batch.shape,"gt",train_batch.shape)
                #                 # Example: extract sample index 1 from pred and gt
                # pred_sample = pred_batch[1]  # shape: (14, 128, 128, 6)
                # gt_sample = train_batch[1]      # shape: (14, 128, 128, 6)

                # # Create output directories
                # os.makedirs("pred_images", exist_ok=True)
                # os.makedirs("gt_images", exist_ok=True)

                # # Loop through time steps and channels
                # for t in range(pred_sample.shape[0]):         # 14
                #     for ch in range(pred_sample.shape[-1]):   # 6
                #         pred_img = pred_sample[t, :, :, ch]
                #         gt_img = gt_sample[t, :, :, ch]

                #         # Normalize to 0–255 if needed
                #         pred_img = ((pred_img - pred_img.min()) / (np.ptp(pred_img )+ 1e-8) * 255).astype('uint8')
                #         gt_img = ((gt_img - gt_img.min()) / (np.ptp(gt_img) + 1e-8) * 255).astype('uint8')

                #         # File names
                #         pred_path = f"pred_images/pred_t{t:02d}_ch{ch}.png"
                #         gt_path = f"gt_images/gt_t{t:02d}_ch{ch}.png"

                #         # Save
                #         cv2.imwrite(pred_path, pred_img)
                #         cv2.imwrite(gt_path, gt_img)

                # sys.exit(0)

                gt_batch = ims[:, 1:]
                gt_batch = torch.from_numpy(gt_batch.astype(np.float32)).to(args.device)  # .reshape(gt.shape[0], 1))
                gt_channel_split= torch.split(gt_batch, gt_batch.shape[-1] // args.num_views, dim=-1)
                gt_batch = torch.cat([t[..., -2:] for t in gt_channel_split], dim=-1)
                input_batch=torch.cat([t[..., :3] for t in gt_channel_split], dim=-1)
                input_batch = input_batch.detach().cpu().numpy()


                for view_idx in range(args.num_views):

                    path=os.path.join(root_res_path,"Train_images", str(epoch))
                    path=os.path.join(path, str(view_idx))
                    os.makedirs(path, exist_ok=True)

                    for i in range(train_batch.shape[1]):
                        name = 'input_{0:02d}_{1:02d}.png'.format(i + 1, view_idx)
                        file_name = os.path.join(path, name)
                        input_gt = np.uint8(input_batch[0, i, :, :, (view_idx * args.img_channel):(
                                    (view_idx + 1) * args.img_channel)] * 255)
                        input_gt = cv2.cvtColor(input_gt, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(file_name, input_gt)

                    for i in range(train_batch.shape[1]):
                        name = 'gtmask_{0:02d}_{1:02d}.png'.format(i + 1, view_idx)
                        file_name = os.path.join(path, name)
                        mask_img_gt = train_batch[0, i, :, :,
                                (view_idx * 3)+2:((view_idx *3)+3 )] 
                        # print("Mask min/max:", mask_img_gt.min(), mask_img_gt.max())
                        mask_img_gt = ((mask_img_gt * 255))                        
                        cv2.imwrite(file_name, np.uint8(mask_img_gt))

                    #gt dont need to multiply with mask because it wont have outliers
                    for i in range(train_batch.shape[1]):
                        name = 'gtt2n0_{0:02d}_{1:02d}.png'.format(i + 1, view_idx)
                        file_name = os.path.join(path, name)
                        t2no_img_gt = train_batch[0, i, :, :,
                                (view_idx * 3):((view_idx *3) +1)]
                        t2no_img_gt = ((t2no_img_gt * 255))                        
                        cv2.imwrite(file_name, np.uint8(t2no_img_gt))

                    for i in range(train_batch.shape[1]):
                        # name = 'gtdeltad_{0:02d}_{1:02d}.png'.format(i + 1, view_idx)
                        name = 'gt2nd_{0:02d}_{1:02d}.png'.format(i + 1, view_idx)
                        file_name = os.path.join(path, name)
                        t2nd_img_gt = train_batch[0, i, :, :,
                                (view_idx * 3)+1:((view_idx *3)+2 )] 
                        t2nd_img_gt = ((t2nd_img_gt * 255))                        
                        cv2.imwrite(file_name, np.uint8(t2nd_img_gt))




                    for i in range(pred_batch.shape[1]):
                        name = 'pdmask_{0:02d}_{1:02d}.png'.format(i + 1, view_idx)
                        file_name = os.path.join(path, name)
                        mask_img_pred = pred_batch[0, i, :, :,
                                (view_idx * 3)+2:((view_idx *3)+3 )] 
                        mask_img_pred = ((mask_img_pred * 255))                        
                        cv2.imwrite(file_name, np.uint8(mask_img_pred))


                    for i in range(pred_batch.shape[1]):
                        name = 'pdt2n0_raw__{0:02d}_{1:02d}.png'.format(i + 1, view_idx)
                        file_name = os.path.join(path, name)
                        t2no_img_pd = pred_batch[0, i, :, :,
                                (view_idx * 3):((view_idx *3) +1)]
                        t2no_img_pd = ((t2no_img_pd * 255))                        
                        cv2.imwrite(file_name, np.uint8(t2no_img_pd))


                    for i in range(pred_batch.shape[1]):
                        name = 'pdt2n0_{0:02d}_{1:02d}.png'.format(i + 1, view_idx)
                        file_name = os.path.join(path, name)
                        mask_img_pred = pred_batch[0, i, :, :,
                                (view_idx * 3)+2:((view_idx *3)+3 )] 
                        t2no_img_pd = pred_batch[0, i, :, :,
                                (view_idx * 3):((view_idx *3) +1)]
                        result_t2no_raw = np.where(mask_img_pred != 0, t2no_img_pd, 1)
                        result_t2no = ((result_t2no_raw * 255))                        
                        cv2.imwrite(file_name, np.uint8(result_t2no))

                   


                    for i in range(pred_batch.shape[1]):
                        name_1 = 'pdt2nd_{0:02d}_{1:02d}.png'.format(i + 1, view_idx)
                        # name_2 = 'pd_deltad{0:02d}_{1:02d}.png'.format(i + 1, view_idx)
                        # name_3 = 'pd_t2nd_raw{0:02d}_{1:02d}.png'.format(i + 1, view_idx)
                        file_name1 = os.path.join(path, name_1)
                        # file_name2 = os.path.join(path, name_2)
                        # file_name3 = os.path.join(path, name_3)
                        mask_img_pred = pred_batch[0, i, :, :,
                                (view_idx * 3)+2:((view_idx *3)+3 )] 
                        deld_img_pd = pred_batch[0, i, :, :,
                                (view_idx * 3)+1:((view_idx *3) +2)]

                        # t2nod_img_pred = deld_img_pd + result_t2no_raw

                        result_t2nd = np.where(mask_img_pred != 0, deld_img_pd, 1)
                        
                        t2nd_img_pd = ((result_t2nd * 255)) 

                        # deld_img_pd=((deld_img_pd*255))

                        # raw_t2nd=((t2nod_img_pred*255))

                        cv2.imwrite(file_name1, np.uint8(t2nd_img_pd))
                        # cv2.imwrite(file_name2, np.uint8(deld_img_pd))
                        # cv2.imwrite(file_name3, np.uint8(raw_t2nd))




                if epoch % 1 == 0:
                    
                    train_model_save_path=os.path.join(root_res_path,"SSTA_model", str(epoch))
                    os.makedirs(train_model_save_path, exist_ok=True)
                    
                    for ssta_key in models.keys():
                        train_model_save_path2=os.path.join(train_model_save_path,ssta_key + '.pt')
                        torch.save(models[ssta_key], train_model_save_path2)

 
    if args.mode=="eval":
        for name, _ in models.items():
                    models[name].eval()
        print("evaluating... ")
        batch_id = 0
        res_path = os.path.join(root_res_path, "eval_images")
        os.makedirs(res_path, exist_ok=True)

        test_input_handle.begin(do_shuffle=False)
        N,iter,sum_loss,Total_eval_images=0,0,0,0
        # print( test_input_handle.get_batch())
        while (test_input_handle.no_batch_left() == False and Total_eval_images<=args.disp_eval_images):

            batch_id = batch_id + 1
            ims = test_input_handle.get_batch()
            test_input_handle.next()
            x_batch = ims[:, :]
            gt_batch = ims[:, 1:]
            x_batch = torch.from_numpy(x_batch.astype(np.float32)).to(args.device)  # .reshape(x.shape[0], 1))
            gt_batch = torch.from_numpy(gt_batch.astype(np.float32)).to(args.device)  # .reshape(gt.shape[0], 1))

            gt_channel_split= torch.split(gt_batch, gt_batch.shape[-1] // args.num_views, dim=-1)
            gt_batch = torch.cat([t[..., -2:] for t in gt_channel_split], dim=-1)

            with torch.no_grad():
                pred_batch, _ ,_= run_steps(x_batch, models, optimizers, connections, vae,
                                                inference=True, args=args)
                
            
        
            ####
            # sum_loss += loss.data * args.vis_bs
            N+=1
            
            pred_batch = pred_batch.detach().cpu().numpy()
            gt_batch = ims[:, 1:]
            gt_batch = torch.from_numpy(gt_batch.astype(np.float32)).to(args.device)  # .reshape(gt.shape[0], 1))

            gt_channel_split= torch.split(gt_batch, gt_batch.shape[-1] // args.num_views, dim=-1)
            gt_batch = torch.cat([t[..., -2:] for t in gt_channel_split], dim=-1)
            input_batch=torch.cat([t[..., :3] for t in gt_channel_split], dim=-1)

            gt_batch = gt_batch.detach().cpu().numpy()
            input_batch = input_batch.detach().cpu().numpy()
        
            # print(input_batch.shape,pred_batch.shape,gt_batch.shape)

            if args.save_eval_images:
                for view_idx in range(args.num_views):

                    path=os.path.join(res_path, str(view_idx))
                    os.makedirs(path, exist_ok=True)

                    for i in range(pred_batch.shape[1]):
                        name = 'input_{0:02d}_{1:02d}.png'.format(iter , view_idx)
                        file_name = os.path.join(path, name)
                        input_gt = np.uint8(input_batch[0, i, :, :, (view_idx * args.img_channel):(
                                    (view_idx + 1) * args.img_channel)] * 255)
                        input_gt = cv2.cvtColor(input_gt, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(file_name, input_gt)


                        name = 'pdt2n0_{0:02d}_{1:02d}.png'.format(iter, view_idx)
                        file_name = os.path.join(path, name)
                        t2no_img_pd = pred_batch[0, i, :, :,
                                (view_idx * 2):((view_idx *2) +1)]
                        t2no_img_pd = ((t2no_img_pd * 255))                        
                        cv2.imwrite(file_name, np.uint8(t2no_img_pd))

                        name = 'gtt2n0_{0:02d}_{1:02d}.png'.format(iter, view_idx)
                        file_name = os.path.join(path, name)
                        t2no_img_gt = gt_batch[0, i, :, :,
                                (view_idx * 2):((view_idx *2) +1)]
                        t2no_img_gt = ((t2no_img_gt * 255))                        
                        cv2.imwrite(file_name, np.uint8(t2no_img_gt))


                        name = 'pdt2nd_{0:02d}_{1:02d}.png'.format(iter, view_idx)
                        file_name = os.path.join(path, name)

                        t2no_img_pd = pred_batch[0, i, :, :,
                                (view_idx * 2):((view_idx *2) +1)]         
                        t2nd_img_pd = pred_batch[0, i, :, :,
                                (view_idx * 2)+1:((view_idx *2)+2 )]
                        
                        t2nd_final_pd    =   t2no_img_pd+    t2nd_img_pd    
                        t2nd_final_pd = ((t2nd_final_pd * 255))  
                        cv2.imwrite(file_name, np.uint8(t2nd_final_pd))

                        name = 'gtt2nd_{0:02d}_{1:02d}.png'.format(iter, view_idx)
                        file_name = os.path.join(path, name)
                        t2nd_img_gt = gt_batch[0, i, :, :,
                                (view_idx * 2)+1:((view_idx *2)+2 )]
                        t2no_img_gt = gt_batch[0, i, :, :,
                                (view_idx * 2):((view_idx *2) +1)]
                        
                        t2nd_final_gt   =   t2no_img_gt+    t2nd_img_gt    
                        t2nd_final_gt = ((t2nd_final_gt * 255))                       
                        cv2.imwrite(file_name, np.uint8(t2nd_final_gt))

                        iter+=1
            Total_eval_images+=pred_batch.shape[1]
            


        # ave_loss = sum_loss / N 
        print("Total eval images computed with sequence:",N)
        # print("Eval averageloss",":",ave_loss)
        




def load_sstas(n, paths_list):

    models = {}
    optimizers = {}
    
    for i in range(n):
        ssta_name = 'ssta_' + str(i)
        models[ssta_name] = torch.load(paths_list[0],weights_only=False)
        optimizers[ssta_name] = optim.Adam(models[ssta_name].parameters(), lr = 0.0001)
        

    #### The connections need to be stored as well, meaning this should be moved and just obtained from a save file ####
    connections = {'ssta_0': ['ssta_1'], 'ssta_1': ['ssta_0']}

    return models, optimizers, connections


def create_sstas(n, input_dim, h_units, act, args):   
    """
    inputs:
    n (int): number of sstas to connect  
    parser.add_argument('--test_sequence', type=int, default=150)
    """

    models = {}
    optimizers = {}
    connections = {}

    for i in range(n):
        ssta_name = 'ssta_' + str(i)
        models[ssta_name] = SSTA_Net(input_dim, h_units, act, args)
        optimizers[ssta_name] = optim.Adam(models[ssta_name].parameters(), lr = 0.0001)

        print('Created {0}'.format(ssta_name))

    ################# NEEDS TO BE GENERALIZED ##################
    connections = {'ssta_0': ['ssta_1'], 'ssta_1': ['ssta_0']}

    return models, optimizers, connections

def get_relevant_msgs(model_key, messages, connections):
    relevant_msgs = []
    relevant_connections = connections[model_key]
    # print(model_key)
    # print(connections)
    # print(relevant_connections)
    # print(messages)
    for ssta in relevant_connections:
        relevant_msgs.append(messages[ssta])
    return relevant_msgs



if __name__ == "__main__":

    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--model_type', type=str, default='ssta',help='ssta / vae')
    parser.add_argument('--data_name', type=str, default='ssta_2025')
    parser.add_argument('--act', type=str, default="relu", help='relu')
    parser.add_argument('--mode', type=str, default="train", help='train / eval/transfer_learning')
    parser.add_argument('--eval_mode', type=str, default='train', help='multi_step_eval / single_step_eval')

    #ssta paramterts
    parser.add_argument('--num_views', type=int, default=2, help='num views')
    parser.add_argument('--train_sequence', type=int, default=15)
    parser.add_argument('--test_sequence', type=int, default=15)

    #the step of start index of sequence
    parser.add_argument('--sequence_index_gap', type=int, default=10)

    parser.add_argument('--n_epoch', type=int, default=2000, help='200')
    parser.add_argument('--continue_epoch', type=int, default=0, help='200')

    parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--vis_bs', type=int, default=2)
    parser.add_argument('--disp_eval_images', type=int, default=60)
    parser.add_argument('--save_eval_images', type=bool, default=True)
    parser.add_argument('--mask_per_step', type=int, default=1000000000)
    
    #hyperparamter for loss(T2no and t2nd)
    parser.add_argument('--alpha', type=float, default=9)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--threshold_time_step', type=int, default=100,help="timestep of t2no/t2nd")
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda:0 cuda:0; cpu:0 cpu:0')
    parser.add_argument('--loss_fn', type=str, default='ssim', help='ce/ mse /bce/ split_mse/ detr/ssim')

    # parser.add_argument('--num_step', type=int, default=15)
    parser.add_argument('--num_past', type=int, default=4)

    # RGB dataset
    parser.add_argument('--img_width', type=int, default=128, help='img width')
    parser.add_argument('--img_channel', type=int, default=3, help='img channel')
    parser.add_argument('--num_save_samples', type=int, default=10)
    parser.add_argument('--num_hidden', type=str, default='128,64,32,16', help='64,64,64,64')
    parser.add_argument('--filter_size', type=int, default=3)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--message_type', type=str, default='vae', help='normal, zeros, randn, raw_data, vae')
    #trained vae model latent dimesion same as loaded model
    parser.add_argument('--vae_latent_dim', type=int, default=5)
    parser.add_argument('--ssta_output_channels', type=int, default=2,help="channels - t2no/t2nd")
    #File paths
    #file to save ssta results
    parser.add_argument('--gen_frm_dir', type=str, default=r'./Trained_models_images/ssim_combo_ssta_128_64_32_16_apr20_25_latent5')
    parser.add_argument('--train_data_paths', type=str, default=r"./dataset_02/train")
    parser.add_argument('--valid_data_paths', type=str, default=r"./dataset_02/test")
    parser.add_argument('--vae_ckpt_dir', type=str, default=r"./vae_file_latent5",help='None')
    parser.add_argument('--ckpt_dir', type=str, default=r'./Trained_models_images/ssim_combo_ssta_128_64_32_16_apr20_25_latent5/SSTA_model/50', help='checkpoint dir')

    args = parser.parse_args()
    args.gen_frm_dir = os.path.join(args.gen_frm_dir)

    if args.loss_fn=="detr": 
        args.ssta_output_channels=3

    if args.loss_fn=="ssim": 
        args.ssta_output_channels=3

    if args.loss_fn=="ce": 
        args.ssta_output_channels=args.threshold_time_step*2


    training(args.n_epoch,args.act, args)
