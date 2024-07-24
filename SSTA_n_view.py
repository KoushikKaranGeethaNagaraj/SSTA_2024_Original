import numpy as np
import time
from matplotlib import pyplot as plt
import os, cv2

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
import pytorch_ssim






seed = 0
random.seed(seed)

class SSTA_Net(nn.Module):
    def __init__(self, input_dim, h_units, act, args):
        super(SSTA_Net, self).__init__()
        # [10, 128, 128, 5]
        self.filter_size = args.filter_size
        self.padding = self.filter_size // 2

        self.frame_predictor = DeterministicConvLSTM(input_dim, h_units[-1], h_units[0], len(h_units), args)
        self.l3 = nn.Conv3d(h_units[-1], 2, kernel_size=self.filter_size, stride=1, padding=self.padding, bias=False)

        if act == "relu":
            self.act = F.relu
        elif act == "sig":
            self.act = F.sigmoid

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
        pred_x_tp1 = F.sigmoid(pred_x_tp1)
        return pred_x_tp1, message, frame_predictor_hidden

    def predict(self, x, m_t, m_t_others, memory):
        pred_x_tp1, message, memory\
            = self.forward(x, m_t, m_t_others, memory)
        return pred_x_tp1.data, message.data, memory
    
def run_steps(x_batch, models, optimizers, connections, vae, inference = True, args = None):
    
    num_hidden = [int(x) for x in args.num_hidden.split(',')]
    batch = x_batch.shape[0]
    height = x_batch.shape[2]
    width = x_batch.shape[3]

    memory = [None for _ in range(args.num_views)]

    # memory_0 = None  # torch.zeros([batch, num_hidden[0], height, width]).to(args.device)
    # memory_1 = None  # torch.zeros([batch, num_hidden[0], height, width]).to(args.device)
    # print(x_batch.shape)
    x_t = torch.split(x_batch, x_batch.shape[-1] // args.num_views, dim=-1)

    # if True:
    #     view_1 = x_t[0].squeeze().detach().cpu()[0,:,:,0:3]
    #     view_2 = x_t[1].squeeze().detach().cpu()[0,:,:,0:3]
    #     views = np.concatenate([view_1, view_2], axis = 0)
    #     cv2.imshow('view',views)
    #     cv2.waitKey(100)
    # x_0_t, x_1_t = torch.split(x_batch, x_batch.shape[-1] // args.num_views, dim=-1)
    pred_batch_list = [[] for _ in range(args.num_views)]
    message_list = [[] for _ in range(args.num_views)]

    message_0 = x_t[0][:, 0:0 + 1,:,:, 0:3]
    # print(message_0.shape)

    messages = {}
    if args.message_type == 'raw_data':
        for view, ssta_name in enumerate(models.keys()):
            messages[ssta_name] = x_t[view][:, 0:0 + 1,:,:, 0:3]

    elif args.message_type == 'vae':
        for view, ssta_name in enumerate(models.keys()):
            messages[ssta_name] = vae.get_message(x_t[view][:, 0:0 + 1,:,:, 0:3])

    else:
        for view, ssta_name in enumerate(models.keys()):
            messages[ssta_name] = torch.zeros((x_batch.shape[0], 1, x_batch.shape[2], x_batch.shape[3], 1)).to(args.device)

    #above messages
    if args.eval_mode == 'multi_step_eval' and inference == True:

        x_t_prev_preds = []
        for view in range(args.num_views):
            x_t_prev_preds.append(x_t[view][:, 0:0 + 1,:,:, 0:3])
 
        use_gt_flag = False
        for t in range(args.valid_sequence - 1):
            message_others = get_relevant_msgs(ssta_key, messages, connections)

            for view, (ssta_key,model) in enumerate(models.items()):
                
                x_t_pred, messages[ssta_key], memory[view] = model(x_t_prev_preds[view], messages[ssta_key], message_others, memory[view])
                
                if args.message_type in ['vae']:
                    if t < args.num_past or np.random.uniform(0, 1) > (1-1/args.mask_per_step):  # t % args.mask_per_step == 0:
                        messages[ssta_key] = vae.get_message(x_t[view][:, t:t + 1,:,:,0:3])
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

        pred_batch_before = [torch.cat(first,1) for first in pred_batch_list]
        pred_batch = torch.cat(pred_batch_before, -1)

        message_list_before = [torch.cat(first,1) for first in message_list]
        message_batch = torch.cat(message_list_before, -1)

        
    else:
        torch.autograd.set_detect_anomaly(True)
        MSE = nn.MSELoss()
        x_t_prev_preds = []
        for view in range(args.num_views):
            x_t_prev_preds.append(x_t[view][:, 0:0 + 1,:,:, 0:3])
        
        for t in range(args.train_sequence-1):
            for view, (ssta_key,model) in enumerate(models.items()):
                model.train()

                for ssta_key in optimizers:
                    optimizers[ssta_key].zero_grad()

                message_others = get_relevant_msgs(ssta_key, messages, connections)
                # print(x_t_prev_preds[view].shape, messages[ssta_key].shape, len(message_others),print(memory[view]))
                x_t_pred, messages[ssta_key], memory_temp = model(x_t_prev_preds[view], messages[ssta_key], message_others, memory[view])
                
                memory[view] = [(mem1.detach(), mem2.detach()) for mem1,mem2 in memory_temp]

                if args.message_type in ['vae']:
                    if t < args.num_past or np.random.uniform(0, 1) > (1-1/args.mask_per_step):  # t % args.mask_per_step == 0:
                        messages[ssta_key] = vae.get_message(x_t[view][:, t:t + 1,:,:,0:3])
                        # message_0 = vae.get_message(x_t[:, t:t + 1])
                    else:
                        messages[ssta_key] = vae.get_message(x_t_prev_preds[view].detach())

                elif args.message_type in ['raw_data']:
                    messages[ssta_key] = x_t[view][:, t:t + 1,:,:,0:3].detach()

                elif args.message_type == 'zeros':
                    messages[ssta_key] = torch.zeros_like(messages[ssta_key])

                elif args.message_type == 'randn':
                    messages[ssta_key] = torch.randn_like(messages[ssta_key])

                
                gt_train =  x_t[view][:, t:t + 1,:,:, 3:]
                # print(x_t_pred.shape, x_t_prev_preds[view].shape, gt_train.shape)



                loss = MSE(x_t_pred, gt_train)
                
                # print(loss, ssta_key)
                loss.backward(retain_graph = True)
                optimizers[ssta_key].step()
                
                pred_batch_list[view].append(x_t_pred)
                message_list[view].append(messages[ssta_key])

                x_t_prev_preds[view] = x_t[view][:, t+1:t+2,:,:, 0:3]

                # if view == 0:
                #     example = x_t_pred[0].squeeze().detach().cpu().numpy()[:,:,0]
                #     example = x_t_prev_preds[view].squeeze().detach().cpu().numpy()
                #     # print(example.shape)
                #     cv2.imshow("ex",example)
                #     cv2.waitKey(10)


        pred_batch_before = [torch.cat(first,1) for first in pred_batch_list]
        pred_batch = torch.cat(pred_batch_before, -1)

        message_list_before = [torch.cat(first,1) for first in message_list]
        message_batch = torch.cat(message_list_before, -1)


    return pred_batch, message_batch



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
    vae = torch.load(vae_path)
    vae = vae.to(args.device)
    print('Loaded VAE model_0 from {}'.format(vae_path))

    # optimizer = optim.Adam(parameters,lr = 0.0001)

    MSE = nn.MSELoss()
    ssim_loss = pytorch_ssim.SSIM(window_size = 11)


    root_res_path = os.path.join(args.gen_frm_dir)
    os.makedirs(root_res_path, exist_ok=True)

    print("START")
    best_eval_loss = np.inf
    total_final_loss=[]
    continue_epoch=0
    
    if args.mode=="transfer_learning":
        continue_epoch=args.continue_epoch

    if args.mode=="train" or args.mode=="transfer_learning":
        for epoch in range(1+continue_epoch, n_epoch + 1):
            print("-----------------------",epoch,"------------------")
            if args.mode == 'train' or args.mode=="transfer_learning":
                for name, _ in models.items():
                    models[name].train()
                
                sum_loss = 0
                N,iter=0,0
                print('Training ... {}'.format(epoch))
                train_input_handle.begin(do_shuffle=True)
                progress_bar = tqdm(total=train_input_handle.total()-1, desc='Epoch Completion')
                
                while (train_input_handle.no_batch_left() == False):
                    ims = train_input_handle.get_batch()
                    train_input_handle.next()
                    x_batch = ims[:, :]
                    
                    gt_batch = ims[:, 1:]
                    x_batch = torch.from_numpy(x_batch.astype(np.float32)).to(args.device)  # .reshape(x.shape[0], 1))
                    gt_batch = torch.from_numpy(gt_batch.astype(np.float32)).to(args.device)  # .reshape(gt.shape[0], 1))

                    gt_channel_split= torch.split(gt_batch, gt_batch.shape[-1] // args.num_views, dim=-1)
                    gt_batch = torch.cat([t[..., -2:] for t in gt_channel_split], dim=-1)


                    pred_batch, message_batch = run_steps(x_batch, models, optimizers, connections, vae,
                                                        inference=False, args=args)
                    
                    #MSE LOSS-together
                    # loss = MSE(pred_batch, gt_batch)
                    ##


                    #####Loss seperated as T2NO and T2nD for each view and ##threshold
                    loss_gt_channel_split= torch.split(gt_batch, gt_batch.shape[-1] // args.num_views, dim=-1)
                    loss_pd_channel_split=torch.split(pred_batch, gt_batch.shape[-1] // args.num_views, dim=-1)

            
                    loss=0.0
                    for i in range(len(loss_gt_channel_split)):
                        ##threshold
                        # thresh_pd_t2no=loss_pd_channel_split[i][:,:,:,:,0]*255
                        # inverse_gt_t2no=loss_gt_channel_split[i][:,:,:,:,0]
                        # # print(inverse_gt_t2no)
                        # inverse_gt_t2no = inverse_gt_t2no.float()/50.0
                        # inverse_gt_t2no[torch.isinf(inverse_gt_t2no)] = 0.000000000000001




                        


                        # a=inverse_gt_t2no[0,0,:,:].detach().cpu().numpy()
                        # print(a)
                        # print(a*255)
                        # print(np.unique(a*255))
                        # cv2.imwrite("a.jpg",a*255)
                        # import sys
                        # sys.exit(0)
                        # print(torch.unique(thresh_pd_t2no),torch.unique(thresh_gt_t2no))


                        loss_t2no_mse=MSE(loss_pd_channel_split[i][:,:,:,:,0],loss_gt_channel_split[i][:,:,:,:,0])
                        loss_t2nd=MSE(loss_pd_channel_split[i][:,:,:,:,1],loss_gt_channel_split[i][:,:,:,:,1])
                        # loss_t2no_bce=BCE(loss_pd_channel_split[i][:,:,:,:,0],loss_gt_channel_split[i][:,:,:,:,0])
                        # print(loss_t2no_bce,loss_t2no_mse)
                        # loss_t2no=(args.alpha*loss_t2no_mse)+(args.alpha2*loss_t2no_bce)
                        # loss_t2no=loss_t2no_bce
                        t2no_ssim_loss = -ssim_loss(loss_pd_channel_split[i][:,:,:,:,0], loss_gt_channel_split[i][:,:,:,:,0])
                        loss_t2no=t2no_ssim_loss
                        # print(t2no_ssim_loss,loss_t2no_mse)
                        loss+=(args.alpha*loss_t2no)+(args.beta*loss_t2nd)
                
                    # print("old",loss)
                    ######

                    sum_loss += loss.data * args.bs
                    # loss.backward()
                    # for optimizer in optimizers:
                    #     optimizer.step()
                    
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
                gt_batch = ims[:, 1:]

                gt_batch = torch.from_numpy(gt_batch.astype(np.float32)).to(args.device)  # .reshape(gt.shape[0], 1))

                gt_channel_split= torch.split(gt_batch, gt_batch.shape[-1] // args.num_views, dim=-1)
                gt_batch = torch.cat([t[..., -2:] for t in gt_channel_split], dim=-1)
                input_batch=torch.cat([t[..., :3] for t in gt_channel_split], dim=-1)

                gt_batch = gt_batch.detach().cpu().numpy()
                input_batch = input_batch.detach().cpu().numpy()

                print(input_batch.shape,pred_batch.shape,gt_batch.shape)

                for view_idx in range(args.num_views):

                    path=os.path.join(root_res_path,"Train_images", str(epoch))
                    path=os.path.join(path, str(view_idx))
                    os.makedirs(path, exist_ok=True)

                    for i in range(pred_batch.shape[1]):
                        name = 'input_{0:02d}_{1:02d}.png'.format(i + 1, view_idx)
                        file_name = os.path.join(path, name)
                        input_gt = np.uint8(input_batch[0, i, :, :, (view_idx * args.img_channel):(
                                    (view_idx + 1) * args.img_channel)] * 255)
                        input_gt = cv2.cvtColor(input_gt, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(file_name, input_gt)

                    for i in range(pred_batch.shape[1]):
                        name = 'pdt2n0_{0:02d}_{1:02d}.png'.format(i + 1, view_idx)
                        file_name = os.path.join(path, name)
                        t2no_img_pd = pred_batch[0, i, :, :,
                                (view_idx * 2):((view_idx *2) +1)]
                        t2no_img_pd = ((t2no_img_pd * 255))                        
                        cv2.imwrite(file_name, np.uint8(t2no_img_pd))

                    for i in range(gt_batch.shape[1]):
                        name = 'gtt2n0_{0:02d}_{1:02d}.png'.format(i + 1, view_idx)
                        file_name = os.path.join(path, name)
                        t2no_img_gt = gt_batch[0, i, :, :,
                                (view_idx * 2):((view_idx *2) +1)]
                        t2no_img_gt = ((t2no_img_gt * 255))                        
                        cv2.imwrite(file_name, np.uint8(t2no_img_gt))

                    for i in range(pred_batch.shape[1]):
                        name = 'pdt2nd_{0:02d}_{1:02d}.png'.format(i + 1, view_idx)
                        file_name = os.path.join(path, name)
                        t2nd_img_pd = pred_batch[0, i, :, :,
                                (view_idx * 2)+1:((view_idx *2)+2 )]
                        t2nd_img_pd = ((t2nd_img_pd * 255))                        
                        cv2.imwrite(file_name, np.uint8(t2nd_img_pd))

                    for i in range(gt_batch.shape[1]):
                        name = 'gtt2nd_{0:02d}_{1:02d}.png'.format(i + 1, view_idx)
                        file_name = os.path.join(path, name)
                        t2nd_img_gt = gt_batch[0, i, :, :,
                                (view_idx * 2)+1:((view_idx *2)+2 )]
                        t2nd_img_gt= ((t2nd_img_gt * 255))                        
                        cv2.imwrite(file_name, np.uint8(t2nd_img_gt))

                if epoch % 1 == 0:
                    
                    train_model_save_path=os.path.join(root_res_path,"SSTA_model", str(epoch))
                    os.makedirs(train_model_save_path, exist_ok=True)
                    
                    for ssta_key in models.keys():
                        train_model_save_path2=os.path.join(train_model_save_path,ssta_key + '.pt')
                        torch.save(models[ssta_key], train_model_save_path2)

                    
                    print("evaluating... ")

                    for name, _ in models.items():
                        models[name].eval()

                    batch_id = 0
                    eval_res_path = os.path.join(root_res_path, "eval_images")
                    eval_res_path=os.path.join(eval_res_path, str(epoch))
                    os.makedirs(eval_res_path, exist_ok=True)

                    test_input_handle.begin(do_shuffle=False)
                    N,iter,sum_loss,Total_eval_images,ave_loss=0,0,0,0,0
                    # print( test_input_handle.get_batch())
                    while (test_input_handle.no_batch_left() == False ):

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
                            pred_batch, _ = run_steps(x_batch, models, optimizers, connections, vae,
                                                            inference=True, args=args)
                            
                        # print(pred_batch.shape)
                        
                        #MSE LOSS-together
                        # loss = MSE(pred_batch, gt_batch)
                        #####
                        loss_gt_channel_split= torch.split(gt_batch, gt_batch.shape[-1] // args.num_views, dim=-1)
                        loss_pd_channel_split=torch.split(pred_batch, gt_batch.shape[-1] // args.num_views, dim=-1)
                        loss=0.0

                        # loss_exp = 
                        for i in range(len(loss_gt_channel_split)):
                            loss_t2no_mse=MSE(loss_pd_channel_split[i][:,:,:,:,0],loss_gt_channel_split[i][:,:,:,:,0])
                            # loss_t2no2=MSE(loss_pd_channel_split[i][:,:,:,:,0]*loss_gt_channel_split[i][:,:,:,:,0],loss_gt_channel_split[i][:,:,:,:,0]*loss_gt_channel_split[i][:,:,:,:,0])

                            loss_t2nd=MSE(loss_pd_channel_split[i][:,:,:,:,1],loss_gt_channel_split[i][:,:,:,:,1])
                            # loss_exp = (loss_pd_channel_split[i][:,:,:,:,0]-loss_gt_channel_split[i][:,:,:,:,0])

                            # print(loss_t2no,loss_t2no2)
                            # loss_t2no_bce=BCE(loss_pd_channel_split[i][:,:,:,:,0],loss_gt_channel_split[i][:,:,:,:,0])
                            # print(loss_t2no_bce,loss_t2no_mse)
                            # loss_t2no=(args.alpha1*loss_t2no_mse)+(args.alpha2*loss_t2no_bce)
                            # loss_t2no=loss_t2no_bcee
                            t2no_ssim_loss = -ssim_loss(loss_pd_channel_split[i][:,:,:,:,0], loss_gt_channel_split[i][:,:,:,:,0])
                            loss_t2no=t2no_ssim_loss
                            
                            loss+=(args.alpha*loss_t2no)+(args.beta*loss_t2nd)
                            

                
                        # print("old",loss)
                        ######
                     
                        sum_loss += loss.data * args.vis_bs
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

                        if args.save_eval_images and Total_eval_images<=args.disp_eval_images:
                            for view_idx in range(args.num_views):

                                path=os.path.join(eval_res_path, str(view_idx))
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
                                    t2nd_img_pd = pred_batch[0, i, :, :,
                                            (view_idx * 2)+1:((view_idx *2)+2 )]
                                    t2nd_img_pd = ((t2nd_img_pd * 255))                        
                                    cv2.imwrite(file_name, np.uint8(t2nd_img_pd))


                                    name = 'gtt2nd_{0:02d}_{1:02d}.png'.format(iter, view_idx)
                                    file_name = os.path.join(path, name)
                                    t2nd_img_gt = gt_batch[0, i, :, :,
                                            (view_idx * 2)+1:((view_idx *2)+2 )]
                                    t2nd_img_gt= ((t2nd_img_gt * 255))                        
                                    cv2.imwrite(file_name, np.uint8(t2nd_img_gt))

                                    iter+=1
                        Total_eval_images+=pred_batch.shape[1]
                    ave_loss = sum_loss / N 
                    loss_val.append(ave_loss.data)
                    print("Total eval images computed with sequence:",N)
                    print("Eval averageloss",":",ave_loss.data)

                        


 
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
                pred_batch, _ = run_steps(x_batch, models, optimizers, connections, vae,
                                                inference=True, args=args)
                
            # print(pred_batch.shape)
            
            #MSE LOSS-together
            loss = MSE(pred_batch, gt_batch)
            ####
            sum_loss += loss.data * args.vis_bs
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
                        t2nd_img_pd = pred_batch[0, i, :, :,
                                (view_idx * 2)+1:((view_idx *2)+2 )]
                        t2nd_img_pd = ((t2nd_img_pd * 255))                        
                        cv2.imwrite(file_name, np.uint8(t2nd_img_pd))


                        name = 'gtt2nd_{0:02d}_{1:02d}.png'.format(iter, view_idx)
                        file_name = os.path.join(path, name)
                        t2nd_img_gt = gt_batch[0, i, :, :,
                                (view_idx * 2)+1:((view_idx *2)+2 )]
                        t2nd_img_gt= ((t2nd_img_gt * 255))                        
                        cv2.imwrite(file_name, np.uint8(t2nd_img_gt))

                        iter+=1
            Total_eval_images+=pred_batch.shape[1]
            


        ave_loss = sum_loss / N 
        print("Total eval images computed with sequence:",N)
        print("Eval averageloss",":",ave_loss)
        




def load_sstas(n, paths_list):

    models = {}
    optimizers = {}
    
    for i in range(n):
        ssta_name = 'ssta_' + str(i)
        models[ssta_name] = torch.load(paths_list[0])
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
    parser.add_argument('--data_name', type=str, default='ssta_2024')
    parser.add_argument('--act', type=str, default="relu", help='relu')
    parser.add_argument('--mode', type=str, default="train", help='train / eval/transfer_learning')
    parser.add_argument('--eval_mode', type=str, default='single_step_eval', help='multi_step_eval / single_step_eval')

    #ssta paramterts
    parser.add_argument('--num_views', type=int, default=2, help='num views')
    parser.add_argument('--train_sequence', type=int, default=100)
    parser.add_argument('--test_sequence', type=int, default=100)
    #the step of start index of sequence
    parser.add_argument('--sequence_index_gap', type=int, default=50)

    parser.add_argument('--n_epoch', type=int, default=300, help='200')
    parser.add_argument('--continue_epoch', type=int, default=4, help='200')

    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--vis_bs', type=int, default=1)
    parser.add_argument('--disp_eval_images', type=int, default=310)
    parser.add_argument('--save_eval_images', type=bool, default=True)
    parser.add_argument('--mask_per_step', type=int, default=1000000000)
    #hyperparamter for loss(T2no and t2nd)
   
    parser.add_argument('--alpha', type=float, default=9)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--threshold_time_step', type=int, default=50,help="timestep of t2no/t2nd")
    ##
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda:0 cuda:0; cpu:0 cpu:0')

    # parser.add_argument('--num_step', type=int, default=15)
    parser.add_argument('--num_past', type=int, default=2)

    # RGB dataset
    parser.add_argument('--img_width', type=int, default=128, help='img width')
    parser.add_argument('--img_channel', type=int, default=3, help='img channel')
  
    parser.add_argument('--num_save_samples', type=int, default=10)
    parser.add_argument('--num_hidden', type=str, default='32,32,32,32', help='64,64,64,64')
    parser.add_argument('--filter_size', type=int, default=3)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--message_type', type=str, default='raw_data', help='normal, zeros, randn, raw_data, vae')
    #trained vae model latent dimesion same as loaded model
    parser.add_argument('--vae_latent_dim', type=int, default=4)
    #File paths
    #file to save ssta results
    parser.add_argument('--gen_frm_dir', type=str, default='inference_32_32_32_32')
    parser.add_argument('--train_data_paths', type=str, default=r"/home/knagaraj31/SSTA_2024/NEW_DATASET/train")
    parser.add_argument('--valid_data_paths', type=str, default=r"/home/knagaraj31/SSTA_2024/NEW_DATASET/val")
    parser.add_argument('--vae_ckpt_dir', type=str, default="/home/knagaraj31/SSTA_2024/vae_file",help='None')
    parser.add_argument('--ckpt_dir', type=str, default='/home/knagaraj31/SSTA_2024/ssta2_sim_model_32_32_32_32_seq_100_new/SSTA_model/3', help='checkpoint dir')

    args = parser.parse_args()
    args.gen_frm_dir = os.path.join(args.gen_frm_dir)
    training(args.n_epoch,args.act, args)
