# coding: utf-8
# https://linuxtut.com/en/fe2d3308b3ba56a80c7a/

import numpy as np
import time
from matplotlib import pyplot as plt
import os, cv2

# pytorch
import torch
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
torch.cuda.empty_cache()
gc.collect()




seed = 0
random.seed(seed)

class SSTA_Net(nn.Module):
    def __init__(self, input_dim, h_units, act, args):
        super(SSTA_Net, self).__init__()
        # [10, 128, 128, 5]
        self.filter_size = args.filter_size
        self.padding = self.filter_size // 2

        self.frame_predictor = DeterministicConvLSTM(input_dim, h_units[-1], h_units[0], len(h_units), args)
        self.l3 = nn.Conv3d(h_units[-1], 3, kernel_size=self.filter_size, stride=1, padding=self.padding, bias=False)

        if act == "relu":
            self.act = F.relu
        elif act == "sig":
            self.act = F.sigmoid

    def __call__(self, x, m_t, m_t_others, memory, h_t, c_t, delta_c_list, delta_m_list):
        pred_x_tp1, message, memory, h_t, c_t, delta_c_list, delta_m_list\
            = self.forward(x, m_t, m_t_others, memory, h_t, c_t, delta_c_list, delta_m_list)
        return pred_x_tp1, message, memory, h_t, c_t, delta_c_list, delta_m_list

    def forward(self, x_t, m_t, m_t_others, frame_predictor_hidden, h_t, c_t, delta_c_list, delta_m_list):
        x = torch.cat([x_t, m_t, m_t_others], -1)
        x = x.permute(0, 4, 1, 2, 3)
        h, frame_predictor_hidden = self.frame_predictor(x, frame_predictor_hidden)
        pred_x_tp1 = self.l3(h)
        message = None
        pred_x_tp1 = pred_x_tp1.permute(0, 2, 3, 4, 1)
        # print(pred_x_tp1.shape)
        # quit()

        pred_x_tp1 = F.sigmoid(pred_x_tp1)
        # pred_x_tp1 = F.sigmoid(pred_x_tp1)
        return pred_x_tp1, message, frame_predictor_hidden, h_t, c_t, delta_c_list, delta_m_list

    def predict(self, x, m_t, m_t_others, memory, h_t, c_t, delta_c_list, delta_m_list):
        pred_x_tp1, message, memory, h_t, c_t, delta_c_list, delta_m_list\
            = self.forward(x, m_t, m_t_others, memory, h_t, c_t, delta_c_list, delta_m_list)
        return pred_x_tp1.data, message.data, memory, h_t, c_t, delta_c_list, delta_m_list


def run_steps(x_batch, model_0, model_1, vae, inference=True, args=None):
    '''
    x_batch: [bs, T, 2*2],
    pred_batch: [bs, T, 1*2],
    x_batch: [bs, T, H, W, C], C=3
    pred_batch: [bs, T, H, W, C],
    '''
    num_hidden = [int(x) for x in args.num_hidden.split(',')]
    batch = x_batch.shape[0]
    height = x_batch.shape[2]
    width = x_batch.shape[3]
    # print(x_batch.shape)
    
    h_t_0 = []
    c_t_0 = []
    h_t_1 = []
    c_t_1 = []
    delta_c_list_0 = []
    delta_m_list_0 = []
    delta_c_list_1 = []
    delta_m_list_1 = []

    decouple_loss = []

    for i in range(len(num_hidden)):
        zeros = torch.zeros([batch, num_hidden[i], height, width]).to(args.device)
        h_t_0.append(zeros)
        c_t_0.append(zeros)
        h_t_1.append(zeros)
        c_t_1.append(zeros)
        delta_c_list_0.append(zeros)
        delta_m_list_0.append(zeros)
        delta_c_list_1.append(zeros)
        delta_m_list_1.append(zeros)

    memory_0 = None  # torch.zeros([batch, num_hidden[0], height, width]).to(args.device)
    memory_1 = None  # torch.zeros([batch, num_hidden[0], height, width]).to(args.device)
    x_0_t, x_1_t = torch.split(x_batch, x_batch.shape[-1] // 2, dim=-1)
    pred_batch_0_list = []
    pred_batch_1_list = []
    message_0_list = []
    message_1_list = []
    if args.message_type == 'raw_data':
        message_0 = x_0_t[:, 0:0 + 1]
        message_1 = x_1_t[:, 0:0 + 1]
    elif args.message_type == 'vae':
        message_0 = vae.get_message(x_0_t[:, 0:0 + 1])
        message_1 = vae.get_message(x_1_t[:, 0:0 + 1])
     
    else:
        message_0 = torch.zeros((x_batch.shape[0], 1, x_batch.shape[2], x_batch.shape[3], 1)).to(args.device)
        message_1 = torch.zeros((x_batch.shape[0], 1, x_batch.shape[2], x_batch.shape[3], 1)).to(args.device)
    #above messages
    if args.eval_mode == 'multi_step_eval' and inference == True:  # and inference == True
        x_0_t_pred_prev = x_0_t[:, 0:1]
        x_1_t_pred_prev = x_1_t[:, 0:1]
 
        use_gt_flag = False
        for t in range(args.eval_num_step + args.num_past):
            x_0_t_pred, message_0, memory_0, h_t_0, c_t_0, delta_c_list_0, delta_m_list_0 = \
                model_0(x_0_t_pred_prev, message_0, message_1, memory_0, h_t_0, c_t_0, delta_c_list_0, delta_m_list_0)
            
            if args.message_type in ['vae']:
                if t < args.num_past or use_gt_flag:
                    message_0 = vae.get_message(x_0_t[:, t:t + 1])
                else:
                    message_0 = vae.get_message(x_0_t_pred_prev.detach())
            elif args.message_type in ['raw_data']:
                if t < args.num_past or use_gt_flag:
                    message_0 = x_0_t[:, t:t + 1]
                else:
                    message_0 = x_0_t_pred
            elif args.message_type == 'zeros':
                message_0 = torch.zeros_like(message_0)
            elif args.message_type == 'randn':
                message_0 = torch.randn_like(message_0)
            x_1_t_pred, message_1, memory_1, h_t_1, c_t_1, delta_c_list_1, delta_m_list_1 = \
                model_1(x_1_t_pred_prev, message_1, message_0, memory_1, h_t_1, c_t_1, delta_c_list_1, delta_m_list_1)
            if args.message_type in ['vae']:
                if t < args.num_past or use_gt_flag:
                    message_1 = vae.get_message(x_1_t[:, t:t + 1])
                else:
                    # message_1 = vae.get_message(x_1_t[:, t:t + 1])
                    message_1 = vae.get_message(x_1_t_pred_prev.detach())
            elif args.message_type in ['raw_data']:
                if t < args.num_past or use_gt_flag:
                    message_1 = x_1_t[:, t:t + 1]
                else:
                    message_1 = x_1_t_pred
            elif args.message_type == 'zeros':
                message_1 = torch.zeros_like(message_1)
            elif args.message_type == 'randn':
                message_1 = torch.randn_like(message_1)
            if t < args.num_past:
                x_0_t_pred_prev = x_0_t[:, t + 1:t + 2]  # torch.cat([x_0_t_pred_prev[:, 1:], x_0_t_pred], 1)
                x_1_t_pred_prev = x_1_t[:, t + 1:t + 2]
#             elif use_gt_flag and (args.mode == 'train' or args.mode=="transfer_learning"):
            elif use_gt_flag:
## uncomment this if you want to use GT to predict and again GT to predict also comment this for testing
                # print("new")
                x_0_t_pred_prev = x_0_t[:, t + 1:t + 2]  # torch.cat([x_0_t_pred_prev[:, 1:], x_0_t_pred], 1)
                x_1_t_pred_prev = x_1_t[:, t + 1:t + 2]
                pred_batch_0_list.append(x_0_t_pred)
                pred_batch_1_list.append(x_1_t_pred)
                message_0_list.append(message_0)
                message_1_list.append(message_1)
            else:
                x_0_t_pred_prev = x_0_t_pred.detach()
                x_1_t_pred_prev = x_1_t_pred.detach()
                pred_batch_0_list.append(x_0_t_pred)
                pred_batch_1_list.append(x_1_t_pred)
                message_0_list.append(message_0)
                message_1_list.append(message_1)
            if t % args.eval_per_step == 0:
                memory_0 = None
                memory_1 = None
                use_gt_flag = not use_gt_flag
                # print('t: {}, use_gt_flag: {}'.format(t, use_gt_flag))
     
    else:
        x_0_t_pred_prev = x_0_t[:, 0:1]
        x_1_t_pred_prev = x_1_t[:, 0:1]
        
        for t in range(args.num_step + args.num_past):
            
            x_0_t_pred, message_0, memory_0, h_t_0, c_t_0, delta_c_list_0, delta_m_list_0 = \
                model_0(x_0_t_pred_prev, message_0, message_1, memory_0, h_t_0, c_t_0, delta_c_list_0,
                        delta_m_list_0)
            if args.message_type in ['vae']:
                if t < args.num_past or np.random.uniform(0, 1) > (1-1/args.mask_per_step):  # t % args.mask_per_step == 0:
                    message_0 = vae.get_message(x_0_t[:, t:t + 1])
                else:
                    message_0 = vae.get_message(x_0_t_pred_prev.detach())
            elif args.message_type in ['raw_data']:
                message_0 = x_0_t[:, t:t + 1]
            elif args.message_type == 'zeros':
                message_0 = torch.zeros_like(message_0)
            elif args.message_type == 'randn':
                message_0 = torch.randn_like(message_0)

            #######################################################################

            x_1_t_pred, message_1, memory_1, h_t_1, c_t_1, delta_c_list_1, delta_m_list_1 = \
                model_1(x_1_t_pred_prev, message_1, message_0, memory_1, h_t_1, c_t_1, delta_c_list_1,
                        delta_m_list_1)
            
           
            if args.message_type in ['vae']:
                if t < args.num_past or np.random.uniform(0, 1) > (1-1/args.mask_per_step): # or t % args.mask_per_step == 0:
                    message_1 = vae.get_message(x_1_t[:, t:t + 1])
                else:
                    message_1 = vae.get_message(x_1_t_pred_prev.detach())
            elif args.message_type in ['raw_data']:
                message_1 = x_1_t[:, t:t + 1]
            elif args.message_type == 'zeros':
                message_1 = torch.zeros_like(message_1)
            elif args.message_type == 'randn':
                message_1 = torch.randn_like(message_1)

            ########################################################################
            if t < args.num_past or np.random.uniform(0, 1) > (1-1/args.mask_per_step): # or t % args.mask_per_step == 0:
                x_0_t_pred_prev = x_0_t[:, t+1:t+2]  # torch.cat([x_0_t_pred_prev[:, 1:], x_0_t_pred], 1)
                x_1_t_pred_prev = x_1_t[:, t+1:t+2]
            else:
                x_0_t_pred_prev = x_0_t_pred.detach()
                x_1_t_pred_prev = x_1_t_pred.detach()
            example = x_0_t_pred[0].squeeze().detach().cpu().numpy()
            # print("####### Pred Image Shape #####", example.shape)
            example = cv2.cvtColor(example, cv2.COLOR_BGR2RGB)
            cv2.imshow("example", example)
            cv2.waitKey(10)
            pred_batch_0_list.append(x_0_t_pred)
            pred_batch_1_list.append(x_1_t_pred)
            message_0_list.append(message_0)
            message_1_list.append(message_1)

    pred_batch_0 = torch.cat(pred_batch_0_list, 1)
    pred_batch_1 = torch.cat(pred_batch_1_list, 1)
    pred_batch = torch.cat([pred_batch_0, pred_batch_1], -1)
    message_batch_0 = torch.cat(message_0_list, 1)
    message_batch_1 = torch.cat(message_1_list, 1)
    message_batch = torch.cat([message_batch_0, message_batch_1], -1)
    return pred_batch, message_batch


def training(N, Nte, bs, n_epoch, act, data_mode, args):
   
    # x_train, t_train, x_test, t_test, step_test = get_data(N, Nte, num_step, data_mode)
    train_input_handle, test_input_handle = data_provider(
        args.data_name, args.train_data_paths, args.valid_data_paths, args.bs, args.img_width,
        seq_length=args.num_step + args.num_past + 1, is_training=True, num_views=args.num_views, img_channel=args.img_channel,
        baseline=args.baseline, eval_batch_size=args.vis_bs, n_epoch=n_epoch, args=args)
    
    if args.message_type in ['raw_data']:
        input_dim = 3 + 3 + 3
    elif args.message_type in ['vae']:
        input_dim = 3 + args.vae_latent_dim + args.vae_latent_dim
    else:
        input_dim = 3+1+1
    # h_units = [input_dim, input_dim]
    h_units = [int(x) for x in args.num_hidden.split(',')]
    if (args.mode == 'eval' or args.mode == 'transfer_learning') and args.ckpt_dir is not None:
        model_0_path = os.path.join(args.ckpt_dir, "model_0.pt")
        model_1_path = os.path.join(args.ckpt_dir, "model_1.pt")
        model_0 = torch.load(model_0_path)
        model_1 = torch.load(model_1_path)
        print('Loaded model_0 from {}, model_1 from {}'.format(model_0_path, model_1_path))
    else:
        model_0 = SSTA_Net(input_dim, h_units, act, args)
        model_1 = SSTA_Net(input_dim, h_units, act, args)

        print('Created model_0, model_1')
    model_0 = model_0.to(args.device)
    model_1 = model_1.to(args.device)
    # vae = VanillaVAE(input_dim, h_units, act, args)
    vae_path = os.path.join(args.vae_ckpt_dir, 'vae.pt')
    vae = torch.load(vae_path)
    vae = vae.to(args.device)
    print('Loaded VAE model_0 from {}'.format(vae_path))
    

    optimizer = optim.Adam(list(model_0.parameters()) + list(model_1.parameters()),lr=0.0005)
    MSE = nn.MSELoss()

    tr_loss = []
    te_loss = []
    root_res_path = os.path.join(args.gen_frm_dir)
    if os.path.exists(os.path.join(root_res_path, "{}/Pred".format(act))) == False:
        os.makedirs(os.path.join(root_res_path, "{}/Pred".format(act)))

    start_time = time.time()
    print("START")
    best_eval_loss = np.inf
    total_final_loss=[]
    for epoch in range(1, n_epoch + 1):
        print("-----------------------",epoch,"------------------")
        if args.mode == 'train' or args.mode=="transfer_learning":
            model_0.train()
            model_1.train()
            sum_loss = 0
            print('Training ... {}'.format(epoch))
            train_input_handle.begin(do_shuffle=True)
            while (train_input_handle.no_batch_left() == False):
                ims = train_input_handle.get_batch()
                train_input_handle.next()
                x_batch = ims[:, :]
                gt_batch = ims[:, 1:]
                x_batch = torch.from_numpy(x_batch.astype(np.float32)).to(args.device)  # .reshape(x.shape[0], 1))
                gt_batch = torch.from_numpy(gt_batch.astype(np.float32)).to(args.device)  # .reshape(gt.shape[0], 1))
                optimizer.zero_grad()                
                pred_batch, message_batch = run_steps(x_batch, model_0, model_1, vae,
                                                      inference=False, args=args)
#                 print("SHAPEEE",pred_batch.shape,gt_batch.shape)
                loss = MSE(pred_batch, gt_batch)
                loss.backward()
                optimizer.step()
                sum_loss += loss.data * bs

            ave_loss = sum_loss / (N - args.num_step)
            tr_loss.append(ave_loss.cpu())
            train_stats = {'ave_loss': ave_loss}
            print("loss",loss)
        
        # print(args.mode ,epoch,epoch % 20 )
        if epoch %  2== 0 or args.mode=="eval":  # 20
            batch_id = 0
            res_path = os.path.join(root_res_path, "evaluating_SSTA_2view_images")
            res_path = os.path.join(res_path, str(epoch))
            os.makedirs(res_path, exist_ok=True)
            avg_mse = 0
            img_mse, ssim, psnr = [], [], []
            lp = []
            for i in range(args.eval_num_step):
                img_mse.append(0)
                ssim.append(0)
                psnr.append(0)
                lp.append(0)
            if args.eval_mode != 'multi_step_eval':
                for i in range(args.num_past):
                    img_mse.append(0)
                    ssim.append(0)
                    psnr.append(0)
                    lp.append(0)
            test_input_handle.begin(do_shuffle=False)

            # print("batch",test_input_handle.no_batch_left())
            while (test_input_handle.no_batch_left() == False):
                # print("XXXXXXXXXXXXXXXXinsideeval")
                batch_id = batch_id + 1
                ims = test_input_handle.get_batch()
                test_input_handle.next()
                x_test = ims[:, :]
                if args.eval_mode == 'multi_step_eval':
                    t_test = ims[:, 1 + args.num_past:]
                else:
                    t_test = ims[:, 1:]
#                 print("innn",x_test.shape)
                with torch.no_grad():
                    y_test, message = run_steps(torch.from_numpy(x_test.astype(np.float32)).to(args.device),
                                                model_0, model_1, vae,
                                                inference=True,
                                                args=args,)
                
                y_test = y_test.detach().cpu().numpy()
                message = message.detach().cpu().numpy()
                # print('message.shape: ', message.shape) # (10, 10, 128, 128, 2)
                # MSE per frame
                # print('y_test.shape: {}, t_test.shape: {}'.format(y_test.shape, t_test.shape))
                for i in range(y_test.shape[1]):
                    x = y_test[:, i, :, :, :]
                    gx = t_test[:, i, :, :, :]
                    mse = np.square(x - gx).mean()
                    img_mse[i] += mse
                    avg_mse += mse
                    # cal lpips
                    img_x = np.zeros([y_test.shape[0], 3, y_test.shape[2], y_test.shape[3]])

                    img_x[:, 0, :, :] = x[:, :, :, 0]
                    img_x[:, 1, :, :] = x[:, :, :, 1]
                    img_x[:, 2, :, :] = x[:, :, :, 2]

                    img_x = torch.FloatTensor(img_x)
                    img_gx = np.zeros([y_test.shape[0], 3, y_test.shape[2], y_test.shape[3]])

                    img_gx[:, 0, :, :] = gx[:, :, :, 0]
                    img_gx[:, 1, :, :] = gx[:, :, :, 1]
                    img_gx[:, 2, :, :] = gx[:, :, :, 2]

                    img_gx = torch.FloatTensor(img_gx)
                    lp_loss = loss_fn_alex(img_x, img_gx)
                    lp[i] += torch.mean(lp_loss).item()

                    real_frm = np.uint8(x * 255)
                    pred_frm = np.uint8(gx * 255)

                    psnr[i] += batch_psnr(pred_frm, real_frm)
                    for b in range(y_test.shape[0]):
                        # score, _ = compare_ssim(pred_frm[b], real_frm[b], full=True, multichannel=True)
                        score = 0
                        ssim[i] += score
                # save prediction examples
                if batch_id <= args.num_save_samples:
                    path = os.path.join(res_path, str(batch_id))
                    os.mkdir(path)
                    for view_idx in range(args.num_views):
                        for i in range(y_test.shape[1]):
                            name = 'gt_{0:02d}_{1:02d}.png'.format(i + 1, view_idx)
                            file_name = os.path.join(path, name)
                            img_gt = np.uint8(t_test[0, i, :, :, (view_idx * args.img_channel):(
                                        (view_idx + 1) * args.img_channel)] * 255)
                            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
                            cv2.imwrite(file_name, img_gt)
                        for i in range(y_test.shape[1]):
                            name = 'pd_{0:02d}_{1:02d}.png'.format(i + 1, view_idx)
                            file_name = os.path.join(path, name)
                            img_pd = y_test[0, i, :, :,
                                     (view_idx * args.img_channel):((view_idx + 1) * args.img_channel)]
                            # in range (0, 1)
                            img_pd = np.uint8(img_pd * 255)
                            img_pd = cv2.cvtColor(img_pd, cv2.COLOR_BGR2RGB)
                            cv2.imwrite(file_name, img_pd)
                        for i in range(y_test.shape[1]):
                            name = 'msg_{0:02d}_{1:02d}.png'.format(i + 1, view_idx)
                            file_name = os.path.join(path, name)
                            # print('message.shape: ', message.shape) #  (10, 10, 128, 128, 2) or (10, 10, 128, 128, 6)
                            if args.message_type in ['raw_data']:
                                img_pd = message[0, i, :, :,
                                         (view_idx * 1) * 3:((view_idx + 1) * 3)]
                            else:
                                img_pd = message[0, i, :, :,
                                     (view_idx * 1):((view_idx + 1) * 1)]
                            # in range (0, 1)
                            # print('img_pd.shape: ', img_pd.shape, img_pd.max(), img_pd.min()) # img_pd.shape:  (128, 128, 1)
                            img_pd = np.uint8(img_pd * 255)
                            cv2.imwrite(file_name, img_pd)

                avg_mse = np.mean(img_mse)
                # print('mse per seq: ' + str(avg_mse))
                # for i in range(args.num_step):
                #     print('step {}, mse: {}'.format(i, img_mse[i]))

                # ssim = np.asarray(ssim, dtype=np.float32)
                # print(finalpsnr per frame: ' + str(np.mean(psnr)))
                # for i in range(args.num_step):
                #     print('step {}, psnr: {}'.format(i, psnr[i]))

                # lp = np.asarray(lp, dtype=np.float32)
                # print('lpips per frame: ' + str(np.mean(lp)))
                # for i in range(args.num_step):
                #     print('step {}, lp: {}'.format(i, lp[i]))
                # print('Save to {}.'.format(res_path))

        if args.mode == 'train' or args.mode=="transfer_learning":
            train_model_save_path=os.path.join(root_res_path,"SSTA_model_0_1")
            train_model_save_path=os.path.join(train_model_save_path,str(epoch))
            os.makedirs(train_model_save_path, exist_ok=True)
            print(train_model_save_path,"XXXXXXXXXXXXXXXXSAVINGMODELVAEXXXXXXXXXXXXXXXXXXXXXXXXXX")
            torch.save(model_0, os.path.join(train_model_save_path, "model_0.pt"))
            torch.save(model_1, os.path.join(train_model_save_path, "model_1.pt"))
            total_final_loss.append([epoch,loss,","])
        
            print("FINAL LOSS",total_final_loss)
        if args.mode=="eval":
            break
    print("END")

    total_time = int(time.time() - start_time)
    print("Time : {} [s]".format(total_time))

    plt.figure(figsize=(5, 4))
    plt.plot(tr_loss, label="training")
    plt.plot(te_loss, label="test")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("loss (MSE)")
    plt.savefig(os.path.join(root_res_path, "{}/loss_history.png".format(act)))
    plt.clf()
    plt.close()

def batch_psnr(gen_frames, gt_frames):
    if gen_frames.ndim == 3:
        axis = (1, 2)
    elif gen_frames.ndim == 4:
        axis = (1, 2, 3)
    x = np.int32(gen_frames)
    y = np.int32(gt_frames)
    num_pixels = float(np.size(gen_frames[0]))
    mse = np.sum((x - y) ** 2, axis=axis, dtype=np.float32) / num_pixels
    psnr = 20 * np.log10(255) - 10 * np.log10(mse)
    return np.mean(psnr)

if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        

    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--data_mode', type=str, default="(y_{t-1}, y_t)->y_{t+1}", help='(y_{t-1}, y_t)->y_{t+1}')
    parser.add_argument('--act', type=str, default="relu", help='relu')
    parser.add_argument('--mode', type=str, default="train", help='train / eval/transfer_learning')
    parser.add_argument('--eval_mode', type=str, default='multi_step_eval', help='multi_step_eval / single_step_eval')
    parser.add_argument('--eval_num_step', type=int, default=30)
    parser.add_argument('--eval_per_step', type=int, default=5)
    parser.add_argument('--mask_per_step', type=int, default=1000000000)
    parser.add_argument('--log_per_epoch', type=int, default=10)
    parser.add_argument('--num_step', type=int, default=15)
    parser.add_argument('--num_past', type=int, default=2)
    parser.add_argument('--num_cl_step', type=int, default=25)
    parser.add_argument('--n_epoch', type=int, default=80, help='200')
    parser.add_argument('--bs', type=int, default=12)
    parser.add_argument('--vis_bs', type=int, default=5)
    parser.add_argument('--Nte', type=int, default=20, help='200')
    parser.add_argument('--N', type=int, default=100, help='1000')

    parser.add_argument('--data_name', type=str, default="carla_town02_8_view_20220713_color_split_2", help='SINE; circle_motion; students003, fluid_flow_1')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda:0 cuda:0; cpu:0 cpu:0')
    parser.add_argument('--with_comm', type=str2bool, default=False, help='whether to use communication')
    # parser.add_argument('--train_data_paths', type=str, default="../../../../tools/circle_motion_30/train",
                        # help='../tools/${DATASET_NAME}/train, ../../../../tools/circle_motion_30/train, sumo_sanjose-2021-11-06_20.28.57_30, carla_town02_20211201, students003')
    # carla_town02_20211201
    # parser.add_argument('--valid_data_paths', type=str, default="../../../../tools/circle_motion_30/eval",
                        # help='../tools/${DATASET_NAME}/eval, ../../../../tools/circle_motion_30/eval, sumo_sanjose-2021-11-06_20.28.57_30, carla_town02_20211201, students003')
    # sumo_sanjose-2021-11-06_20.28.57_30if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
    # RGB dataset
    parser.add_argument('--img_width', type=int, default=128, help='img width')
    parser.add_argument('--num_views', type=int, default=2, help='num views')
    parser.add_argument('--img_channel', type=int, default=3, help='img channel')
    parser.add_argument('--baseline', type=str, default='1_NN_4_img_GCN',
                        help='1_NN_1_img_no_GCN, 1_NN_4_img_no_GCN, 4_NN_4_img_GCN, 1_NN_4_img_GCN, 4_NN_4_img_no_GCN, '
                             '4_NN_4_img_FC, 4_NN_4_img_Identity')
    parser.add_argument('--gen_frm_dir', type=str, default='/home/sgarikipati7/SSTA_2024/preds_vaessta')
    parser.add_argument('--num_save_samples', type=int, default=10)
    parser.add_argument('--layer_norm', type=int, default=1)
    parser.add_argument('--num_hidden', type=str, default='16,16', help='64,64,64,64')
    parser.add_argument('--filter_size', type=int, default=5)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--version', type=str, default='predrnn', help='version')
    parser.add_argument('--message_type', type=str, default='vae', help='normal, zeros, randn, raw_data, vae')
    
    parser.add_argument('--ckpt_dir', type=str, default='/storage/home/hcoda1/9/knagaraj31/p-skousik3-0/SSTA_2022/sim_SSTA2_files/ssta2_sim_model_16_8/SSTA_model_0_1/5', help='checkpoint dir: dir/model_1.pt, 20220717-195317')
    parser.add_argument('--vae_ckpt_dir', type=str, default="/home/sgarikipati7/SSTA_2024/vae_file",
                        help='vae checkpoint dir: kessel: 1c-results/20220109-212140/circle_motion/vae.pt, '
                             '20220110-234817, chpc-gpu005: 1c-20220113-154323, wo ln: 20220117-044713')
    parser.add_argument('--cl_mode', type=str, default='full_history', help='full_history, sliding_window')
    parser.add_argument('--vae_latent_dim', type=int, default=4)
    parser.add_argument('--train_data_paths', type=str, default=r"/home/sgarikipati7/SSTA_2024/DATASET/test")
    parser.add_argument('--valid_data_paths', type=str, default=r"/home/sgarikipati7/SSTA_2024/DATASET/val")

    args = parser.parse_args()

    h_units = [10,10]
#     timestr = time.strftime("%Y%m%d-%H%M%S")
    args.gen_frm_dir = os.path.join(args.gen_frm_dir)
#     args.train_data_paths = "/storage/home/hcoda1/9/knagaraj31/p-skousik3-0/train".format(args.data_name)
# ##     for validation
#     args.valid_data_paths = "/storage/home/hcoda1/9/knagaraj31/p-skousik3-0/val".format(args.data_name)
# #     for testing:
#     args.valid_data_paths = "/storage/home/hcoda1/9/knagaraj31/p-skousik3-0/test".format(args.data_name)
    training(args.N, args.Nte, args.bs, args.n_epoch, args.act, args.data_mode, args)
