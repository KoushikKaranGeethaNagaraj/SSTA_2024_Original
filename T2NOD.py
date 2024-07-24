'''
https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_video/py_bg_subtraction/py_bg_subtraction.html
'''
import numpy as np
import cv2, os, json
import queue
import moviepy.editor as mpy
import matplotlib.pyplot as plt
import pdb
from PIL import Image
import gc
import sys

# Function to extract the integer part from the filename for sorting
def numerical_sort(filename):
    parts = filename.split('_')  # Assuming the filename format is "image_X.jpg"
    number = int(parts[1].split('.')[0])  # Extracts the integer value X from "image_X.jpg"
    return number

def T2NO(input_video_name='/storage/home/hcoda1/9/knagaraj31/p-skousik3-0/T2NOD/background_test_dataset/val', num_test=150, first_skip=0, thres=70., his_length=20, alg='MOG', mode='static', vis=False, base_idx=0):
    '''

    mode: static / mean
    thres: for static mode: 70.; for mean mode: 30.;
    alg = 'MOG'  # MOG, MOG2, GMG #default =KNN
    '''
    
    #################### saving directories ########################################
    save_dir = input_video_name + '_{}_t2no_{:02d}'.format(alg, his_length)
    save_rgb_dir = input_video_name + '_{}_rgb_{:02d}'.format(alg, his_length)
    bg_mask_dir = input_video_name + '_{}_mask'.format(alg)
    input_video_paths = sorted(os.listdir(input_video_name))
    ##############iterating between cameras###################################
    for input_video_path in input_video_paths:
        # First one camera at a time 
        input_video_path = [input_video_path]
        rgb_frames = {}
        all_img_list = []
        for each_view in input_video_path:
            print(each_view)
            rgb_frames[each_view] = []
            for i, each_img in enumerate(sorted(os.listdir(os.path.join(input_video_name, each_view)),key=numerical_sort)):
                if i < first_skip:
                    continue
                rgb_frames[each_view].append(each_img)
                if mode == 'mean':
                    all_img_list.append(cv2.cvtColor(cv2.imread(os.path.join(input_video_name, each_view, each_img)).astype(np.uint8), cv2.COLOR_BGR2GRAY))
                if num_test is not None and i >= num_test:
                    break
        #the all_img_list has each camera view total images to get background image
        #saving the t2NO directories
        print("Saving Directories")
        os.makedirs(save_dir, exist_ok=True)
#         os.makedirs(save_rgb_dir, exist_ok=True)
                               
        ##################### initialising for T2NOD####################                       
        frame_idx = 0
        bg_error = {}
        q_E = queue.Queue(his_length)
        q_E_t2nd = queue.Queue(his_length)

        ######################iterate over each camera view######################
        for each_view, value_path in sorted(rgb_frames.items()):
            os.makedirs(os.path.join(save_dir, each_view), exist_ok=True)
#             os.makedirs(os.path.join(save_rgb_dir, each_view), exist_ok=True)
            #iterate over each image
            print(each_view,"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            for value_idx, each_img in sorted(enumerate(value_path)):
                print('each_img: ', each_view, each_img)
                frame = cv2.imread(os.path.join(input_video_name, each_view, each_img)).astype(np.uint8)
                               
                ######based on your length of prediction the images will be skipped for saving but T2nod will be computed and added in the queue
                if int(each_img[6:-4]) - his_length > 0:
                    print("inside saving")
                    save_idx = '{0:08d}.png'.format(int(each_img[6:-4]) - his_length + base_idx)
                    frame_input = cv2.imread(os.path.join(input_video_name, each_view, each_img))
                    #If you want to save the original rgb images
#                     cv2.imwrite(os.path.join(save_rgb_dir, each_view, save_idx), frame_input)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(frame, (128, 128))
                
#########################preprocessing the image from RGB to T2NOD to first obtain background image################################
                if value_idx == 0:
                    if mode == 'static':
                        print(os.path.join("/storage/home/hcoda1/9/knagaraj31/p-skousik3-0/T2NOD/background_test_dataset/background_image",str(each_view)+ '_background_image.jpg'))
                        B = cv2.imread(os.path.join("/storage/home/hcoda1/9/knagaraj31/p-skousik3-0/T2NOD/background_test_dataset/background_image",str(each_view)+ '_background.jpg')).astype(np.uint8)
                        B = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
                    #creation of the background image
                    elif mode == 'mean':
                        print('len(all_img_list): ', np.array(all_img_list).shape)
                        B = np.mean(np.array(all_img_list), axis=0).astype(np.uint8)
                        cv2.imwrite("/storage/home/hcoda1/9/knagaraj31/p-skousik3-0/T2NOD/background_test_dataset/back.jpg", B)
                    else:
                        print("No background image method")
                    B = cv2.resize(B, (128, 128))              

#########################Beginning T2NOD process################################
                absdiff = cv2.absdiff(B, frame) # range of absdiff: [0, 255]
                diff_img_T = (cv2.absdiff(B, frame) > thres) * 255.  # [False, True]
                diff_img_t2nd_T = (cv2.absdiff(B, frame) < thres) * 255.  # [False, True]
                while not q_E.full():
                    q_E.put(diff_img_T)
                while not q_E_t2nd.full():
                    q_E_t2nd.put(diff_img_t2nd_T)

                q_E_t2no_array = np.asarray(
                    [ele for ele in list(q_E.queue)] + [np.ones_like(diff_img_T) * 255.])
                t2no_img = np.argmax(q_E_t2no_array, axis=0)

                q_E_t2nd_array = np.asarray([ele for ele in list(q_E_t2nd.queue)])  # + [np.ones_like(diff_img_t2nd)])
                t2nd_img = np.argmax(q_E_t2nd_array, axis=0)
                infty_mask = np.logical_or((np.abs(t2no_img - his_length) < 1e-2), (np.abs(t2nd_img) < 1e-2))
                t2nd_img[infty_mask] = his_length
                # t2no_img = q_E_t2nd_array.shape[0] - t2nd_img

                q_E.get()
                diff_img_t2nd_t = q_E_t2nd.get()
                # t2nd_img = (t2nd_img * 255 / his_length).astype('uint8')
                # t2no_img = (t2no_img * 255 / his_length).astype('uint8')
                if vis:
                    t2nd_img_vis = (t2nd_img * 255. / his_length).astype('uint8')
                    t2no_img_vis = (t2no_img * 255. / his_length).astype('uint8')
                    # t2nd_img = (t2nd_img * 255 / his_length).astype('uint8')
                    # t2no_img = (t2no_img * 255 / his_length).astype('uint8')
                    concat_img = np.concatenate(
                        [t2no_img_vis, np.zeros_like(t2no_img_vis[:, :4]), t2nd_img_vis], axis=1)
                bg_error[each_img] = np.sum(diff_img_T).tolist()
                # print(fgmask.shape, fgmask[:100]) # (600, 600)
                # cv2.imwrite(os.path.join(save_dir, each_view, 'diff_'+each_img,), diff_img)
                # cv2.imwrite(os.path.join(save_dir, each_view, 'diffT2ND_' + each_img, ), diff_img_t2nd)
                               
 ######based on your length of prediction the images will be skipped for saving but T2nod will be computed and added in the queue
                if int(each_img[6:-4]) - his_length > 0:
                    save_idx = '{0:08d}.png'.format(int(each_img[6:-4]) - his_length + base_idx)
                    cv2.imwrite(os.path.join(save_dir, each_view, 't2no_' + save_idx, ), t2no_img)
                # To save T2Nd also
                    cv2.imwrite(os.path.join(save_dir, each_view, 't2nd_' + save_idx, ), t2nd_img)
                    if vis:
#                         frame_resize=cv2.resize(frame_input, (128, 128)) 
#                         t2no_img_vis=t2no_img_vis.reshape(t2no_img_vis.shape[0],t2no_img_vis.shape[1],1)
#                         t2no_img_vis = cv2.cvtColor(t2no_img_vis, cv2.COLOR_GRAY2BGR)
#                         print(frame_resize.shape,t2no_img_vis.shape)
#                         concatenate_t2n0_rgb=np.hstack([frame_resize,t2no_img_vis])
                        cv2.imwrite(os.path.join(save_dir, each_view, 'vis_t2no_' + save_idx, ), t2no_img_vis)
            # To save T2Nd also
#                         cv2.imwrite(os.path.join(save_dir, each_view, 'vis_t2nd_' + save_idx, ), t2nd_img_vis)
                frame_idx += 1
            
        with open(os.path.join(save_dir, 'diff_error.json'), 'w') as f:
            json.dump(bg_error, f)

if __name__ == "__main__":
    T2NO(input_video_name="/storage/home/hcoda1/9/knagaraj31/p-skousik3-0/T2NOD/DATASET/train",
            mode='static', num_test=5000, first_skip=0, vis=True, base_idx=0, his_length=25)
   