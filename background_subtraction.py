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

def T2NO(input_video_name='carla_town02', num_test=110, first_skip=100, thres=70., his_length=20, alg='KNN', mode='static', vis=False, base_idx=0):
    '''

    mode: static / mean
    thres: for static mode: 70.; for mean mode: 30.;
    alg = 'MOG'  # MOG, MOG2, GMG
    '''
    # input_video_name = 'carla_town02'
    save_dir = input_video_name + '_{}_t2no_{:02d}'.format(alg, his_length)
    save_rgb_dir = input_video_name + '_{}_rgb_{:02d}'.format(alg, his_length)
    bg_mask_dir = input_video_name + '_{}_mask'.format(alg)
    input_video_paths = sorted(os.listdir(input_video_name))
    print('input_video_path: ', input_video_paths)
    for input_video_path in input_video_paths:
        input_video_path = [input_video_path]
        rgb_frames = {}
        all_img_list = []
        for each_view in input_video_path:
            rgb_frames[each_view] = []
            for i, each_img in enumerate(sorted(os.listdir(os.path.join(input_video_name, each_view)))):
                if i < first_skip:
                    continue
                rgb_frames[each_view].append(each_img)
                if mode == 'mean':
                    all_img_list.append(cv2.cvtColor(cv2.imread(os.path.join(input_video_name, each_view, each_img)).astype(np.uint8), cv2.COLOR_BGR2GRAY))
                if num_test is not None and i >= num_test:
                    break

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(save_rgb_dir, exist_ok=True)
        frame_idx = 0
        bg_error = {}

        q_E = queue.Queue(his_length)
        q_E_t2nd = queue.Queue(his_length)
        for each_view, value_path in rgb_frames.items():
            os.makedirs(os.path.join(save_dir, each_view), exist_ok=True)
            os.makedirs(os.path.join(save_rgb_dir, each_view), exist_ok=True)
            for value_idx, each_img in enumerate(value_path):
                print('each_img: ', each_img)
                frame = cv2.imread(os.path.join(input_video_name, each_view, each_img)).astype(np.uint8)
                if int(each_img[:8]) - his_length > 0:
                    save_idx = '{0:08d}.png'.format(int(each_img[:8]) - his_length + base_idx)
                    frame_input = cv2.imread(os.path.join(input_video_name, each_view, each_img))
                    cv2.imwrite(os.path.join(save_rgb_dir, each_view, save_idx), frame_input)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(frame, (128, 128))
                print('frame.max(): {}, frame.min(): {}'.format(frame.max(), frame.min()))
                if value_idx == 0:
                    if mode == 'static':
                        if input_video_path[0] == '_out_0':
                            # B = cv2.imread(os.path.join(input_video_name, each_view, '00000240.png')).astype(np.uint8)
                            # B = cv2.imread(os.path.join(input_video_name, each_view, '00000266.png')).astype(np.uint8)
                            B = cv2.imread(os.path.join(input_video_name, each_view, '00000012.png')).astype(np.uint8)
                        elif input_video_path[0] == '_out_1':
                            # B = cv2.imread(os.path.join(input_video_name, each_view, '00000240.png')).astype(np.uint8)
                            B = cv2.imread(os.path.join(input_video_name, each_view, '00000012.png')).astype(np.uint8)
                        elif input_video_path[0]  == '_out_2':
                            B = cv2.imread(os.path.join(input_video_name, each_view, '00000012.png')).astype(np.uint8)
                        elif input_video_path[0] == '_out_3':
                            # B = cv2.imread(os.path.join(input_video_name, each_view, '00000240.png')).astype(np.uint8)
                            B = cv2.imread(os.path.join(input_video_name, each_view, '00000012.png')).astype(np.uint8)
                        B = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
                        print(B.shape)
                    elif mode == 'mean':
                        print('len(all_img_list): ', np.array(all_img_list).shape)
                        B = np.mean(np.array(all_img_list), axis=0).astype(np.uint8)
                        print(B.shape)
                    else:
                        bg_mask = cv2.imread(os.path.join(bg_mask_dir, each_view, each_img)).astype(np.uint8)
                        inverse_mask = cv2.bitwise_not(bg_mask).astype(np.uint8)
                        print('inverse_mask.shape: ', inverse_mask.shape, frame.shape)
                        # B = frame * inverse_mask
                        B = cv2.add(frame, np.zeros(np.shape(frame), dtype=np.uint8), mask=inverse_mask.max(-1))
                        # B = frame[inverse_mask]
                    B = cv2.resize(B, (128, 128))
                # print('B.shape, frame.shape: ', B.shape, frame.shape, 'cv2.absdiff(B, frame).max(): ', np.mean(cv2.absdiff(B, frame)))
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
                print('self.step_index: {}, (t2no_img.min(): {}, t2no_img.max(): {})'.format(0,
                                                                                             t2no_img.min(),
                                                                                             t2no_img.max()))  # 1, 2

                q_E_t2nd_array = np.asarray([ele for ele in list(q_E_t2nd.queue)])  # + [np.ones_like(diff_img_t2nd)])
                t2nd_img = np.argmax(q_E_t2nd_array, axis=0)
                print('self.step_index: {}, (t2nd_img.min(): {}, t2nd_img.max(): {})'.format(0,
                                                                                             t2nd_img.min(),
                                                                                             t2nd_img.max()))  # 1, (0, his_length)
                infty_mask = np.logical_or((np.abs(t2no_img - his_length) < 1e-2), (np.abs(t2nd_img) < 1e-2))
                # print('np.sum(infty_mask): ', np.sum(infty_mask))
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
                    # plt.figure(figsize=(30 * 4, 30))
                    print(concat_img.max(), concat_img.min())
                    # pdb.set_trace()
                    # plt.imshow(concat_img, cmap='gray', )
                    # plt.imshow(concat_img, cmap='gray', )
                    # plt.show()
                # print('t2no_img.shape: ', t2no_img.shape, 'diff_img.shape: ', diff_img.shape,
                # 'diff_img.max(): ', diff_img.max(), 'diff_img.min(): ', diff_img.min())
                bg_error[each_img] = np.sum(diff_img_T).tolist()
                # print(fgmask.shape, fgmask[:100]) # (600, 600)
                # cv2.imwrite(os.path.join(save_dir, each_view, 'diff_'+each_img,), diff_img)
                # cv2.imwrite(os.path.join(save_dir, each_view, 'diffT2ND_' + each_img, ), diff_img_t2nd)
                if int(each_img[:8]) - his_length > 0:
                    save_idx = '{0:08d}.png'.format(int(each_img[:8]) - his_length + base_idx)
                    cv2.imwrite(os.path.join(save_dir, each_view, 't2no_' + save_idx, ), t2no_img)
                    if vis:
                        cv2.imwrite(os.path.join(save_dir, each_view, 'vis_t2no_' + save_idx, ), t2no_img_vis)
                        print('t2no_img_vis.shape: ', t2no_img_vis.shape)
                        # img = Image.fromarray(t2no_img_vis)
                        # img.save(os.path.join(save_dir, each_view, 'vis_t2no_' + save_idx + 'imageio.png', ))
                        # img.show()
                        # # vis with plt
                        # fig = plt.figure(frameon=False)
                        # fig.set_size_inches(128, 128)
                        # ax = plt.Axes(fig, [0., 0., 1., 1.])
                        # ax.set_axis_off()
                        # fig.add_axes(ax)
                        # ax.imshow(t2no_img_vis, aspect='auto')
                        # fig.savefig(os.path.join(save_dir, each_view, 'vis_t2no_' + save_idx + 'plt.png', ))
                        # # Clear the current axes.
                        # plt.cla()
                        # # Clear the current figure.
                        # plt.clf()
                        # # Closes all the figure windows.
                        # plt.close('all')
                        # plt.close(fig)
                        # gc.collect()

                    # cv2.imwrite(os.path.join(save_dir, each_view, 't2nd_' + each_img, ), t2nd_img)
                    # cv2.imwrite(os.path.join(save_dir, each_view, 'frame_' + each_img, ), frame)
                    if mode not in ['static', 'mean']:
                        cv2.imwrite(os.path.join(save_dir, each_view, 'mask_' + save_idx, ), inverse_mask)
                    # cv2.imwrite(os.path.join(save_dir, each_view, 'bg_' + each_img, ), B)
                    # print('Save to {}, diff error: {}'.format(os.path.join(save_dir, each_view, each_img,), bg_error))
                    print('Save to {}'.format(os.path.join(save_dir, each_view, save_idx, )))
                # all_fgmask.append(fgmask)
                frame_idx += 1
        with open(os.path.join(save_dir, 'diff_error.json'), 'w') as f:
            json.dump(bg_error, f)
        # all_fgmask_np = np.asarray(all_fgmask)
        # np.save('{}.npy'.format(input_video_name), all_fgmask_np)
        # print('Save to {}'.format('{}.npy'.format(input_video_name)))

def convert_cv2_plt(input_video_name='/home/jsun/data/Programs/msl-traffic-prediction/packages/carla_tools/aug',
            save_idx=140):
    # save_idx =525
    alg = 'KNN'
    his_lengths = [2, 10, 20, 30, 60]
    save_dir = 'time_horizon'
    os.makedirs(save_dir, exist_ok=True)
    each_views = ['_out_0', '_out_1', '_out_2', '_out_3']

    for his_length in his_lengths:
        cur_save_idx = save_idx # + max(0, his_length - 12)
        each_img = 't2no_{:08d}.png'.format(cur_save_idx)
        input_dir = input_video_name + '_{}_t2no_{:02d}'.format(alg, his_length)
        for each_view in each_views:
            # for cv2_image_path in cv2_image_paths:
            t2no_img = cv2.imread(os.path.join(input_dir, each_view, each_img))
            print(t2no_img.shape, np.all(t2no_img[:, :, 0] == t2no_img[:, :, 1]), np.all(t2no_img[:, :, 1] == t2no_img[:, :, 2]))
            t2no_img = t2no_img[:, :, 0]
            t2no_img_vis = (t2no_img * 255. / his_length).astype('uint8')
            # print('Save to {}'.format(os.path.join(save_dir, 'vis_t2no_{}_{:02d}_{:08d}.png'.format(each_view, his_length, save_idx))))
            # cv2.imwrite(
            #     os.path.join(save_dir, 'vis_t2no_{}_{:02d}_{:08d}.png'.format(each_view, his_length, save_idx)), t2no_img_vis)
            fig = plt.figure(frameon=False)
            fig.set_size_inches(128, 128)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(t2no_img_vis, aspect='auto')
            print('Save to {}'.format(os.path.join(save_dir, 'vis_t2no_{}_{:02d}_{:08d}_plt.png'.format(each_view, his_length, cur_save_idx))))
            fig.savefig(os.path.join(save_dir, 'vis_t2no_{}_{:02d}_{:08d}_plt.png'.format(each_view, his_length, cur_save_idx)))
            # plt.show()
            # Clear the current axes.
            plt.cla()
            # Clear the current figure.
            plt.clf()
            # Closes all the figure windows.
            plt.close('all')
            plt.close(fig)
            gc.collect()

    input_dir = input_video_name
    each_img = '{:08d}.png'.format(save_idx)
    for each_view in each_views:
        rgb_img = cv2.imread(os.path.join(input_dir, each_view, each_img))
        print('Save to {}'.format(
            os.path.join(save_dir, 'vis_rgb_{}_{:08d}_plt.png'.format(each_view, save_idx))))
        cv2.imwrite(os.path.join(save_dir, 'vis_rgb_{}_{:08d}_plt.png'.format(each_view, save_idx)), rgb_img)
        if each_view == '_out_0':
            # B = cv2.imread(os.path.join(input_video_name, each_view, '00000240.png')).astype(np.uint8)
            ref_save_idx = 12
            B = cv2.imread(os.path.join(input_dir, each_view, '{:08d}.png'.format(ref_save_idx))).astype(np.uint8)
        elif each_view == '_out_1':
            # B = cv2.imread(os.path.join(input_video_name, each_view, '00000240.png')).astype(np.uint8)
            ref_save_idx = 12
            B = cv2.imread(os.path.join(input_dir, each_view, '{:08d}.png'.format(ref_save_idx))).astype(np.uint8)
        elif each_view == '_out_2':
            ref_save_idx = 12
            B = cv2.imread(os.path.join(input_dir, each_view, '{:08d}.png'.format(ref_save_idx))).astype(np.uint8)
        elif each_view == '_out_3':
            ref_save_idx = 12
            # B = cv2.imread(os.path.join(input_video_name, each_view, '00000240.png')).astype(np.uint8)
            B = cv2.imread(os.path.join(input_dir, each_view, '{:08d}.png'.format(ref_save_idx))).astype(np.uint8)
        print('Save to {}'.format(
            os.path.join(save_dir, 'vis_ref_{}_{:08d}_plt.png'.format(each_view, ref_save_idx))))
        B = cv2.resize(B, (128, 128))
        cv2.imwrite(os.path.join(save_dir, 'vis_ref_{}_{:08d}_plt.png'.format(each_view, ref_save_idx)), B)


def T2NO_v2(input_video_name='carla_town02', num_test=110, first_skip=100, thres=70., his_length=10, alg='KNN', mode='static', vis=False):
    '''

    mode: static / mean
    thres: for static mode: 70.; for mean mode: 30.;
    alg = 'MOG'  # MOG, MOG2, GMG
    '''
    # input_video_name = 'carla_town02'
    save_dir = input_video_name + '_{}_t2no'.format(alg)
    bg_mask_dir = input_video_name + '_{}_mask'.format(alg)
    input_video_path = sorted(os.listdir(input_video_name))
    input_video_path = [input_video_path[0]]
    rgb_frames = {}
    all_img_list = []
    for each_view in input_video_path:
        rgb_frames[each_view] = []
        for i, each_img in enumerate(sorted(os.listdir(os.path.join(input_video_name, each_view)))):
            if i < first_skip:
                continue
            rgb_frames[each_view].append(each_img)
            if mode == 'mean':
                all_img_list.append(cv2.cvtColor(cv2.imread(os.path.join(input_video_name, each_view, each_img)).astype(np.uint8), cv2.COLOR_BGR2GRAY))
            if num_test is not None and i >= num_test:
                break

    os.makedirs(save_dir, exist_ok=True)
    frame_idx = 0
    bg_error = {}

    q_E = queue.Queue(his_length)
    q_E_t2nd = queue.Queue(his_length)
    for each_view, value_path in rgb_frames.items():
        os.makedirs(os.path.join(save_dir, each_view), exist_ok=True)
        for value_idx, each_img in enumerate(value_path):
            frame = cv2.imread(os.path.join(input_video_name, each_view, each_img)).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (128, 128))
            print('frame.max(): {}, frame.min(): {}'.format(frame.max(), frame.min()))
            if value_idx == 0:
                if mode == 'static':
                    B = cv2.imread(os.path.join(input_video_name, each_view, '00000240.png')).astype(np.uint8)
                    B = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
                    print(B.shape)
                elif mode == 'mean':
                    print('len(all_img_list): ', np.array(all_img_list).shape)
                    B = np.mean(np.array(all_img_list), axis=0).astype(np.uint8)
                    print(B.shape)
                else:
                    bg_mask = cv2.imread(os.path.join(bg_mask_dir, each_view, each_img)).astype(np.uint8)
                    inverse_mask = cv2.bitwise_not(bg_mask).astype(np.uint8)
                    print('inverse_mask.shape: ', inverse_mask.shape, frame.shape)
                    # B = frame * inverse_mask
                    B = cv2.add(frame, np.zeros(np.shape(frame), dtype=np.uint8), mask=inverse_mask.max(-1))
                    # B = frame[inverse_mask]
                B = cv2.resize(B, (128, 128))
            # print('B.shape, frame.shape: ', B.shape, frame.shape, 'cv2.absdiff(B, frame).max(): ', np.mean(cv2.absdiff(B, frame)))
            absdiff = cv2.absdiff(B, frame) # range of absdiff: [0, 255]
            diff_img = (cv2.absdiff(B, frame) > thres) * 255.  # [False, True]
            diff_img_t2nd = (cv2.absdiff(B, frame) < thres) * 255.  # [False, True]
            while not q_E.full():
                q_E.put(diff_img)
            while not q_E_t2nd.full():
                q_E_t2nd.put(diff_img_t2nd)

            t2nd_img = np.argmax(np.asarray([ele for ele in reversed(list(q_E_t2nd.queue))]), axis=0) # * int(255 / his_length)
            q_E_t2nd_array = np.asarray([ele for ele in reversed(list(q_E_t2nd.queue))])
            t2no_img = q_E_t2nd_array.shape[0] - t2nd_img
            # t2nd_img = (t2nd_img * 255 / his_length).astype('uint8')
            # t2no_img = (t2no_img * 255 / his_length).astype('uint8')
            if vis:
                # t2nd_img = (t2nd_img * 255 / his_length).astype('uint8')
                # t2no_img = (t2no_img * 255 / his_length).astype('uint8')
                t2nd_img_vis = (t2nd_img * 255. / his_length).astype('uint8')
                t2no_img_vis = (t2no_img * 255. / his_length).astype('uint8')
                concat_img = np.concatenate(
                    [t2no_img_vis, np.zeros_like(t2no_img_vis[:, :4]), t2nd_img_vis], axis=1)
                # plt.figure(figsize=(30 * 4, 30))
                print(concat_img.max(), concat_img.min())
                # pdb.set_trace()
                # plt.imshow(concat_img, cmap='gray', )
                plt.imshow(concat_img, cmap='gray', )
                plt.show()
            # print('t2no_img.shape: ', t2no_img.shape, 'diff_img.shape: ', diff_img.shape,
            # 'diff_img.max(): ', diff_img.max(), 'diff_img.min(): ', diff_img.min())
            bg_error[each_img] = np.sum(diff_img).tolist()
            # print(fgmask.shape, fgmask[:100]) # (600, 600)
            # cv2.imwrite(os.path.join(save_dir, each_view, 'diff_'+each_img,), diff_img)
            # cv2.imwrite(os.path.join(save_dir, each_view, 'diffT2ND_' + each_img, ), diff_img_t2nd)
            cv2.imwrite(os.path.join(save_dir, each_view, 't2no_' + each_img, ), t2no_img)
            # cv2.imwrite(os.path.join(save_dir, each_view, 't2nd_' + each_img, ), t2nd_img)
            # cv2.imwrite(os.path.join(save_dir, each_view, 'frame_' + each_img, ), frame)
            if mode not in ['static', 'mean']:
                cv2.imwrite(os.path.join(save_dir, each_view, 'mask_' + each_img, ), inverse_mask)
            # cv2.imwrite(os.path.join(save_dir, each_view, 'bg_' + each_img, ), B)
            # print('Save to {}, diff error: {}'.format(os.path.join(save_dir, each_view, each_img,), bg_error))
            print('Save to {}'.format(os.path.join(save_dir, each_view, each_img, )))
            # all_fgmask.append(fgmask)
            frame_idx += 1
            q_E.get()
            q_E_t2nd.get()
    with open(os.path.join(save_dir, 'diff_error.json'), 'w') as f:
        json.dump(bg_error, f)
    # all_fgmask_np = np.asarray(all_fgmask)
    # np.save('{}.npy'.format(input_video_name), all_fgmask_np)
    # print('Save to {}'.format('{}.npy'.format(input_video_name)))

def T2NO_v1(input_video_name='carla_town02', num_test=110, first_skip=100, thres=70., his_length=10, alg='KNN', mode='static', vis=False):
    '''

    mode: static / mean
    thres: for static mode: 70.; for mean mode: 30.;
    alg = 'MOG'  # MOG, MOG2, GMG
    '''
    # input_video_name = 'carla_town02'
    save_dir = input_video_name + '_{}_t2no'.format(alg)
    bg_mask_dir = input_video_name + '_{}_mask'.format(alg)
    input_video_path = sorted(os.listdir(input_video_name))
    input_video_path = [input_video_path[0]]
    rgb_frames = {}
    all_img_list = []
    for each_view in input_video_path:
        rgb_frames[each_view] = []
        for i, each_img in enumerate(sorted(os.listdir(os.path.join(input_video_name, each_view)))):
            if i < first_skip:
                continue
            rgb_frames[each_view].append(each_img)
            if mode == 'mean':
                all_img_list.append(cv2.cvtColor(cv2.imread(os.path.join(input_video_name, each_view, each_img)).astype(np.uint8), cv2.COLOR_BGR2GRAY))
            if num_test is not None and i >= num_test:
                break

    os.makedirs(save_dir, exist_ok=True)
    # all_fgmask = []
    frame_idx = 0
    bg_error = {}

    q_E = queue.Queue(his_length)
    q_E_t2nd = queue.Queue(his_length)
    for each_view, value_path in rgb_frames.items():
        os.makedirs(os.path.join(save_dir, each_view), exist_ok=True)
        for value_idx, each_img in enumerate(value_path):
            frame = cv2.imread(os.path.join(input_video_name, each_view, each_img)).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print(frame.max(), frame.min())
            if value_idx == 0:
                if mode == 'static':
                    B = cv2.imread(os.path.join(input_video_name, each_view, '00000240.png')).astype(np.uint8)
                    B = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
                    print(B.shape)
                elif mode == 'mean':
                    print('len(all_img_list): ', np.array(all_img_list).shape)
                    B = np.mean(np.array(all_img_list), axis=0).astype(np.uint8)
                    print(B.shape)
                else:
                    bg_mask = cv2.imread(os.path.join(bg_mask_dir, each_view, each_img)).astype(np.uint8)
                    inverse_mask = cv2.bitwise_not(bg_mask).astype(np.uint8)
                    print('inverse_mask.shape: ', inverse_mask.shape, frame.shape)
                    # B = frame * inverse_mask
                    B = cv2.add(frame, np.zeros(np.shape(frame), dtype=np.uint8), mask=inverse_mask.max(-1))
                    # B = frame[inverse_mask]
            print('B.shape, frame.shape: ', B.shape, frame.shape, 'cv2.absdiff(B, frame).max(): ', np.mean(cv2.absdiff(B, frame)))
            absdiff = cv2.absdiff(B, frame) # range of absdiff: [0, 255]
            diff_img = (cv2.absdiff(B, frame) > thres) *255.  # [False, True]
            diff_img_t2nd = (cv2.absdiff(B, frame) < thres) *255.  # [False, True]
            # diff_img = (absdiff > np.mean(absdiff)) *255.  # [False, True]
            while not q_E.full():
                q_E.put(diff_img)
            while not q_E_t2nd.full():
                q_E_t2nd.put(diff_img_t2nd)
            # t2no_img = np.argmax(np.asarray([ele for ele in reversed(list(q_E.queue))]), axis=0)*int(255/his_length)
            if value_idx == 0:
                t2no_img = (np.argmax(np.asarray([ele for ele in list(q_E.queue)]), axis=0))*int(255/his_length)
            else:
                t2no_img_binary = np.argmax(np.asarray([ele for ele in list(q_E.queue)]), axis=0)
                # _, mask = cv2.threshold(t2no_img_binary, thresh=0, maxval=255, type=cv2.THRESH_BINARY)
                t2no_img = (t2no_img_binary > 0) * (t2no_img_binary - his_length + min(value_idx, his_length - 1) + 1) # * int(255 / his_length)

            t2nd_img = np.argmax(np.asarray([ele for ele in reversed(list(q_E_t2nd.queue))]), axis=0) # * int(255 / his_length)
            if vis:
                t2nd_img = (t2nd_img * 255 / his_length).astype('uint8')
                t2no_img = (t2no_img * 255 / his_length).astype('uint8')
            print('t2no_img.shape: ', t2no_img.shape, 'diff_img.shape: ', diff_img.shape,
            'diff_img.max(): ', diff_img.max(), 'diff_img.min(): ', diff_img.min())
            bg_error[each_img] = np.sum(diff_img).tolist()
            # print(fgmask.shape, fgmask[:100]) # (600, 600)
            cv2.imwrite(os.path.join(save_dir, each_view, 'diff_'+each_img,), diff_img)
            cv2.imwrite(os.path.join(save_dir, each_view, 'diffT2ND_' + each_img, ), diff_img_t2nd)
            cv2.imwrite(os.path.join(save_dir, each_view, 't2no_' + each_img, ), t2no_img)
            cv2.imwrite(os.path.join(save_dir, each_view, 't2nd_' + each_img, ), t2nd_img)
            cv2.imwrite(os.path.join(save_dir, each_view, 'frame_' + each_img, ), frame)
            if mode not in ['static', 'mean']:
                cv2.imwrite(os.path.join(save_dir, each_view, 'mask_' + each_img, ), inverse_mask)
            cv2.imwrite(os.path.join(save_dir, each_view, 'bg_' + each_img, ), B)
            # print('Save to {}, diff error: {}'.format(os.path.join(save_dir, each_view, each_img,), bg_error))
            print('Save to {}'.format(os.path.join(save_dir, each_view, each_img, )))
            # all_fgmask.append(fgmask)
            frame_idx += 1
            q_E.get()
            q_E_t2nd.get()
    with open(os.path.join(save_dir, 'diff_error.json'), 'w') as f:
        json.dump(bg_error, f)
    # all_fgmask_np = np.asarray(all_fgmask)
    # np.save('{}.npy'.format(input_video_name), all_fgmask_np)
    # print('Save to {}'.format('{}.npy'.format(input_video_name)))

def cv2_tracking(input_video_name='carla_town02', num_test=200, first_skip=100, thres=70., his_length=10, alg='KNN', mode='static'):
    '''

    mode: static / mean
    thres: for static mode: 70.; for mean mode: 30.;
    alg = 'MOG'  # MOG, MOG2, GMG
    '''
    # input_video_name = 'carla_town02'
    save_dir = input_video_name + '_{}_t2no'.format(alg)
    bg_mask_dir = input_video_name + '_{}_mask'.format(alg)
    input_video_path = sorted(os.listdir(input_video_name))
    input_video_path = [input_video_path[1]]
    rgb_frames = {}
    all_img_list = []
    for each_view in input_video_path:
        rgb_frames[each_view] = []
        for i, each_img in enumerate(sorted(os.listdir(os.path.join(input_video_name, each_view)))):
            if i < first_skip:
                continue
            rgb_frames[each_view].append(each_img)
            if mode == 'mean':
                all_img_list.append(cv2.cvtColor(cv2.imread(os.path.join(input_video_name, each_view, each_img)).astype(np.uint8), cv2.COLOR_BGR2GRAY))
            if num_test is not None and i >= num_test:
                break

    os.makedirs(save_dir, exist_ok=True)
    # all_fgmask = []
    frame_idx = 0
    bg_error = {}

    col_images = []
    q_E = queue.Queue(his_length)
    q_E_t2nd = queue.Queue(his_length)
    for each_view, value_path in rgb_frames.items():
        os.makedirs(os.path.join(save_dir, each_view), exist_ok=True)
        for value_idx, each_img in enumerate(value_path):
            frame = cv2.imread(os.path.join(input_video_name, each_view, each_img)).astype(np.uint8)
            col_images.append(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print(frame.max(), frame.min())
            if value_idx == 0:
                if mode == 'static':
                    B = cv2.imread(os.path.join(input_video_name, each_view, '00000240.png')).astype(np.uint8)
                    B = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
                    print(B.shape)
                elif mode == 'mean':
                    print('len(all_img_list): ', np.array(all_img_list).shape)
                    B = np.mean(np.array(all_img_list), axis=0).astype(np.uint8)
                    print(B.shape)
                else:
                    bg_mask = cv2.imread(os.path.join(bg_mask_dir, each_view, each_img)).astype(np.uint8)
                    inverse_mask = cv2.bitwise_not(bg_mask).astype(np.uint8)
                    print('inverse_mask.shape: ', inverse_mask.shape, frame.shape)
                    # B = frame * inverse_mask
                    B = cv2.add(frame, np.zeros(np.shape(frame), dtype=np.uint8), mask=inverse_mask.max(-1))
                    # B = frame[inverse_mask]
            print('B.shape, frame.shape: ', B.shape, frame.shape, 'cv2.absdiff(B, frame).max(): ', np.mean(cv2.absdiff(B, frame)))
            absdiff = cv2.absdiff(B, frame) # range of absdiff: [0, 255]
            diff_img = (cv2.absdiff(B, frame) > thres) *255.  # [False, True]
            diff_img_t2nd = (cv2.absdiff(B, frame) < thres) *255.  # [False, True]
            # diff_img = (absdiff > np.mean(absdiff)) *255.  # [False, True]
            while not q_E.full():
                q_E.put(diff_img)
            while not q_E_t2nd.full():
                q_E_t2nd.put(diff_img_t2nd)
            # t2no_img = np.argmax(np.asarray([ele for ele in reversed(list(q_E.queue))]), axis=0)*int(255/his_length)
            if value_idx == 0:
                t2no_img = (np.argmax(np.asarray([ele for ele in list(q_E.queue)]), axis=0))*int(255/his_length)
            else:
                t2no_img_binary = np.argmax(np.asarray([ele for ele in list(q_E.queue)]), axis=0)
                # _, mask = cv2.threshold(t2no_img_binary, thresh=0, maxval=255, type=cv2.THRESH_BINARY)
                t2no_img = (t2no_img_binary > 0) * (t2no_img_binary - his_length + min(value_idx, his_length - 1) + 1) * int(255 / his_length)

            t2nd_img = np.argmax(np.asarray([ele for ele in reversed(list(q_E_t2nd.queue))]), axis=0) * int(255 / his_length)
            print('t2no_img.shape: ', t2no_img.shape, 'diff_img.shape: ', diff_img.shape,
            'diff_img.max(): ', diff_img.max(), 'diff_img.min(): ', diff_img.min())
            bg_error[each_img] = np.sum(diff_img).tolist()
            # print(fgmask.shape, fgmask[:100]) # (600, 600)
            cv2.imwrite(os.path.join(save_dir, each_view, 'diff_'+each_img,), diff_img)
            cv2.imwrite(os.path.join(save_dir, each_view, 'diffT2ND_' + each_img, ), diff_img_t2nd)
            cv2.imwrite(os.path.join(save_dir, each_view, 't2no_' + each_img, ), t2no_img)
            cv2.imwrite(os.path.join(save_dir, each_view, 't2nd_' + each_img, ), t2nd_img)
            cv2.imwrite(os.path.join(save_dir, each_view, 'frame_' + each_img, ), frame)
            if mode not in ['static', 'mean']:
                cv2.imwrite(os.path.join(save_dir, each_view, 'mask_' + each_img, ), inverse_mask)
            cv2.imwrite(os.path.join(save_dir, each_view, 'bg_' + each_img, ), B)
            print('Save to {}, diff error: {}'.format(os.path.join(save_dir, each_view, each_img,), bg_error))
            # all_fgmask.append(fgmask)
            frame_idx += 1
            q_E.get()
            q_E_t2nd.get()
    with open(os.path.join(save_dir, 'diff_error.json'), 'w') as f:
        json.dump(bg_error, f)

    # kernel for image dilation
    kernel = np.ones((4, 4), np.uint8)

    # font style
    font = cv2.FONT_HERSHEY_SIMPLEX

    # directory to save the ouput frames
    pathIn = "contour_frames_3/"
    os.makedirs(pathIn, exist_ok=True)

    for i in range(len(col_images) - 1):

        # frame differencing
        grayA = cv2.cvtColor(col_images[i], cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(col_images[i + 1], cv2.COLOR_BGR2GRAY)
        diff_image = cv2.absdiff(grayB, grayA)

        # image thresholding
        ret, thresh = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)

        # image dilation
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        # find contours
        contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # shortlist contours appearing in the detection zone
        valid_cntrs = []
        for cntr in contours:
            x, y, w, h = cv2.boundingRect(cntr)
            if (x <= 200) & (y >= 80) & (cv2.contourArea(cntr) >= 25):
                if (y >= 90) & (cv2.contourArea(cntr) < 40):
                    break
                valid_cntrs.append(cntr)

        # add contours to original frames
        dmy = col_images[i].copy()
        cv2.drawContours(dmy, valid_cntrs, -1, (127, 200, 0), 2)

        cv2.putText(dmy, "vehicles detected: " + str(len(valid_cntrs)), (55, 15), font, 0.6, (0, 180, 0), 2)
        cv2.line(dmy, (0, 80), (256, 80), (100, 255, 255))
        cv2.imwrite(pathIn + str(i) + '.png', dmy)
        # all_fgmask_np = np.asarray(all_fgmask)

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[2]

    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()

    bbox_list = []
    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(col_images[0], False)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(col_images[0], bbox)

    for i in range(len(col_images) - 1):
        # Read a new frame
        frame = col_images[i]

        # Update tracker
        ok, bbox = tracker.update(frame)
        print('bbox: ', bbox)
        bbox_list.append(bbox)
        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(200) & 0xff
    np.save('bbox_{}.npy'.format(input_video_name), np.array(bbox_list))
    print('Save to {}'.format('bbox_{}.npy'.format(input_video_name)))

def image_background_subtractor(alg='MOG', input_video_name='carla_town02', num_test=200, first_skip=20):
    '''
    alg = 'BackgroundSubtractorMOG'  # BackgroundSubtractorMOG, BackgroundSubtractorMOG2, BackgroundSubtractorGMG
    '''
    # input_video_name = 'carla_town02'
    save_dir = input_video_name + '_{}_mask'.format(alg)
    input_video_path = sorted(os.listdir(input_video_name))
    rgb_frames = {}
    for each_view in input_video_path:
        rgb_frames[each_view] = []
        for i, each_img in enumerate(sorted(os.listdir(os.path.join(input_video_name, each_view)))):
            if i < first_skip:
                continue
            rgb_frames[each_view].append(each_img)
            if num_test is not None and i > num_test:
                break

    if alg == 'MOG':
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    elif alg == 'MOG2':
        fgbg = cv2.createBackgroundSubtractorMOG2() # detectShadows=False
    elif alg == 'GSOC':
        fgbg = cv2.bgsegm.createBackgroundSubtractorGSOC()
    elif alg == 'KNN':
        fgbg = cv2.createBackgroundSubtractorKNN()  # detectShadows=False
    elif alg == 'GMG':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

    os.makedirs(save_dir, exist_ok=True)
    all_fgmask = []
    frame_idx = 0
    for each_view, value_path in rgb_frames.items():
        os.makedirs(os.path.join(save_dir, each_view), exist_ok=True)
        for each_img in value_path:
            frame = cv2.imread(os.path.join(input_video_name, each_view, each_img))

            fgmask = fgbg.apply(frame)
            if alg == 'GMG':
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            # print(fgmask.shape, fgmask[:100]) # (600, 600)
            cv2.imwrite(os.path.join(save_dir, each_view, each_img,), fgmask)
            print('Save to {}'.format(os.path.join(save_dir, each_view, each_img,)))
            all_fgmask.append(fgmask)
            frame_idx += 1
    all_fgmask_np = np.asarray(all_fgmask)
    np.save('{}.npy'.format(input_video_name), all_fgmask_np)
    print('Save to {}'.format('{}.npy'.format(input_video_name)))


def video_background_subtractor(alg='MOG', input_video_name='sumo_sanjose_t-2021-10-30_22.18.58'):
    '''
    alg = 'BackgroundSubtractorMOG'  # BackgroundSubtractorMOG, BackgroundSubtractorMOG2, BackgroundSubtractorGMG
    '''
    # input_video_name = 'carla_town02'
    save_dir = input_video_name + '_mask'
    input_video_path = '{}.mp4'.format(input_video_name)
    cap = cv2.VideoCapture(input_video_path)

    if alg == 'MOG':
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    elif alg == 'MOG2':
        fgbg = cv2.createBackgroundSubtractorMOG2()
    elif alg == 'GSOC':
        fgbg = cv2.bgsegm.createBackgroundSubtractorGSOC()
    elif alg == 'KNN':
        fgbg = cv2.createBackgroundSubtractorKNN()  # detectShadows=False
    elif alg == 'GMG':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

    # os.makedir(input_video_name, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    all_fgmask = []
    frame_idx = 0
    while(1):
        ret, frame = cap.read()

        fgmask = fgbg.apply(frame)
        if alg == 'BackgroundSubtractorGMG':
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        cv2.imshow('frame', fgmask)
        cv2.imwrite(os.path.join(save_dir, '{:07d}.png'.format(frame_idx)), fgmask)
        print('Save to {}'.format(os.path.join(save_dir, '{:07d}.png'.format(frame_idx))))
        all_fgmask.append(fgmask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        frame_idx += 1
    all_fgmask_np = np.asarray(all_fgmask)
    np.save('{}.npy'.format(input_video_name), all_fgmask_np)
    print('Save to {}'.format('{}.npy'.format(input_video_name)))
    cap.release()
    cv2.destroyAllWindows()

def img2video(input_video_name='carla_town02', num_test=110, first_skip=100, thres=70., his_length=10, alg='KNN', mode='static'):
    '''

    mode: static / mean
    thres: for static mode: 70.; for mean mode: 30.;
    alg = 'MOG'  # MOG, MOG2, GMG
    '''
    input_video_path = sorted(os.listdir(input_video_name))
    view_0_folder = os.path.join(input_video_name, input_video_path[0])
    view_0 = sorted(os.listdir(view_0_folder))
    view_1_folder = os.path.join(input_video_name, input_video_path[1])
    view_1 = sorted(os.listdir(view_1_folder))
    min_length = min(len(view_0), len(view_1))
    frame_list = []
    for temporal0, temporal1 in zip(view_0[:min_length], view_1[:min_length]):
        view_0_img = os.path.join(view_0_folder, temporal0)
        print('Loading img from {}.'.format(view_0_img))
        BGRimage_0 = cv2.imread(view_0_img)
        RGBimage_0 = cv2.cvtColor(BGRimage_0, cv2.COLOR_BGR2RGB)

        view_1_img = os.path.join(view_1_folder, temporal1)
        print('Loading img from {}.'.format(view_1_img))
        BGRimage_1 = cv2.imread(view_1_img)  # (600, 600, 3)
        RGBimage_1 = cv2.cvtColor(BGRimage_1, cv2.COLOR_BGR2RGB)
        frame = np.concatenate([RGBimage_0, np.ones_like(RGBimage_0)[:, :4],
                                RGBimage_1], axis=1)
        frame_list.append(frame)

    clip = mpy.ImageSequenceClip(frame_list, fps=8)
    clip.write_videofile(os.path.join('vis_waypoint_dataset.mp4'.format()), fps=8)

if __name__ == "__main__":
    # video_background_subtractor(alg='MOG', input_video_name='sumo_sanjose_t-2021-10-30_22.18.58')
    # image_background_subtractor(alg='KNN', input_video_name='carla_town02_8_view_20220303_color')
    # image_background_subtractor(alg='MOG', input_video_name='carla_town02_8_view_20220303_color')
    # image_background_subtractor(alg='MOG2', input_video_name='carla_town02_8_view_20220303_color')
    # image_background_subtractor(alg='GSOC', input_video_name='carla_town02_8_view_20220303_color')
    # image_background_subtractor(alg='GMG', input_video_name='carla_town02_8_view_20220303_color')
    # T2NO(input_video_name='carla_town02_8_view_20220303_color', mode='static')
    # T2NO(input_video_name='carla_town02_2_view_20220524_color_waypoint', mode='mean')
    # T2NO(input_video_name='/home/jsun/data/Programs/msl-traffic-prediction/packages/carla_tools/carla_town02_4_view_20230109_icra', mode='static', num_test=2000, first_skip=0)
    # for his_length in [2, 10, 20, 30, 60]:
    #     T2NO(
    #         input_video_name='/home/jsun/data/Programs/msl-traffic-prediction/packages/carla_tools/aug',
    #         mode='static', num_test=2000, first_skip=0, vis=True, base_idx=0, his_length=his_length)
    convert_cv2_plt()
    # cv2_tracking(input_video_name='carla_town02_8_view_20220303_color', mode='static')
    # img2video(input_video_name='carla_town02_8_view_20220524_color_waypoint', mode='static')