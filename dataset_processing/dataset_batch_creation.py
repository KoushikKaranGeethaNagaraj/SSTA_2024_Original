import numpy as np
import os
import cv2
from PIL import Image
import logging
import random
from itertools import zip_longest, chain
import re

logger = logging.getLogger(__name__)
class InputHandle:
    def __init__(self, datas, indices, input_param,ssta_t2n):
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.num_views = input_param.get('num_views', 2)
        self.minibatch_size = input_param['minibatch_size']
        self.image_width = input_param['image_width']
        self.datas = datas
        self.indices = indices
        self.current_position = 0
        self.current_batch_indices = []
        self.current_input_length = input_param['seq_length']
        self.img_channel = input_param.get('img_channel', 3)
        self.n_epoch = input_param.get('n_epoch', 1)
        self.ssta_t2no=ssta_t2n


        
    # This fnction returns the total number of sequences
    def total(self):
        return len(self.indices)
    
    #initialises the batch to 0 and shuffles the dataset if train and not if val/test
    def begin(self, do_shuffle=False, epoch=None):
        logger.info("Initialization for read data ")
        if do_shuffle:
            random.shuffle(self.indices)
            pass
        if epoch:
            self.current_position = int(self.total() * epoch / self.n_epoch)
        else:
            self.current_position = 0
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

        # print(self.indices)
        # print( self.current_batch_indices)
        # print(self.current_position)

    #moving to next batch
    def next(self):
        self.current_position += self.minibatch_size
        if self.no_batch_left():
            return None
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    #checks if all the batches have been completed
    def no_batch_left(self, epoch=None):
        if self.current_position + self.minibatch_size >= self.total():
            return True
        else:
            return False

    #gets the specific batch for training/validation
    def get_batch(self):
        if self.no_batch_left():
            logger.error(
                "There is no batch left in " + self.name + ". Consider to user iterators.begin() to rescan from the beginning of the iterators")
            return None
        if self.ssta_t2no:
            input_batch = np.zeros(
                (self.minibatch_size, self.current_input_length, self.image_width, self.image_width,
                (self.img_channel * self.num_views)+(self.num_views*2))).astype(self.input_data_type)

        else:
            input_batch = np.zeros(
                (self.minibatch_size, self.current_input_length, self.image_width, self.image_width,
                self.img_channel * self.num_views)).astype(self.input_data_type)
            
        for i in range(self.minibatch_size):
            batch_ind = self.current_batch_indices[i]
            begin = batch_ind
            end = begin + self.current_input_length
            data_slice = self.datas[begin:end, :, :, :]
            # print(input_batch[i, :self.current_input_length, :, :, :].shape, data_slice.shape)
            input_batch[i, :self.current_input_length, :, :, :] = data_slice
            
        input_batch = input_batch.astype(self.input_data_type)
        return input_batch

    def print_stat(self):
        print("-"*20)
        print("Iterator Name: " + self.name)
        print("    current_position: " + str(self.current_position))
        print("    Minibatch Size: " + str(self.minibatch_size))
        print("    total Size: " + str(self.total()))
        print("    current_input_length: " + str(self.current_input_length))
        print("    Input Data Type: " + str(self.input_data_type))
        logger.info("Iterator Name: " + self.name)
        logger.info("    current_position: " + str(self.current_position))
        logger.info("    Minibatch Size: " + str(self.minibatch_size))
        logger.info("    total Size: " + str(self.total()))
        logger.info("    current_input_length: " + str(self.current_input_length))
        logger.info("    Input Data Type: " + str(self.input_data_type))
        print("-"*20)


# This Class is used to load/combine the dataset and get the details of sequences for the VAE/SSTA
class DataProcess:
    def __init__(self, input_param):
        self.input_param = input_param
        self.paths = input_param['paths']
        self.image_width = input_param['image_width'] 
        self.seq_len = input_param['seq_length']
        self.num_views = input_param.get('num_views', 2)
        self.img_channel = input_param.get('img_channel', 3)
        self.t2no_output=self.num_views
        self.t2no_output_channels=self.num_views*1
        self.model_type = input_param.get('model_type')
        self.sequence_index_gap=input_param.get('sequence_index_gap')

        self.ssta_t2n = True if self.model_type == "ssta" else False
        self.datautility=DatasetUtility(self.num_views,self.ssta_t2n)
        

    def load_data(self, path,mode="None"):
        # This function takes in (N,Height,Width,ImageChannels) and converts to (N/ssta_views,Height,Width,(ImageChannels*ssta_views))
        
        if self.ssta_t2n:
            frames_np,t2nd_frames_np,t2no_frames_np=self.datautility.dataset_SSTA_alternator(path)
            t2nd_data=np.zeros((t2nd_frames_np.shape[0], self.image_width, self.image_width, 1))
            t2no_data=np.zeros((t2no_frames_np.shape[0], self.image_width, self.image_width, 1))
        #this segment is used the most as the views are appended into data to return in teh combined format (N/ssta_views,Height,Width,(ImageChannels*ssta_views)  
        else:
            frames_np=self.datautility.dataset_SSTA_alternator(path)
        data = np.zeros((frames_np.shape[0], self.image_width, self.image_width, self.img_channel))
        
        print(frames_np.shape,t2nd_frames_np.shape,t2no_frames_np.shape)
        

        for i in range(len(frames_np)):
            temp = np.float32(frames_np[i])
            data[i, :, :, :] = cv2.resize(temp, (self.image_width, self.image_width)) / 255 

            if self.ssta_t2n:
                temp_t2nd=np.float32(t2nd_frames_np[i])
                temp_t2no=np.float32(t2no_frames_np[i])

                
                t_t2nd= cv2.resize(temp_t2nd, (self.image_width, self.image_width)) / 50
                t2nd_data[i, :, :, :]=t_t2nd[..., np.newaxis]

                t_t2no= cv2.resize(temp_t2no, (self.image_width, self.image_width)) / 50
                t2no_data[i, :, :, :]=t_t2no[..., np.newaxis]
        
        if self.ssta_t2n:
            batch_channel=self.img_channel+2

            new_data = np.zeros((frames_np.shape[0] // (self.num_views), self.image_width, self.image_width, (self.img_channel * self.num_views)+(2*self.num_views)))
        
            for i in range(self.num_views):
                new_data[:, :, :, batch_channel*i:(batch_channel*(i+1))-2] = data[i*(frames_np.shape[0] // (self.num_views)):(i+1)*frames_np.shape[0] // (self.num_views),:,:,:]
                new_data[:, :, :, (batch_channel*(i+1))-2][...,np.newaxis] = t2no_data[i*(t2no_frames_np.shape[0] // (self.num_views)):(i+1)*t2no_frames_np.shape[0] // (self.num_views),:,:,:]
                new_data[:, :, :,(batch_channel*(i+1))-1][...,np.newaxis] = t2nd_data[i*(t2nd_frames_np.shape[0] // (self.num_views)):(i+1)*t2nd_frames_np.shape[0] // (self.num_views),:,:,:]
        
        else:
            new_data = np.zeros((frames_np.shape[0] // self.num_views, self.image_width, self.image_width, self.img_channel * self.num_views))
            for i in range(self.num_views):
                new_data[:, :, :, self.img_channel*i:self.img_channel*(i+1)] = data[i::self.num_views][:frames_np.shape[0] // self.num_views]
        
        data = new_data

        # is it a begin index of sequence
        indices = []
        index = len(data) - 1
        while index >= self.seq_len - 1:
            indices.append(index - self.seq_len + 1)
            index -= self.sequence_index_gap
        indices.append(0) if 0 not in indices else None
        print(indices)

   

        self.processed_dataset_info(path,frames_np,data,indices,mode,self.num_views)
        # indices are the total sequences in the combined dataset that is calculated by Total_images -(num_past+num_step), that is each batch
        
        #to display dataset
        # for i in range(data.shape[1]):
        #     name = 't2n0_{0:02d}_{1:02d}.png'.format(i + 1, 1)
        #     file_name = os.path.join(path, name)
        #     img_gt = np.uint8(data[ i, :, :,5:8 ] * 255.)
        #     # print(img_gt.shape)
        #     img_gt=img_gt[:,:,0]
        #     img_pd =np.uint8(data[ i, :, :,9 ] * 255.)
        #     views = np.concatenate([img_gt, img_pd], axis = 1)
            
        #     # img_gt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     # print(img_gt*255)
        #     cv2.imshow("ex",views)
        #     cv2.waitKey(50)
        #     # cv2.imwrite(file_name, img_gt)

        # quit()
        return data, indices
    
    def processed_dataset_info(self,path,frames_np,data,indices,mode,ssta_no):
        print("-"*20)
        print('Loaded data from ' + str(path))
        print("Mode: " ,mode,"SSTA_num:",ssta_no)
        print("Dataset Loaded and Alternated: " ,frames_np.shape)
        print("there are " + str(data.shape[0]) + " pictures ")
        print("Combined Dataset for VAE/SSTA " ,data.shape)
        print("there are " + str(len(indices)) + " sequences")
        print("-"*20)


    def get_train_input_handle(self):
        train_data, train_indices = self.load_data(self.paths[0],mode="train")
        return InputHandle(train_data, train_indices, self.input_param,self.ssta_t2n)

    def get_test_input_handle(self):
        test_data, test_indices = self.load_data(self.paths[0],mode="test")
        return InputHandle(test_data, test_indices, self.input_param,self.ssta_t2n)


#utility class for creating the alternate dataset
class DatasetUtility:
    def __init__(self,total_views,t2n=False):
        #If t2nod/ssta then the outputs are also concatenated
        self.t2n=t2n
        self.total_views=total_views

    #utility class for creating the alternate dataset
        
    # This function is used as a key to sort numerically
    def numerical_sort(self,filename):
        """Extracts the numerical part of a filename for sorting."""
        number_extractor = re.compile(r'\d+')
        match = number_extractor.search(filename)
        if match:
            return int(match.group())
        return 0
    
    def t2n_numerical_sort(self,file_path):
        # Extract numbers from the filename using regex
        numbers = re.findall(r'\d+', os.path.basename(file_path))
        return int(numbers[-1]) if numbers else 0
    

    #This function takes the dataset, example: ssta_num=4, camera_0,camera_1,camera_2,camera_3, and then converts[0,0,...][1,1,1..][2,2,2..][3,3,3..]to [0,1,2,3,0,1,2,3...]
    def dataset_SSTA_alternator(self,dataset_path):
        # arguments:
        #     dataset_paths-contains a  path to train/val/test dataset 
        #     total_views-SSTA views  
        # returns:Numpy array of the path of dataset passed in as shape (N,Height,Width,ImageChannels)--- N is total number of images from all camera/folders
        dataset_files = [[] for _ in range(self.total_views)]
        t2no_files=[[] for _ in range(self.total_views)]
        t2nd_files=[[] for _ in range(self.total_views)]
        

        for _,camera_view in enumerate(os.listdir(dataset_path)):
            full_path = os.path.join(dataset_path, camera_view)
  
            if camera_view[0]=="c":
                    dataset_files[int(camera_view[-1])].extend([os.path.join(full_path, file) for file in sorted(os.listdir(full_path),key=self.numerical_sort)])
                    continue
          
            if self.t2n and camera_view[0]!="c":
                for _,t2n_camera in enumerate(os.listdir(full_path)):
                    t2n_full_path = os.path.join(full_path, t2n_camera)
                    if t2n_camera[0]=="c":
                        t2nd_files[int(t2n_camera[-1])].extend([os.path.join(t2n_full_path, file) for file in sorted(os.listdir(t2n_full_path),key=self.numerical_sort)    if 't2nd' in file])

                        t2no_files[int(t2n_camera[-1])].extend([os.path.join(t2n_full_path, file) for file in sorted(os.listdir(t2n_full_path),key=self.numerical_sort)    if 't2no' in file])
           
        
        ## Combine files alternately
        combined_files_dataset= []
        t2nd_combined_files_dataset= []
        t2no_combined_files_dataset= []
        #alternate the images based on the camera views
        combined_files_dataset = list(chain.from_iterable(filter(None, x) for x in zip_longest(*dataset_files)))
        combined_files_dataset_np = np.stack([np.array(Image.open(path)) for path in combined_files_dataset])

        if self.t2n:
        
            comb_files_dataset = [path for sublist in dataset_files for path in sublist]
    
            comb_files_dataset_np = np.stack([np.array(Image.open(path)) for path in comb_files_dataset])

            
            t2no_sorted_file_paths = [sorted(sublist, key=self.t2n_numerical_sort) for sublist in t2no_files]
            t2no_combined_files_dataset = [path for sublist in t2no_sorted_file_paths for path in sublist]
            t2no_combined_files_dataset_np = np.stack([np.array(Image.open(path)) for path in t2no_combined_files_dataset])

            t2nd_sorted_file_paths = [sorted(sublist, key=self.t2n_numerical_sort) for sublist in t2nd_files]
            t2nd_combined_files_dataset = [path for sublist in t2nd_sorted_file_paths for path in sublist]
            t2nd_combined_files_dataset_np = np.stack([np.array(Image.open(path)) for path in t2nd_combined_files_dataset])

            # print(t2no_combined_files_dataset)
            # quit()
            return comb_files_dataset_np,t2nd_combined_files_dataset_np[..., np.newaxis],t2no_combined_files_dataset_np[..., np.newaxis]

        return combined_files_dataset_np
  





