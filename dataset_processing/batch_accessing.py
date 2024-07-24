from dataset_processing.dataset_batch_creation import DataProcess

# This class is used to create the batches and can be accessed from class InputHandle
def data_provider(dataset_name, train_data_paths, valid_data_paths, batch_size,
                  img_width, seq_length, is_training=True, num_views=1, img_channel=3, baseline='SSTA_view_view',
                  eval_batch_size=1, n_epoch=1, args=None):
    train_data_list = train_data_paths.split(',')
    valid_data_list = valid_data_paths.split(',')
     
    try:
        test_seq_length = args.eval_num_step + args.num_past + 1
    except:
        print('No args.eval_num_step, use seq_length as test_seq_length')
        test_seq_length = args.test_sequence

    input_param = {'paths': valid_data_list,
                   'image_width': img_width,
                   'minibatch_size': eval_batch_size,
                   'seq_length': test_seq_length,
                   'input_data_type': 'float32',
                   'name': dataset_name + ' iterator',
                    'num_views': num_views,
                   'img_channel': img_channel,
                   'baseline': baseline,
                   'n_epoch': n_epoch,
                   "model_type":args.model_type,
                   "sequence_index_gap":args.sequence_index_gap
                   
                   }
    
    input_handle = DataProcess(input_param)
    if is_training:
        train_input_param = {'paths': train_data_list,
                            'image_width': img_width,
                            'minibatch_size': batch_size,
                            'seq_length': seq_length,
                            'input_data_type': 'float32',
                            'name': dataset_name + ' iterator',
                            'num_views': num_views,
                            'img_channel': img_channel,
                            'n_epoch': n_epoch,
                            "model_type":args.model_type,
                            "sequence_index_gap":args.sequence_index_gap}
        train_input_handle = DataProcess(train_input_param)
        train_input_handle = train_input_handle.get_train_input_handle()
        train_input_handle.begin(do_shuffle=True)
        test_input_handle = input_handle.get_test_input_handle()
        test_input_handle.begin(do_shuffle=False)
        return train_input_handle, test_input_handle
    else:
        test_input_handle = input_handle.get_test_input_handle()
        test_input_handle.begin(do_shuffle=False)
        return test_input_handle

