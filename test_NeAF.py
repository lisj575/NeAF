#coding=utf-8

import os
import sys
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import time
from dataset_model_test import PointcloudPatchDataset, SequentialShapeRandomPointcloudPatchSampler, SequentialPointcloudPatchSampler
from torch.autograd import Variable
from eval import eval_pcs

def coarse_normal_prediction(opt, epoch, model, dataset, test_datasampler, coarse_normal_num=1, save_flag=False):
    print('coarse normal prediction...')
    opt.test_name = opt.name.split()

    if opt.seed < 0:
        opt.seed = random.randint(1, 10000)

    for model_name in opt.test_name:
        # fetch the model from the log dir
        # append model name to output directory and create directory if necessary
        model_log_dir =  os.path.join(opt.logdir , model_name,'trained_models')
        param_filename = os.path.join(model_log_dir, model_name+opt.parmpostfix)
        output_dir = os.path.join(opt.logdir, model_name, 'results_epoch' + str(epoch))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        trainopt = torch.load(param_filename)

    # randoam
    test_query_size = opt.test_query_size
    # get indices in targets and predictions corresponding to each output
    dataloader = torch.utils.data.DataLoader(dataset, sampler=test_datasampler, batch_size=opt.pred_batchSize, num_workers=int(opt.workers))

    regressor = model
    regressor.eval()

    num_batch = len(dataloader)
    batch_enum = enumerate(dataloader, 0)

    shape_ind = 0
    shape_patch_offset = 0
    if opt.sampling == 'full':
        shape_patch_count = dataset.shape_patch_count[shape_ind]
    elif opt.sampling == 'sequential_shapes_random_patches':
        shape_patch_count = min(opt.patches_per_shape, dataset.shape_patch_count[shape_ind])
    else:
        raise ValueError('Unknown sampling strategy: %s' % opt.sampling)
    
    normal_prop = torch.zeros([shape_patch_count, 3]) # output normals
    total_patches = 0
    for ind in range(len(dataset.shape_names)):
        total_patches += dataset.shape_patch_count[ind]
    coarse_normals = torch.zeros(total_patches, coarse_normal_num, 3, dtype = torch.float32).cuda() # coarse normals
    
    offset = 0
    shape_offset = dataset.shape_patch_count[shape_ind]
    for batchind, data in batch_enum:
        # get batch and upload to GPU
        points = data[0]
        data_trans = data[-3]
        points = points.transpose(2, 1)
        points = points.cuda()
        data_trans = data_trans.cuda()

        batchsize = points.shape[0]

        # start_time = time.time()
        query_vectors = torch.randn(batchsize, test_query_size, 3).cuda()
        query_vectors = torch.nn.functional.normalize(query_vectors, dim=-1)
    
        with torch.no_grad():
            pred_angle_offset, _, _ = regressor(points, query_vectors) # batchsize * norm_size(1)

        # end_time = time.time()
        n_est = choose_min_query_vector(pred_angle_offset, query_vectors.detach(), coarse_normal_num).detach()
        coarse_normals[offset:offset+batchsize,:,:] = torch.nn.functional.normalize(n_est, dim=-1).detach().clone()
        offset += batchsize

        if batchind % 20 == 0:
            print('Prediction [%s %d/%d] shape %s' % (model_name, batchind, num_batch-1, dataset.shape_names[shape_ind]))
        
        # Save estimated normals to file
        if save_flag:  
            if trainopt.use_pca:
                n_est = n_est[:,0,:] # only predict one normal
                # transform predictions with inverse pca rotation (back to world space)
                n_est[:, :] = torch.bmm(n_est.unsqueeze(1), data_trans.transpose(2, 1)).squeeze(dim=1)
            batch_offset = 0

            while batch_offset < n_est.shape[0] and shape_ind + 1 <= len(dataset.shape_names):
                shape_patches_remaining = shape_patch_count - shape_patch_offset
                batch_patches_remaining = n_est.shape[0] - batch_offset

                # append estimated patch properties batch to properties for the current shape on the CPU
                normal_prop[shape_patch_offset:shape_patch_offset + min(shape_patches_remaining,
                                                                        batch_patches_remaining), :] = \
                    n_est[batch_offset:batch_offset + min(shape_patches_remaining, batch_patches_remaining), :]

                batch_offset = batch_offset + min(shape_patches_remaining, batch_patches_remaining)
                shape_patch_offset = shape_patch_offset + min(shape_patches_remaining, batch_patches_remaining)

                if shape_patches_remaining <= batch_patches_remaining:
                    normals_to_write = normal_prop.cpu().numpy()
                    eps=1e-6
                    normals_to_write[np.logical_and(normals_to_write < eps, normals_to_write > -eps)] = 0.0
                    out_name = dataset.shape_names[shape_ind] + '_coarse.normals'
                    np.savetxt(os.path.join(output_dir, out_name),
                            normals_to_write)
                    print('saved normals for ' + dataset.shape_names[shape_ind])
                    sys.stdout.flush()
                    shape_patch_offset = 0
                    shape_ind += 1
                    if shape_ind < len(dataset.shape_names):
                        shape_patch_count = dataset.shape_patch_count[shape_ind]
                        normal_prop = torch.zeros([shape_patch_count, 3])
        elif offset > shape_offset:
            shape_ind += 1
            shape_offset += dataset.shape_patch_count[shape_ind]

    if save_flag:
        eval_pcs(opt, epoch, '_coarse')
    return coarse_normals


def coarse_normal_refinement(opt, epoch, model):
    
    opt.test_name = opt.name.split()
    if opt.seed < 0:
        opt.seed = random.randint(1, 10000)
    
    for model_name in opt.test_name:
        # fetch the model from the log dir
        # append model name to output directory and create directory if necessary
        model_log_dir =  os.path.join(opt.logdir , model_name, 'trained_models')
        param_filename = os.path.join(model_log_dir, model_name+opt.parmpostfix)
        output_dir = os.path.join(opt.logdir, model_name, 'results_epoch' + str(epoch))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        random.seed(opt.seed)
        torch.manual_seed(opt.seed)

        # load model and training parameters
        trainopt = torch.load(param_filename)

    
    regressor = model
    regressor.eval()
    # get indices in targets and predictions corresponding to each output
    target_features, output_target_ind, output_pred_ind, output_loss_weight, pred_dim = get_target_features((trainopt))
    dataloader, dataset, datasampler = get_data_loaders(opt, trainopt, target_features, opt.refine_batchSize)
    
    coarse_normal_num = opt.coarse_normal_num
    checkpoints = opt.checkpoints
    # initializing
    if opt.need_prediction:
        coarse_normals = coarse_normal_prediction(opt, epoch, model, dataset, datasampler, coarse_normal_num, opt.save_prediction)
    else:
        # sample query vectors on the unit sphere randomly
        total_patches = 0
        for shape_ind in range(len(dataset.shape_names)):
            total_patches += dataset.shape_patch_count[shape_ind]
        coarse_normals = torch.randn(total_patches, coarse_normal_num, 3, dtype=torch.float32).cuda() 
        coarse_normals = torch.nn.functional.normalize(coarse_normals, dim=-1)

    if opt.res_type == 'avg':
        res_function = average_normals
    else:
        res_function = choose_min_query_vector  # By default, the vector with the minimum angle offset is selected

    print('coarse normal refinement...')
    n_epoch = checkpoints[-1] + 1 # max iteration step
    ckpts_ind = 0
    tmp_iter = 0

    for i in range(1, n_epoch):
        batch_enum = enumerate(dataloader, 0)
        shape_ind = 0
        shape_patch_offset = 0

        if opt.sampling == 'full':
            shape_patch_count = dataset.shape_patch_count[shape_ind]
        elif opt.sampling == 'sequential_shapes_random_patches':
            shape_patch_count = min(opt.patches_per_shape, dataset.shape_patch_count[shape_ind])
        else:
            raise ValueError('Unknown sampling strategy: %s' % opt.sampling)

        normal_prop = torch.zeros([shape_patch_count, 3])
        offset = 0
        shape_offset = dataset.shape_patch_count[shape_ind]
        num_batch = len(dataloader)

        for batchind, data in batch_enum:
            
            tmp_iter += 1
            # get batch and upload to GPU
            points = data[0].transpose(2, 1).cuda()
            data_trans = data[-3]
            batchsize = points.shape[0]
            batch_coarse_normals = coarse_normals[offset:offset+batchsize,:,:]

            # gradient decent based refinement
            batch_coarse_normals = Variable(batch_coarse_normals, requires_grad=True)
            optimizer = torch.optim.Adam([batch_coarse_normals], lr=opt.update_lr, weight_decay=0.0000001, eps=trainopt.opt_eps)

            optimizer.zero_grad()
            pred_angle_offset, _, _ = regressor(points, batch_coarse_normals)
            n_est = res_function(pred_angle_offset, batch_coarse_normals.detach()).cpu()
            
            loss = torch.nn.L1Loss()(pred_angle_offset, torch.zeros_like(pred_angle_offset))
            loss.backward()
            optimizer.step()

            coarse_normals[offset:offset+batchsize,:,:] = torch.nn.functional.normalize(batch_coarse_normals, dim=-1).detach().clone()

            offset += batchsize
            if batchind % 20 == 0:
                print('Refinement [%s iter %d batch %d/%d] shape %s' % (model_name, i, batchind, num_batch-1, dataset.shape_names[shape_ind]))
        
            if i == checkpoints[ckpts_ind]:
                if trainopt.use_pca:
                    # transform predictions with inverse pca rotation (back to world space)
                    n_est[:, :] = torch.bmm(n_est.unsqueeze(1), data_trans.transpose(2, 1)).squeeze(dim=1)

                # Save estimated normals to file
                batch_offset = 0
                
                while batch_offset < n_est.shape[0] and shape_ind + 1 <= len(dataset.shape_names):
                    shape_patches_remaining = shape_patch_count - shape_patch_offset
                    batch_patches_remaining = n_est.shape[0] - batch_offset

                    # append estimated patch properties batch to properties for the current shape on the CPU
                    normal_prop[shape_patch_offset:shape_patch_offset + min(shape_patches_remaining,
                                                                            batch_patches_remaining), :] = \
                        n_est[batch_offset:batch_offset + min(shape_patches_remaining, batch_patches_remaining), :]
                    batch_offset = batch_offset + min(shape_patches_remaining, batch_patches_remaining)
                    shape_patch_offset = shape_patch_offset + min(shape_patches_remaining, batch_patches_remaining)

                    if shape_patches_remaining <= batch_patches_remaining:
                        normals_to_write = normal_prop.cpu().numpy()
                        eps=1e-6
                        normals_to_write[np.logical_and(normals_to_write < eps, normals_to_write > -eps)] = 0.0
                        out_name = dataset.shape_names[shape_ind] +'_iter'+str(i) + '.normals'
                        np.savetxt(os.path.join(output_dir, out_name), normals_to_write)

                        print('saved normals for ' + dataset.shape_names[shape_ind])
                        sys.stdout.flush()
                        shape_patch_offset = 0
                        shape_ind += 1
                        if shape_ind < len(dataset.shape_names):
                            shape_patch_count = dataset.shape_patch_count[shape_ind]
                            normal_prop = torch.zeros([shape_patch_count, 3])
                        # evaluate after saving all normals 
                        else:
                            if ckpts_ind < len(checkpoints) - 1:
                                ckpts_ind += 1
                            eval_pcs(opt, epoch, '_iter'+str(i))
            elif offset > shape_offset:
                shape_ind += 1
                shape_offset += dataset.shape_patch_count[shape_ind]


            


            
def choose_min_query_vector(pred_angle_offset, query_vector, len=1):
    pred_angle_offset = abs(pred_angle_offset)
    sorted, ind = torch.sort(pred_angle_offset)
    arrange = torch.arange(0, pred_angle_offset.shape[0]).reshape(-1,1).repeat(1, pred_angle_offset.shape[1])
    sorted_query_vector = query_vector[arrange, ind]
    return sorted_query_vector[:, 0:len,:]

def average_normals(pred_angle_offset, refined_normal):
    first_normal = refined_normal[:,0,:].unsqueeze(-1) # reference normal (n^c_r)'
    sign = torch.sign(torch.matmul(refined_normal, first_normal))
    signed_normal = refined_normal * sign
    n_pred = signed_normal.mean(dim=-2)
    return torch.nn.functional.normalize(n_pred, dim=-1)


def get_data_loaders(opt, trainopt, target_features, batchSize):
    # create dataset loader

    model_batchSize = batchSize

    test_dataset = PointcloudPatchDataset(
        root=opt.indir,
        shape_list_filename=opt.testset,
        patch_radius=trainopt.patch_radius,
        points_per_patch=trainopt.points_per_patch,
        patch_features=target_features,
        point_count_std=trainopt.patch_point_count_std,
        seed=opt.seed,
        identical_epochs=trainopt.identical_epochs,
        use_pca=trainopt.use_pca,
        center=trainopt.patch_center,
        point_tuple=trainopt.point_tuple,
        sparse_patches=opt.sparse_patches,
        cache_capacity=opt.cache_capacity,
        neighbor_search_method=trainopt.neighbor_search)

    if opt.sampling == 'full':
        test_datasampler = SequentialPointcloudPatchSampler(test_dataset)
    elif opt.sampling == 'sequential_shapes_random_patches':
        test_datasampler = SequentialShapeRandomPointcloudPatchSampler(
            test_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            sequential_shapes=True,
            identical_epochs=False)
    else:
        raise ValueError('Unknown sampling strategy: %s' % opt.sampling)
    

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        sampler=test_datasampler,
        batch_size=model_batchSize,
        num_workers=int(opt.workers))

    return test_dataloader, test_dataset, test_datasampler


def get_target_features(opt):
    # get indices in targets and predictions corresponding to each output
    target_features = []
    output_target_ind = []
    output_pred_ind = []
    output_loss_weight = []
    pred_dim = 0
    for o in opt.outputs:
        if o == 'unoriented_normals' or o == 'oriented_normals':
            if 'normal' not in target_features:
                target_features.append('normal')

            output_target_ind.append(target_features.index('normal'))
            output_pred_ind.append(pred_dim)
            output_loss_weight.append(1.0)
            pred_dim += 3
        elif o == 'max_curvature' or o == 'min_curvature':
            if o not in target_features:
                target_features.append(o)

            output_target_ind.append(target_features.index(o))
            output_pred_ind.append(pred_dim)
            if o == 'max_curvature':
                output_loss_weight.append(0.7)
            else:
                output_loss_weight.append(0.3)
            pred_dim += 1
        elif o == 'neighbor_normals':
            target_features.append(o)
            output_target_ind.append(target_features.index(o))
            output_pred_ind.append(pred_dim)
        else:
            raise ValueError('Unknown output: %s' % (o))

    if pred_dim <= 0:
        raise ValueError('Prediction is empty for the given outputs.')

    return target_features, output_target_ind, output_pred_ind, output_loss_weight, pred_dim


