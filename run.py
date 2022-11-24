# train_n_est.py train a DeepFit model
# Author:Itzik Ben Sabat sitzikbs[at]gmail.com
# If you use this code,see LICENSE.txt file and cite our work

from __future__ import print_function
import argparse
import os
import sys
import random
import math
import shutil
import torch
import torch.nn.parallel 
import torch.optim as optim
import torch.utils.data
from tensorboardX import SummaryWriter # https://github.com/lanpa/tensorboard-pytorch

import sys

from test_NeAF import coarse_normal_prediction, coarse_normal_refinement
from dataset_model_with_query_vector import PointcloudPatchDataset, RandomPointcloudPatchSampler, SequentialShapeRandomPointcloudPatchSampler
import models.my_model as DeepFit

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='choose to train or test')
    # naming / file handling
    parser.add_argument('--name', type=str, default='DeepFit_no_noise', help='training run name')
    parser.add_argument('--arch', type=str, default='simple', help='arcitecture name:  "simple" | "3dmfv"')
    parser.add_argument('--desc', type=str, default='My training run for single-scale normal estimation.', help='description')
    parser.add_argument('--indir', type=str, default='/data/lisj/AdaFit/data/pclouds/', help='input folder (point clouds)')
    parser.add_argument('--logdir', type=str, default='./log_multi_scale/my_experiments/', help='training log folder')
    parser.add_argument('--trainset', type=str, default='trainingset_whitenoise.txt', help='training set file name')
    parser.add_argument('--saveinterval', type=int, default=10, help='save model each n epochs')
    parser.add_argument('--refine', action="store_true", help='flag to refine the model, path determined by outri and model name')
    parser.add_argument('--refine_epoch', type=int, default=500, help='refine model from this epoch')
    parser.add_argument('--overwrite', action="store_true", help='to overwrite existing log directory')
    parser.add_argument('--gpu', type=str, default=['0'], help='set < 0 to use CPU', nargs='+')

    # training parameters
    parser.add_argument('--nepoch', type=int, default=1000, help='number of epochs to train for')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer adam / SGD / rmsprop')
    parser.add_argument('--opt_eps', type=float, default=1e-08, help='optimizer epsilon')
    parser.add_argument('--batchSize', type=int, default=512, help='input batch size')
    parser.add_argument('--patch_radius', type=float, default=[0.05], nargs='+', help='patch radius in multiples of the shape\'s bounding box diagonal, multiple values for multi-scale.')
    parser.add_argument('--patch_center', type=str, default='point', help='center patch at...\n'
                        'point: center point\n'
                        'mean: patch mean')
    parser.add_argument('--patch_point_count_std', type=float, default=0, help='standard deviation of the number of points in a patch')
    parser.add_argument('--patches_per_shape', type=int, default=1000, help='number of patches sampled from each shape in an epoch')
    parser.add_argument('--workers', type=int, default=1, help='number of data loading workers - 0 means same thread as main execution')
    parser.add_argument('--cache_capacity', type=int, default=100, help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')
    parser.add_argument('--seed', type=int, default=3627473, help='manual seed')
    parser.add_argument('--training_order', type=str, default='random', help='order in which the training patches are presented:\n'
                        'random: fully random over the entire dataset (the set of all patches is permuted)\n'
                        'random_shape_consecutive: random over the entire dataset, but patches of a shape remain consecutive (shapes and patches inside a shape are permuted)')
    parser.add_argument('--identical_epochs', type=int, default=False, help='use same patches in each epoch, mainly for debugging')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--scheduler_type', type=str, default='step', help='step or plateau')
    parser.add_argument('--momentum', type=float, default=0.9, help='gradient descent momentum')
    parser.add_argument('--normal_loss', type=str, default='sin', help='Normal loss type:\n'
                        'ms_euclidean: mean square euclidean distance\n'
                        'ms_oneminuscos: mean square 1-cos(angle error)\n'
                        'sin: mean sin(angle error)')

    # model hyperparameters
    parser.add_argument('--outputs', type=str, nargs='+', default=['unoriented_normals', 'neighbor_normals'], help='outputs of the network, a list with elements of:\n'
                        'unoriented_normals: unoriented (flip-invariant) point normals\n'
                        'oriented_normals: oriented point normals\n'
                        'max_curvature: maximum curvature\n'
                        'min_curvature: mininum curvature')
    parser.add_argument('--sym_op', type=str, default='max', help='symmetry operation')
    parser.add_argument('--point_tuple', type=int, default=1, help='use n-tuples of points as input instead of single points')

    parser.add_argument('--use_point_stn', type=int, default=True, help='use point spatial transformer')
    parser.add_argument('--use_feat_stn', type=int, default=True, help='use feature spatial transformer')
    parser.add_argument('--use_pca', type=int, default=True, help='use pca on point clouds, must be true for jet fit type')


    parser.add_argument('--n_gaussians', type=int, default=1, help='use feature spatial transformer')

    parser.add_argument('--jet_order', type=int, default=3, help='jet polynomial fit order')
    parser.add_argument('--points_per_patch', type=int, default=700, help='max. number of points per patch')
    parser.add_argument('--neighbor_search', type=str, default='k', help='[k | r] for k nearest and radius')
    parser.add_argument('--weight_mode', type=str, default="sigmoid", help='which function to use on the weight output: softmax, tanh, sigmoid')
    parser.add_argument('--use_consistency', type=int, default=True, help='flag to use consistency loss')
    parser.add_argument('--con_reg', type=str, default='log', help='choose consistency regularizer: mean, uniform')
    
    parser.add_argument('--batch_query_size', type=int, default=400, help='')
    parser.add_argument('--use_bn',type=int, default=True, help='use batch normalization')
    parser.add_argument('--load_param', type=int, default=1, help='initialize encoder')
    parser.add_argument('--decoder_wn', type=int, default=0, help='use weight normalization')
    parser.add_argument('--query_vector_path', type=str, default='./query_vector_5k.xyz')

    # -----------------------------------------------------test----------------------------------------------------
    parser.add_argument('--test_epoch', type=int, default=50, help='epoch of testing model')
    parser.add_argument('--testset', type=str, default='testset_all.txt', help='shape set file name')
    parser.add_argument('--test_query_size', type=int, default=10000, help='size of initial query vectors at inference')
    parser.add_argument('--parmpostfix', type=str, default='_params.pth', help='parameter file postfix')
    parser.add_argument('--sampling', type=str, default='full', help='sampling strategy, any of:\n'
                        'full: evaluate all points in the dataset\n'
                        'sequential_shapes_random_patches: pick n random points from each shape as patch centers, shape order is not randomized')
    parser.add_argument('--sparse_patches', type=int, default=1, help='evaluate on a sparse set of patches, given by a .pidx file containing the patch center point indices.')
    parser.add_argument('--checkpoints', type=int, default=[5], nargs='+', help='check iters in coarse normal refinement')
    parser.add_argument('--refine_batchSize', type=int, default=600, help='batch size of coarse normal refinement')
    parser.add_argument('--pred_batchSize', type=int, default=128, help='batch size of coarse normal prediction')
    parser.add_argument('--need_prediction', type=int, default=1, help='random coarse normals or predicted coarse normals')
    parser.add_argument('--save_prediction', type=int, default=0, help='1 means saving predicted coarse normals in refinement')
    parser.add_argument('--coarse_normal_num', type=int, default=10, help='number of coarse normals')
    parser.add_argument('--res_type', type=str, default='avg', help='averaging coarse normals')
    parser.add_argument('--update_lr', type=float, default=0.005, help='learning rate of refinement')

    # -----------------------------------------------------eval----------------------------------------------------
    parser.add_argument('--dataset_list', type=str,
                        default=['testset_no_noise', 'testset_low_noise', 'testset_med_noise', 'testset_high_noise',
                                'testset_vardensity_striped', 'testset_vardensity_gradient'], nargs='+',
                        help='list of .txt files containing sets of point cloud names for evaluation')

    return parser.parse_args()

def log_string(out_str, log_file):
    log_file.write(out_str+'\n')
    log_file.flush()
    print(out_str)

def update_learning_rate(opt, iter_step, loader, optimizer):
    warn_up = 3 * len(loader)
    max_iter = opt.nepoch * len(loader)
    init_lr = opt.lr
    lr =  (iter_step / warn_up) if iter_step < warn_up else 0.5 * (math.cos((iter_step - warn_up)/(max_iter - warn_up) * math.pi) + 1) 
    lr = lr * init_lr
    for g in optimizer.param_groups:
        g['lr'] = lr


def train_NeAF(opt):

    all_gpu = ','.join(opt.gpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = all_gpu

    # colored console output
    green = lambda x: '\033[92m' + x + '\033[0m'

    log_dirname = os.path.join(opt.logdir, opt.name)
    out_dir = os.path.join(log_dirname, 'trained_models')
    params_filename = os.path.join(out_dir, '%s_params.pth' % (opt.name))
    model_filename = os.path.join(out_dir, '%s_model.pth' % (opt.name))
    desc_filename = os.path.join(out_dir, '%s_description.txt' % (opt.name))
    log_filename = os.path.join(log_dirname, 'out.log')
    
    
    if (os.path.exists(log_dirname) or os.path.exists(model_filename)) and not opt.name == 'DeepFit_trainall' and opt.refine == '':
        if opt.overwrite:
            response = 'y'
        else:
            response = input('A training run named "%s" already exists, overwrite? (y/n) ' % (opt.name))
        if response == 'y':
            if os.path.exists(log_dirname):
                shutil.rmtree(os.path.join(opt.logdir, opt.name))
        else:
            sys.exit()

    train_writer = SummaryWriter(os.path.join(log_dirname, 'train'))
    log_file = open(log_filename, 'w')
    
    model = get_model(opt)
    tmp_iter = 0

    # load initial parameters
    if opt.load_param:
        print("Loading initial parameters")
        AdaFit_model_path = 'ckpts/AdaFit_model_599.pth'
        AdaFit_dict = torch.load(AdaFit_model_path)
        model_dict = model.state_dict()
        for k,v in AdaFit_dict.items():
            model_dict[k] = AdaFit_dict[k]
        model.load_state_dict(model_dict)
    
    device_id = []
    for i in range(len(opt.gpu)):
        device_id.append(i)
    model = torch.nn.DataParallel(model, device_ids=device_id)
    if opt.refine:
        refine_model_filename = os.path.join(out_dir, '{}_model_{}.pth' .format(opt.name, opt.refine_epoch))
        print("refining %s ..." %(refine_model_filename))
        model.load_state_dict(torch.load(refine_model_filename, map_location={'cuda:2':'cuda:0'}))
        tmp_iter=opt.refine_epoch * len(train_dataloader)
    else:
        opt.refine_epoch = 0

    if opt.seed < 0:
        opt.seed = random.randint(1, 10000)

    print("Random Seed: %d" % (opt.seed))
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    target_features, output_target_ind, output_pred_ind, output_loss_weight = get_target_features((opt))
    train_dataloader, train_dataset, train_datasampler = get_data_loaders(opt, target_features)

    # keep the exact training shape names for later reference
    opt.train_shapes = train_dataset.shape_names

    log_string('training set: %d patches (in %d batches) -' % (len(train_datasampler), len(train_dataloader)), log_file)

    try:
        os.makedirs(out_dir)
    except OSError:
        pass

    if opt.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)
    elif opt.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=0.0000001, eps=opt.opt_eps)
    elif opt.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=opt.lr, weight_decay=0.0000001, eps=opt.opt_eps)
    else:
        raise ValueError("Unsupported optimizer")

    train_num_batch = len(train_dataloader)

    # save parameters
    torch.save(opt, params_filename)

    # save description
    with open(desc_filename, 'w+') as text_file:
        print(opt.desc, file=text_file)

    
    for epoch in range(opt.refine_epoch, opt.nepoch):

        train_enum = enumerate(train_dataloader)
        for train_batchind, data_packet in train_enum:

            tmp_iter += 1
            update_learning_rate(opt, tmp_iter, train_dataloader, optimizer)

            # set to training mode
            model.train()
            data, generated_data = data_packet

            # get trainingset batch and upload to GPU
            points = data[0]

            points = points.transpose(2, 1) # batchsize * 3 * patchsize
            points = points.cuda()
            train_query_vectors, train_angle_offsets = generated_data
            train_query_vectors = train_query_vectors.cuda()
            train_angle_offsets = train_angle_offsets.cuda()

            optimizer.zero_grad()
            
            pred, trans, trans2 = model(points, train_query_vectors)
            loss, angle_loss, regular_trans = compute_loss(pred=pred, target=train_angle_offsets, trans=trans, trans2=trans2)

            # backpropagate through entire network to compute gradients of loss w.r.t. parameters
            loss.backward()
                
            # parameter optimization step
            optimizer.step()

            train_fraction_done = (train_batchind+1) / train_num_batch

            # print info and update log file
            log_string('[%s %d: %d/%d] %s loss: %f lr: %f' % (opt.name, epoch, train_batchind, train_num_batch-1, green('train'), loss.item(), optimizer.param_groups[0]['lr']), log_file)

            train_writer.add_scalar('total_loss', loss.item(),
                                    (epoch + train_fraction_done) * train_num_batch * opt.batchSize)
            train_writer.add_scalar('angle_loss', angle_loss.item(),
                                    (epoch + train_fraction_done) * train_num_batch * opt.batchSize)
            train_writer.add_scalar('trans_loss', regular_trans.item(),
                                    (epoch + train_fraction_done) * train_num_batch * opt.batchSize)                   
            train_writer.add_scalar('lr', optimizer.param_groups[0]['lr'],
                                    (epoch + train_fraction_done) * train_num_batch * opt.batchSize)

        if epoch % opt.saveinterval == 0 or epoch == opt.nepoch-1:
            log_string("saving model to file :{}".format(model_filename),log_file)
            torch.save(model.state_dict(), model_filename)

        # save model in a separate file in epochs 0,5,10,50,100,500,1000, ...
        if epoch % (5 * 10**math.floor(math.log10(max(2, epoch-1)))) == 0 or epoch % 1 == 0 or epoch == opt.nepoch-1:
            log_string("saving model to file :{}".format('%s_model_%d.pth' % (opt.name, epoch)), log_file)
            torch.save(model.state_dict(), os.path.join(out_dir, '%s_model_%d.pth' % (opt.name, epoch)))


        if ((opt.nepoch - epoch) <= 300 and epoch % 100 == 0) or ((opt.nepoch -1 - epoch) <= 30 and (opt.nepoch -1 - epoch) % 10 == 0):
            coarse_normal_refinement(opt, epoch, model)
        
        

    

def test_NeAF(opt):
    all_gpu = ','.join(opt.gpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = all_gpu

    log_dirname = os.path.join(opt.logdir, opt.name)
    out_dir = os.path.join(log_dirname, 'trained_models')
    model = get_model(opt)
  
    device_id = []
    for i in range(len(opt.gpu)):
        device_id.append(i)
    model = torch.nn.DataParallel(model, device_ids=device_id)

    model_path = os.path.join(out_dir, '{}_model_{}.pth' .format(opt.name, opt.test_epoch))
    model.load_state_dict(torch.load(model_path, map_location={'cuda:2':'cuda:0'}))

    if opt.seed < 0:
        opt.seed = random.randint(1, 10000)
    print("Random Seed: %d" % (opt.seed))
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    coarse_normal_refinement(opt, opt.test_epoch, model)

    
def compute_loss(pred, target, trans, trans2, loss_function=torch.nn.L1Loss()):
    loss = loss_function(pred, target)
    regularizer_trans = compute_regularizer(trans, trans2)
    total_loss = loss + regularizer_trans
    return total_loss, loss, regularizer_trans

def compute_regularizer(trans, trans2):
    regularizer_trans = 0
    if trans is not None:
        regularizer_trans += 0.1 * torch.nn.MSELoss()(trans * trans.permute(0, 2, 1),
                                                torch.eye(3, device=trans.device).unsqueeze(0).repeat(trans.size(0), 1, 1))
    if trans2 is not None:
        regularizer_trans += 0.01 * torch.nn.MSELoss()(trans2 * trans2.permute(0, 2, 1),
                                                 torch.eye(64, device=trans.device).unsqueeze(0).repeat(trans.size(0), 1, 1))
    return regularizer_trans


def get_data_loaders(opt, target_features):
    # create train and test dataset loaders
    train_dataset = PointcloudPatchDataset(
        root=opt.indir,
        shape_list_filename=opt.trainset,
        patch_radius=opt.patch_radius,
        points_per_patch=opt.points_per_patch,
        patch_features=target_features,
        point_count_std=opt.patch_point_count_std,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs,
        use_pca=opt.use_pca,
        center=opt.patch_center,
        point_tuple=opt.point_tuple,
        cache_capacity=opt.cache_capacity,
        neighbor_search_method=opt.neighbor_search,
        query_vector_path=opt.query_vector_path,
        batch_query_size=opt.batch_query_size)
    if opt.training_order == 'random':
        train_datasampler = RandomPointcloudPatchSampler(
            train_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    elif opt.training_order == 'random_shape_consecutive':
        train_datasampler = SequentialShapeRandomPointcloudPatchSampler(
            train_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    else:
        raise ValueError('Unknown training order: %s' % (opt.training_order))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_datasampler,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers))


    return train_dataloader, train_dataset, train_datasampler


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

    return target_features, output_target_ind, output_pred_ind, output_loss_weight


def get_model(opt):
    # create model
    if opt.arch == 'simple':
        model = DeepFit.DeepFit(1, opt.points_per_patch,
                                            use_point_stn=opt.use_point_stn, use_feat_stn=opt.use_feat_stn,
                                            point_tuple=opt.point_tuple, sym_op=opt.sym_op,
                                            jet_order=opt.jet_order,
                                            weight_mode=opt.weight_mode, use_consistency=opt.use_consistency, 
                                            use_batchNormalization=opt.use_bn, use_wn=opt.decoder_wn).cuda()
    elif opt.arch == '3dmfv':
        model = DeepFit.DeepFit(1, opt.points_per_patch,
                                            use_point_stn=opt.use_point_stn,
                                            use_feat_stn=opt.use_feat_stn, point_tuple=opt.point_tuple,
                                            sym_op=opt.sym_op, arch=opt.arch, n_gaussians=opt.n_gaussians,
                                            jet_order=opt.jet_order,
                                            weight_mode=opt.weight_mode, use_consistency=opt.use_consistency).cuda()
    else:
        raise ValueError('Unsupported architecture type')
    return model


if __name__ == '__main__':
    train_opt = parse_arguments()
    if train_opt.mode == 'train':
        train_NeAF(train_opt)
    elif train_opt.mode == 'test':
        test_NeAF(train_opt)
