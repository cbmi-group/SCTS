import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results

import pycocotools.mask as mask_util
from collections import defaultdict
import copy
import json
import cv2
import numpy as np

def cal_iou(mask1,mask2):
    area1 = mask1.sum()
    area2 = mask2.sum()
    inter = ((mask1+mask2)==2).sum()
    mask_iou = inter / (area1+area2-inter)
    mask_ioa = inter / min(area1,area2)
    return mask_iou,mask_ioa

def post_process(result):
    ### cal_iou
    iou_thr = 0.3
    ioa_thr = 0.5
    segm_this_image = result[0][1][0]
    bbox_this_image = result[0][0][0]
    scores = []
    del_indexs = []
    before_len = len(result[0][0][0])
    for i in range(len(result[0][1][0])):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(result[0][1][0][i].astype(np.uint8), connectivity=8)
        if num_labels > 2:
            result[0][1][0][i] = np.array(labels == np.argmax(stats[1:,-1]) + 1) 
    for i in range(len(bbox_this_image)):
        scores.append(bbox_this_image[i][-1])
    
    iou_map = np.zeros((len(segm_this_image),len(segm_this_image)))
    ioa_map = np.zeros((len(segm_this_image),len(segm_this_image)))

    for i  in range(len(segm_this_image)):
        for j in range(len(segm_this_image)):
            if j > i:
                iou_map[i][j],ioa_map[i][j] = cal_iou(segm_this_image[i]*1,segm_this_image[j]*1)

    ### where > thr?
    iou_res = np.where(iou_map>iou_thr)
    ioa_res = np.where(ioa_map>ioa_thr)
    if len(iou_res[0]) > 0:
        print(iou_res)
        for i in range(len(iou_res[0])):
            com_index = [iou_res[0][i],iou_res[1][i]]
            com = [scores[iou_res[0][i]],scores[iou_res[1][i]]]
            min_index = com.index(min(com))
            del_indexs.append(com_index[min_index]) 
    
    # import ipdb
    # ipdb.set_trace()
    new_segm = []
    save_index = [index for index in range(len(result[0][1][0])) if index not in del_indexs]
    for i in range(len(result[0][1][0])):
        if i not in del_indexs:
            new_segm.append(result[0][1][0][i])
    result[0][1][0] = new_segm
    result[0][0][0] = result[0][0][0][save_index]

    print("######################",del_indexs,before_len,len(result[0][0][0]))
    return result


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        before_result = copy.deepcopy(result)
        print(data['img_metas'][0].data[0][0]['ori_filename'])
        # if "210112 hek93 21hour DTT 8mM 013_4.png" not in data['img_metas'][0].data[0][0]['ori_filename']:
        #     continue
        # import ipdb
        # ipdb.set_trace()
        result = post_process(result)
        batch_size = len(result)

        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                    before_out_file = osp.join(out_dir, img_meta['ori_filename'].replace(".png","_before.png"))
                else:
                    out_file = None
                model.module.show_result(
                    img_show,
                    before_result[i],
                    show=show,
                    out_file=before_out_file,
                    score_thr=show_score_thr)

                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
