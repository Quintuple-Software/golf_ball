# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import argparse
import math
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable

from scipy.misc import imread
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import vis_detections_2
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models',
                      default="models")
  parser.add_argument('--input_image_dir', dest='input_image_dir',
                      help='directory to load images for demo',
                      default="images/input")
  parser.add_argument('--output_image_dir', dest='output_image_dir',
                      help='directory to save images for demo',
                      default="images/output")
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10021, type=int)
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  parser.add_argument('--webcam_num', dest='webcam_num',
                      help='webcam ID number',
                      default=-1, type=int)
  parser.add_argument('--video_path', dest='video_path',
                      help='video file path',
                      default="", type=str)
  parser.add_argument('--out_path', dest='out_path',
                      help='output video file path',
                      default="", type=str)
  parser.add_argument('--init_x', dest='init_x',
                      help='initial position',
                      default="0", type=int)
  parser.add_argument('--init_y', dest='init_y',
                      help='initial position',
                      default="0", type=int)
  parser.add_argument('--start_frame', dest='start_frame',
                      help='start frame index',
                      default="0", type=int)

  args = parser.parse_args()
  return args

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def _get_sub_image(im, pos, ratio):
    margin = int(168 / ratio)
    h, w = im.shape[0], im.shape[1]

    top = pos[1] - margin
    left = pos[0] - margin
    bottom = pos[1] + margin
    right = pos[0] + margin

    if top < 0:
        top = 0
        bottom = top + margin * 2
    if left < 0:
        left = 0
        right = left + margin * 2
    if right >= w:
        right = w - 1
        left = right - margin * 2
    if bottom >= h:
        bottom = h - 1
        top = bottom - margin * 2

    width = int((right - left)*ratio)
    height = int((bottom - top)*ratio)

    offset = (left, top)
    return cv2.resize(im[top:bottom, left:right], (width, height), interpolation=cv2.INTER_CUBIC), offset

def _get_best_det(dets, last_pos, offset, ratio, last_ball_size, thresh):
    best_bbox = None
    best_score = 0
    best_ratio = 2

    noise = 20 / ratio * 10
    size_factor = 1.8

    count = np.minimum(10, dets.shape[0])
    if count > 1:
      a = 1
    for i in range(count):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        bbox = (
          int(bbox[0] / ratio) + offset[0],
          int(bbox[1] / ratio) + offset[1],
          int(bbox[2] / ratio) + offset[0],
          int(bbox[3] / ratio) + offset[1]
          )
        score = dets[i, -1]

        # filter by score
        if score < thresh:
          continue

        dist = (bbox[0] + bbox[2] - last_pos[0]*2) * (bbox[0] + bbox[2] - last_pos[0]*2) + (bbox[1] + bbox[3] - last_pos[1]*2) * (bbox[1] + bbox[3] - last_pos[1]*2)
        dist = math.sqrt(dist)
        
        # filter by distance
        if dist > noise:
          continue        
        
        # get the best size fit
        ball_size = calc_ball_size(bbox)
        size_ratio = 1
        if last_ball_size is not None:
          size_ratio = ball_size / last_ball_size
          if size_ratio < 1:
            size_ratio = 1 / size_ratio
          #if ball_size > last_ball_size * size_factor or ball_size < last_ball_size / size_factor:
          #  continue

        if best_ratio > size_ratio:
            best_score = score
            best_bbox = bbox
            best_ratio = size_ratio

    return best_bbox, best_score

def calc_ratio(ball_size, expected_ball_size):
  ratio = expected_ball_size / ball_size
  ratio = int(math.pow(2, math.ceil(math.log2(ratio))))
  if ratio < 1:
    ratio = 1
  if ratio > 16:
    ratio = 16
  return ratio

def calc_ball_size(det):
  ball_size = (det[2] - det[0]) * (det[3] - det[1])
  ball_size = math.sqrt(ball_size)
  return ball_size

def predict(tracks):
  if len(tracks) == 0:
    return None
  if len(tracks) == 1:
    return tracks[0]
  pt2 = tracks[-1]
  pt1 = tracks[-2]
  return (pt2[0]*2-pt1[0], pt2[1]*2-pt1[1])

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  cfg.USE_GPU_NMS = args.cuda

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.

  input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

  pascal_classes = np.asarray(['__background__',
                       'golfball'])

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  print("load checkpoint %s" % (load_name))
  if args.cuda > 0:
    checkpoint = torch.load(load_name)
  else:
    checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']


  print('load model successfully!')

  # pdb.set_trace()

  print("load checkpoint %s" % (load_name))

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda > 0:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data, volatile=True)
  im_info = Variable(im_info, volatile=True)
  num_boxes = Variable(num_boxes, volatile=True)
  gt_boxes = Variable(gt_boxes, volatile=True)

  if args.cuda > 0:
    cfg.CUDA = True

  if args.cuda > 0:
    fasterRCNN.cuda()

  fasterRCNN.eval()

  start = time.time()
  max_per_image = 100
  thresh = 0.05
  lost_count_thresh = 10
  vis = True

  webcam_num = args.webcam_num
  video_path = args.video_path
  # Set up webcam or get image directories
  if webcam_num >= 0 :
    cap = cv2.VideoCapture(webcam_num)
    num_images = 0
  elif video_path != "":
    cap = cv2.VideoCapture(video_path)
    num_images = 0
    out = None
  else:
    imglist = os.listdir(args.input_image_dir)
    num_images = len(imglist)

  print('Loaded Photo: {} images.'.format(num_images))

  tracks = []
  pos = (args.init_x, args.init_y)

  zoom_ratio = 1
  expected_ball_size = 15
  lost_count = 0
  last_ball_size = None

  f = open("output.txt", "w")
  f.write(f'frame,x,y,w,h,track,size,zoom\n')

  frame_index = 0
  while (num_images >= 0):
      total_tic = time.time()
      frame_index = frame_index + 1
      if webcam_num == -1 and video_path == "":
        num_images -= 1

      # Get image from the webcam
      if webcam_num >= 0:
        if not cap.isOpened():
          raise RuntimeError("Webcam could not open. Please check connection.")
        ret, frame = cap.read()
        im_in = np.array(frame)
      elif video_path != "":
        if not cap.isOpened():
          raise RuntimeError("Video could not open. Please check video file path.")
        ret, frame = cap.read()
        if not ret:
          break
        im_in = np.array(frame)

        if out is None:
          fps = cap.get(cv2.CAP_PROP_FPS)
          frame_width = int(cap.get(3))
          frame_height = int(cap.get(4))
          out = cv2.VideoWriter(args.out_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width,frame_height))
      # Load the demo image
      else:
        im_file = os.path.join(args.input_image_dir, imglist[num_images])
        # im = cv2.imread(im_file)
        im_in = np.array(imread(im_file))
      if len(im_in.shape) == 2:
        im_in = im_in[:,:,np.newaxis]
        im_in = np.concatenate((im_in,im_in,im_in), axis=2)
      # rgb -> bgr
      im = im_in[:,:,::-1]

      if frame_index < args.start_frame:
        continue

      ratio = zoom_ratio
      sub_im, offset = _get_sub_image(im, pos, ratio=ratio)
      cv2.imwrite("sub_im.jpg", cv2.cvtColor(sub_im, cv2.COLOR_BGR2RGB))
      
      blobs, im_scales = _get_image_blob(sub_im)
      assert len(im_scales) == 1, "Only single-image batch implemented"
      im_blob = blobs
      im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

      im_data_pt = torch.from_numpy(im_blob)
      im_data_pt = im_data_pt.permute(0, 3, 1, 2)
      im_info_pt = torch.from_numpy(im_info_np)

      with torch.no_grad():
              im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
              im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
              gt_boxes.resize_(1, 1, 5).zero_()
              num_boxes.resize_(1).zero_()

      # pdb.set_trace()
      det_tic = time.time()

      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

      scores = cls_prob.data
      boxes = rois.data[:, :, 1:5]

      if cfg.TEST.BBOX_REG:
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred.data
          if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
          # Optionally normalize targets by a precomputed mean and stdev
            if args.class_agnostic:
                if args.cuda > 0:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                box_deltas = box_deltas.view(1, -1, 4)
            else:
                if args.cuda > 0:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

          pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
          pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
      else:
          # Simply repeat the boxes, once for each class
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))

      pred_boxes /= im_scales[0]

      scores = scores.squeeze()
      pred_boxes = pred_boxes.squeeze()
      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()
      if vis:
          im2show = np.copy(im)
      detected = False
      for j in xrange(1, len(pascal_classes)):
          inds = torch.nonzero(scores[:,j]>thresh).view(-1)
          # if there is det
          if inds.numel() > 0:
            cls_scores = scores[:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
              cls_boxes = pred_boxes[inds, :]
            else:
              cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            # keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
            keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]

            det, score = _get_best_det(cls_dets.cpu().numpy(), pos, offset, ratio, last_ball_size, thresh=0.5)
            if det is not None:
              # update position
              last_det = det
              pos = (int((det[0] + det[2])/2), int((det[1] + det[3])/2))

              # update size ratio and ball size
              ball_size = calc_ball_size(det)
              cur_ratio = calc_ratio(ball_size, expected_ball_size)
              if zoom_ratio < cur_ratio:
                zoom_ratio = cur_ratio
              if last_ball_size is None or last_ball_size > ball_size:
                last_ball_size = int(ball_size)

              detected = True
            if vis:
              im2show = vis_detections_2(im2show, det, score)

      if detected:
        lost_count = 0
        print(f'  Zoom: {zoom_ratio}x  Size: {int(last_ball_size)}  tracked')
      else:
        lost_count = lost_count + 1
        pos = predict(tracks)
        print(f'  lost {lost_count}')

      tracks.append(pos)

      f.write(f'{frame_index},{last_det[0]},{last_det[1]},{last_det[2]-last_det[0]},{last_det[3]-last_det[1]},{detected},{last_ball_size},{zoom_ratio}\n')
      f.flush()

      if lost_count >= lost_count_thresh:
        print('  break')
        break

      misc_toc = time.time()
      nms_time = misc_toc - misc_tic

      if webcam_num == -1 and video_path == "":
          sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                           .format(num_images + 1, len(imglist), detect_time, nms_time))
          sys.stdout.flush()

      if vis and webcam_num == -1 and video_path == "":
          # cv2.imshow('test', im2show)
          # cv2.waitKey(0)
          result_path = os.path.join(args.output_image_dir, imglist[num_images][:-4] + "_det.jpg")
          cv2.imwrite(result_path, im2show)
      else:
          im2showRGB = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
          #result_path = os.path.join(args.output_image_dir, f"{frame_index}_det.jpg")
          #cv2.imwrite(result_path, im2showRGB)
          out.write(im2showRGB)
          cv2.imwrite("temp.jpg", im2showRGB)
          #cv2.imshow("frame", im2showRGB)
          total_toc = time.time()
          total_time = total_toc - total_tic
          frame_rate = 1 / total_time
          print(f'[{frame_index}] [{pos[0]}, {pos[1]}] Frame rate:', frame_rate)
          #if cv2.waitKey(1) & 0xFF == ord('q'):
          #    break
  if webcam_num >= 0 or video_path != "":
      cap.release()
      out.release()
      #cv2.destroyAllWindows()
      f.close()
