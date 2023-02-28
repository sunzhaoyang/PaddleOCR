import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
import uuid
from decimal import Decimal
from logging.handlers import RotatingFileHandler
from tempfile import NamedTemporaryFile
from typing import Union

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Form, UploadFile
from pydantic import BaseModel

import tools.infer.utility as utility
from paddleocr import PaddleOCR
from ppocr.data import create_operators, transform
from ppocr.postprocess import build_post_process
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import check_and_read, get_image_file_list
from ppocr.utils.visual import draw_ser_results
from ppstructure.utility import parse_args

app = FastAPI()

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(handlers=[RotatingFileHandler('/tmp/sight_out.log', maxBytes=2000, backupCount=10)],
                    level=logging.DEBUG, format=LOG_FORMAT)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

def str2bool(v):
    return v.lower() in ("true", "t", "1")


def init_args():
    parser = argparse.ArgumentParser()
    # params for prediction engine
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--use_xpu", type=str2bool, default=False)
    parser.add_argument("--use_npu", type=str2bool, default=False)
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--min_subgraph_size", type=int, default=15)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--gpu_mem", type=int, default=500)
    parser.add_argument("--gpu_id", type=int, default=0)

    # params for text detector
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--page_num", type=int, default=0)
    parser.add_argument("--det_algorithm", type=str, default='DB')
    parser.add_argument("--det_model_dir", type=str)
    parser.add_argument("--det_limit_side_len", type=float, default=960)
    parser.add_argument("--det_limit_type", type=str, default='max')
    parser.add_argument("--det_box_type", type=str, default='quad')

    # DB parmas
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.6)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.5)
    parser.add_argument("--max_batch_size", type=int, default=10)
    parser.add_argument("--use_dilation", type=str2bool, default=False)
    parser.add_argument("--det_db_score_mode", type=str, default="fast")

    # EAST parmas
    parser.add_argument("--det_east_score_thresh", type=float, default=0.8)
    parser.add_argument("--det_east_cover_thresh", type=float, default=0.1)
    parser.add_argument("--det_east_nms_thresh", type=float, default=0.2)

    # SAST parmas
    parser.add_argument("--det_sast_score_thresh", type=float, default=0.5)
    parser.add_argument("--det_sast_nms_thresh", type=float, default=0.2)

    # PSE parmas
    parser.add_argument("--det_pse_thresh", type=float, default=0)
    parser.add_argument("--det_pse_box_thresh", type=float, default=0.85)
    parser.add_argument("--det_pse_min_area", type=float, default=16)
    parser.add_argument("--det_pse_scale", type=int, default=1)

    # FCE parmas
    parser.add_argument("--scales", type=list, default=[8, 16, 32])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--fourier_degree", type=int, default=5)

    # params for text recognizer
    parser.add_argument("--rec_algorithm", type=str, default='SVTR_LCNet')
    parser.add_argument("--rec_model_dir", type=str)
    parser.add_argument("--rec_image_inverse", type=str2bool, default=True)
    parser.add_argument("--rec_image_shape", type=str, default="3, 48, 320")
    parser.add_argument("--rec_batch_num", type=int, default=6)
    parser.add_argument("--max_text_length", type=int, default=25)
    parser.add_argument(
        "--rec_char_dict_path",
        type=str,
        default="./ppocr/utils/ppocr_keys_v1.txt")
    parser.add_argument("--use_space_char", type=str2bool, default=True)
    parser.add_argument(
        "--vis_font_path", type=str, default="/opt/PaddleOCR/doc/fonts/simfang.ttf")
    parser.add_argument("--drop_score", type=float, default=0.5)

    # params for e2e
    parser.add_argument("--e2e_algorithm", type=str, default='PGNet')
    parser.add_argument("--e2e_model_dir", type=str)
    parser.add_argument("--e2e_limit_side_len", type=float, default=768)
    parser.add_argument("--e2e_limit_type", type=str, default='max')

    # PGNet parmas
    parser.add_argument("--e2e_pgnet_score_thresh", type=float, default=0.5)
    parser.add_argument(
        "--e2e_char_dict_path", type=str, default="./ppocr/utils/ic15_dict.txt")
    parser.add_argument("--e2e_pgnet_valid_set", type=str, default='totaltext')
    parser.add_argument("--e2e_pgnet_mode", type=str, default='fast')

    # params for text classifier
    parser.add_argument("--use_angle_cls", type=str2bool, default=False)
    parser.add_argument("--cls_model_dir", type=str)
    parser.add_argument("--cls_image_shape", type=str, default="3, 48, 192")
    parser.add_argument("--label_list", type=list, default=['0', '180'])
    parser.add_argument("--cls_batch_num", type=int, default=6)
    parser.add_argument("--cls_thresh", type=float, default=0.9)

    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--cpu_threads", type=int, default=10)
    parser.add_argument("--use_pdserving", type=str2bool, default=False)
    parser.add_argument("--warmup", type=str2bool, default=False)

    # SR parmas
    parser.add_argument("--sr_model_dir", type=str)
    parser.add_argument("--sr_image_shape", type=str, default="3, 32, 128")
    parser.add_argument("--sr_batch_num", type=int, default=1)

    #
    parser.add_argument(
        "--draw_img_save_dir", type=str, default="./inference_results")
    parser.add_argument("--save_crop_res", type=str2bool, default=False)
    parser.add_argument("--crop_res_save_dir", type=str, default="./output")

    # multi-process
    parser.add_argument("--use_mp", type=str2bool, default=False)
    parser.add_argument("--total_process_num", type=int, default=1)
    parser.add_argument("--process_id", type=int, default=0)

    parser.add_argument("--benchmark", type=str2bool, default=False)
    parser.add_argument("--save_log_path", type=str, default="./log_output/")

    parser.add_argument("--show_log", type=str2bool, default=True)
    parser.add_argument("--use_onnx", type=str2bool, default=False)

    # KIE/SER
    # params for output
    parser.add_argument("--output", type=str, default='./output')
    # params for table structure
    parser.add_argument("--table_max_len", type=int, default=488)
    parser.add_argument("--table_algorithm", type=str, default='TableAttn')
    parser.add_argument("--table_model_dir", type=str)
    parser.add_argument(
        "--merge_no_span_structure", type=str2bool, default=True)
    parser.add_argument(
        "--table_char_dict_path",
        type=str,
        default="../ppocr/utils/dict/table_structure_dict_ch.txt")
    # params for layout
    parser.add_argument("--layout_model_dir", type=str)
    parser.add_argument(
        "--layout_dict_path",
        type=str,
        default="../ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt")
    parser.add_argument(
        "--layout_score_threshold",
        type=float,
        default=0.5,
        help="Threshold of score.")
    parser.add_argument(
        "--layout_nms_threshold",
        type=float,
        default=0.5,
        help="Threshold of nms.")
    # params for kie
    parser.add_argument("--kie_algorithm", type=str, default='LayoutXLM')
    #parser.add_argument("--ser_model_dir", type=str, default="/opt/model/recognize-2/ser/v1/")
    parser.add_argument("--ser_model_dir", type=str, default="/opt/model/ser/2/v2.0/best_accuracy/export/")
    parser.add_argument("--re_model_dir", type=str)
    parser.add_argument("--use_visual_backbone", type=str2bool, default=True)
    #parser.add_argument("--ser_dict_path",type=str,default="/opt/model/recognize-2/ser/v1/class_list_xfun.txt")
    parser.add_argument("--ser_dict_path",type=str,default="/opt/model/ser/2/v2.0/predefined_classes.txt")
    # need to be None or tb-yx
    parser.add_argument("--ocr_order_method", type=str, default=None)
    # params for inference
    parser.add_argument(
        "--mode",
        type=str,
        choices=['structure', 'kie'],
        default='structure',
        help='structure and kie is supported')
    parser.add_argument(
        "--image_orientation",
        type=bool,
        default=False,
        help='Whether to enable image orientation recognition')
    parser.add_argument(
        "--layout",
        type=str2bool,
        default=True,
        help='Whether to enable layout analysis')
    parser.add_argument(
        "--table",
        type=str2bool,
        default=True,
        help='In the forward, whether the table area uses table recognition')
    parser.add_argument(
        "--ocr",
        type=str2bool,
        default=True,
        help='In the forward, whether the non-table area is recognition by ocr')
    # param for recovery
    parser.add_argument(
        "--recovery",
        type=str2bool,
        default=False,
        help='Whether to enable layout of recovery')
    parser.add_argument(
        "--use_pdf2docx_api",
        type=str2bool,
        default=False,
        help='Whether to use pdf2docx api')

    return parser


def parse_args():
    parser = init_args()
    return parser.parse_args()

def random_str():
    return str(uuid.uuid4())

class SerPredictor(object):
    def __init__(self, args):
        self.ocr_engine = PaddleOCR(
            use_angle_cls=False,
            det_model_dir=args.det_model_dir,
            rec_model_dir=args.rec_model_dir,
            show_log=False,
            use_gpu=args.use_gpu)

        pre_process_list = [{
            'VQATokenLabelEncode': {
                'algorithm': args.kie_algorithm,
                'class_path': args.ser_dict_path,
                'contains_re': False,
                'ocr_engine': self.ocr_engine,
                'order_method': args.ocr_order_method,
            }
        }, {
            'VQATokenPad': {
                'max_seq_len': 512,
                'return_attention_mask': True
            }
        }, {
            'VQASerTokenChunk': {
                'max_seq_len': 512,
                'return_attention_mask': True
            }
        }, {
            'Resize': {
                'size': [224, 224]
            }
        }, {
            'NormalizeImage': {
                'std': [58.395, 57.12, 57.375],
                'mean': [123.675, 116.28, 103.53],
                'scale': '1',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': [
                    'input_ids', 'bbox', 'attention_mask', 'token_type_ids',
                    'image', 'labels', 'segment_offset_id', 'ocr_info',
                    'entities'
                ]
            }
        }]
        postprocess_params = {
            'name': 'VQASerTokenLayoutLMPostProcess',
            "class_path": args.ser_dict_path,
        }

        self.preprocess_op = create_operators(pre_process_list,
                                              {'infer_mode': True})
        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors, self.config = \
            utility.create_predictor(args, 'ser', logger)

    def __call__(self, img):
        ori_im = img.copy()
        img_data = {'image': img}
        data = transform(img_data, self.preprocess_op)
        if not data or data[0] is None:
            return None, 0
        starttime = time.time()

        for idx in range(len(data)):
            if isinstance(data[idx], np.ndarray):
                data[idx] = np.expand_dims(data[idx], axis=0)
            else:
                data[idx] = [data[idx]]

        for idx in range(len(self.input_tensor)):
            self.input_tensor[idx].copy_from_cpu(data[idx])

        self.predictor.run()

        outputs = []
        for output_tensor in self.output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)
        preds = outputs[0]

        post_result = self.postprocess_op(
            preds, segment_offset_ids=data[6], ocr_infos=data[7])
        elapse = time.time() - starttime
        return post_result, data, elapse

args = parse_args()
ser_predictor = SerPredictor(args)

class Result(BaseModel):
    os_al: Union[Decimal, None] = None
    os_k1: Union[Decimal, None]= None
    os_k2:Union[Decimal, None]= None
    od_al:Union[Decimal, None]= None
    od_k1:Union[Decimal, None]= None
    od_k2:Union[Decimal, None]= None

class ImageResult(BaseModel):

    al: Union[Decimal, None] = None
    k1: Union[Decimal, None] = None
    k2: Union[Decimal, None] = None

@app.post("/api/ocr/ser")
def recognize_eyeball_report(image_file:UploadFile = Form()):

    with NamedTemporaryFile(suffix='.jpg') as f:
        shutil.copyfileobj(image_file.file, f)

        img, flag, _ = check_and_read(f.name)
        if not flag:
            img = cv2.imread(f.name)
            img = img[:, :, ::-1]
        if img is None:
            logger.info("error in loading image:{}".format(f.name))
            return None
        ser_res, _, elapse = ser_predictor(img)
        ser_res = ser_res[0]
        print(json.dumps(ser_res))
        result = {
            'os':ImageResult(),
            'od':ImageResult()
        }

    for shape in ser_res:
        key = shape.get('pred').lower()
        if key == 'od_al':
            match = re.search('''(\d+\.\d+)''',shape.get('transcription'))
            if match:
                groups = match.groups()
                result['od'].al = Decimal(groups[0])
        if key == 'os_al':
            match = re.search('''(\d+\.\d+)''',shape.get('transcription'))
            if match:
                groups = match.groups()
                result['os'].al = Decimal(groups[0])

        elif key == 'os_k':
            match = re.search('''(\d+\.\d+)/(\d+\.\d+)''',shape.get('transcription'))
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    result['os'].k1 = Decimal(groups[0])
                    result['os'].k2 = Decimal(groups[1])
        elif key == 'od_k':
            match = re.search('''(\d+\.\d+)/(\d+\.\d+)''',shape.get('transcription'))
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    result['od'].k1 = Decimal(groups[0])
                    result['od'].k2 = Decimal(groups[1])
        
    return result

if __name__ == "__main__":
    uvicorn.run(app, headers=[("ocr-ser", "ocr http server")], host="0.0.0.0", port=80)
