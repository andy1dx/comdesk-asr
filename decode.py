'''
@File : utils.py
@Author: Yongjie Lv
@Date : 15:20 2020/9/28
@Desc :
'''
import argparse
import json
import os
import subprocess
from utils import data2json, decode, getResults

def main(args):
  # the directory includes audio files that need to be recognized
  wavs_dir = args.wavs_dir
  # wavs_dir = '/home/acc12416pz/espnet/egs/csj/demo/xxxx'

  # the decoding dir, store the result and some temporary files of this recognition
  decode_dir = os.path.join('decode_x', os.path.basename(wavs_dir.rstrip('/')))

  data_dir = os.path.join(decode_dir, 'data')

  if not os.path.exists(data_dir):
    os.makedirs(data_dir)

  print('\n====stage 1 json Data Preparation !====')
  # Combine the previously prepared data files into a json file
  data2json(data_dir, decode_dir)

  print('\n====stage 2 decoding !====')
  decode_config = args.config
  acoustic_model = args.am
  rnnlm = args.lm
  ngpu = int(args.ngpu)
  decode(decode_config, decode_dir + '/dump/data.json', decode_dir + '/results.json',
    acoustic_model, rnnlm=rnnlm, ngpu=ngpu)

  print('\n====stage 3 summary results !====')
  # decoding results will be showed on a json file(detail info) and a text file (brief info)
  getResults(decode_dir + '/results.json', decode_dir + '/result_text')
  print('the recognition results of {} is stored at {}'.format(wavs_dir, decode_dir))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('wavs_dir', type=str, help='audio files directory')
  parser.add_argument('--config', type=str, default='models/decode.yaml')
  parser.add_argument('--am', type=str, default='models/acoustic_model/model.acc.best')
  parser.add_argument('--lm', type=str, default='models/lm_model/rnnlm.model.best')
  parser.add_argument('--ngpu', type=int, default=0)

  cmd_args = parser.parse_args()
  main(cmd_args)
