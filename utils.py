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


def data_preap(wav_dir, data_dir):
    wav_scp = open(os.path.join(data_dir, 'wav.scp'), 'w', encoding='utf-8')
    text = open(os.path.join(data_dir, 'text'), 'w', encoding='utf-8')
    utt2spk = open(os.path.join(data_dir, 'utt2spk'), 'w', encoding='utf-8')
    spk2utt = open(os.path.join(data_dir, 'spk2utt'), 'w', encoding='utf-8')

    for wav in os.listdir(wav_dir):
        wav_path = os.path.join(wav_dir, wav)
        uttid = wav.split('.')[0]
        wav_scp.write(uttid + ' ' + wav_path + '\n')
        text.write(uttid + ' ' + 'X' + '\n')
        utt2spk.write(uttid + ' ' + uttid + '\n')
        spk2utt.write(uttid + ' ' + uttid + '\n')
    wav_scp.close()
    text.close()
    utt2spk.close()
    spk2utt.close()
    subprocess.run(r'utils/fix_data_dir.sh ' + data_dir, shell=True)


def feature_extraction(data_dir, cmvn, log_dir, decode_dir, nj=1, do_delta='false'):
    if do_delta not in ['true', 'false']:
        print('Error:', 'do_delta value must be true or false')
        exit(1)
    raw_feat_dir = os.path.join(decode_dir, 'fbank')
    subprocess.run(r'steps/make_fbank_pitch.sh --cmd run.pl --nj ' + str(
        nj) + ' --write_utt2num_frames true ' + data_dir + ' ' + log_dir + ' ' + raw_feat_dir, shell=True)
    feat_recog_dir = os.path.join(decode_dir, 'dump')
    if not os.path.exists(feat_recog_dir):
        os.makedirs(feat_recog_dir)

    subprocess.run("source ./path.sh\ndump.sh --cmd run.pl --nj " + str(
        nj) + ' --do_delta ' + do_delta + ' ' + data_dir + '/feats.scp' + ' ' + cmvn + ' ' + log_dir + ' ' + feat_recog_dir,
                   shell=True)


def data2json(data_dir, decode_dir):
    dict = os.path.join(data_dir, 'dict')
    subprocess.run(r'echo "<unk> 1" > ' + dict, shell=True)
    feat_recog_dir = os.path.join(decode_dir, 'dump')
    subprocess.run(
        'source ./path.sh\ndata2json.sh --feat ' + feat_recog_dir + '/feats.scp ' + data_dir + ' ' + dict + '> ' + feat_recog_dir + '/data.json',
        shell=True)
    # subprocess.run('rm -f '+dict)


def decode(decode_config, recog_json, result_json, model, rnnlm=None, ngpu=1):
    if ngpu > 0:
        api = 'v2'
    else:
        api = 'v1'
    parser = ['--config', decode_config, '--ngpu', str(ngpu), '--backend', 'pytorch', '--recog-json', recog_json,
              '--result-label', result_json, '--model', model, '--api', api, '--rnnlm', rnnlm]
    from espnet.bin.asr_recog import main
    main(parser)


def getResults(result_json, result_file):
    with open(result_json, 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
    keys = json_data['utts'].keys()
    with open(result_file, 'w', encoding='utf-8') as f:
        for k in keys:
            # print(k, json_data['utts'][k]['output'][0]['rec_text'].replace('<eos>', ''))
            f.write(k + ' ' + json_data['utts'][k]['output'][0]['rec_text'].replace('<eos>', '') + '\n')


def main(args):
    # the directory includes audio files that need to be recognized
    wavs_dir = args.wavs_dir
    # wavs_dir = '/home/acc12416pz/espnet/egs/csj/demo/xxxx'

    # the decoding dir, store the result and some temporary files of this recognition
    decode_dir = os.path.join('decode_x', os.path.basename(wavs_dir.rstrip('/')))

    data_dir = os.path.join(decode_dir, 'data')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    print('\n====stage 1 data prepration !====')
    data_preap(wavs_dir, data_dir)

    print('\n====stage 2 feature extraction !====')
    # cmvn file
    cmvn = args.cmvn
    feature_extraction(data_dir, cmvn, decode_dir + '/log_dir', decode_dir, nj=2)

    print('\n====stage 2 json Data Preparation !====')
    # Combine the previously prepared data files into a json file
    data2json(data_dir, decode_dir)

    print('\n====stage 3 decoding !====')
    decode_config = args.config
    acoustic_model = args.am
    rnnlm = args.lm
    ngpu = int(args.ngpu)
    decode(decode_config, decode_dir + '/dump/data.json', decode_dir + '/results.json',
           acoustic_model, rnnlm=rnnlm, ngpu=ngpu)

    print('\n====stage 4 summary results !====')
    # decoding results will be showed on a json file(detail info) and a text file (brief info)
    getResults(decode_dir + '/results.json', decode_dir + '/result_text')
    print('the recognition results of {} is stored at {}'.format(wavs_dir, decode_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('wavs_dir', type=str, help='audio files directory')
    parser.add_argument('--cmvn', type=str, default='models/cmvn.ark')
    parser.add_argument('--config', type=str, default='models/decode.yaml')
    parser.add_argument('--am', type=str, default='models/acoustic_model/model.acc.best')
    parser.add_argument('--lm', type=str, default='models/lm_model/rnnlm.model.best')
    parser.add_argument('--ngpu', type=int, default=0)
    parser.add_argument('--nj', type=int, default=2, help='the number of processes')

    cmd_args = parser.parse_args()
    main(cmd_args)
