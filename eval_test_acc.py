import argparse
import os
import re

from check_auto_formula import eval_testset


def eval_test(args):
    """
    评估测试集的 acc
    """
    gpuid = args.gpuid
    test_type = args.test_type
    max_degree = args.max_degree
    input_seq_len = args.input_seq_len
    beam_size = args.beam_size
    nbest = args.nbest
    model_path = args.model_path
    batch_size = args.batch_size
    max_tokens = args.max_tokens
    data_type = args.data_type
    vocab_type = args.vocab_type

    # env, file_name = get_env()

    index_lis = [substr.start() for substr in re.finditer('/', model_path)]
    model_name_path = model_path[index_lis[-2] + 1:]  # 获取model_name  model_name_path  "sum_data_sl"
    model_name_path = model_name_path.replace('/', '_')

    if data_type == 'random':
        data_bin_path = f'data-bin/random_data1w_{vocab_type}'
        results_path = f'result/test/random1wData_{model_name_path}_beamsize{beam_size}_nbest{nbest}/'
        test_path = "data_oeis/random_1w_testData.json"
    else:
        results_path = f'{os.path.dirname(model_path)}/{model_name_path}_{test_type}_len{input_seq_len}_beamsize{beam_size}_nbest{nbest}/'
        test_path = f"data_oeis/1wan_{test_type}_testdata_{input_seq_len + 10}.csv"

        if input_seq_len == 15:
            data_bin_path = f'data-bin/{test_type}1w_25_vocab4'
        elif max_degree == 12:
            data_bin_path = f'data-bin/{test_type}1w'
        elif max_degree == 6:
            data_bin_path = f'data-bin/{test_type}1w_0-6recur'
        else:
            data_bin_path = f'data-bin/{test_type}1w_{vocab_type}'

    print("databin:", data_bin_path)
    print("test_path:", test_path)
    # exit()

    jiema = f"CUDA_VISIBLE_DEVICES={gpuid} fairseq-generate {data_bin_path} \
        --path {model_path} \
        --max-tokens {max_tokens} --batch-size {batch_size} --beam {beam_size} --nbest {nbest} --results-path {results_path}"
    proce_res = f"grep ^H {results_path}/generate-test.txt | sort -n -k 2 -t '-' | cut -f 3 > {results_path}pre_res.txt"
    #
    # os.system(jiema)
    # os.system(proce_res)

    formula_res_path = f"{results_path}pre_res.txt"
    correct_res_save_path = f"{results_path}" + 'result.csv'
    acc_res_save_path = f"{results_path}" + 'result.txt'
    if data_type == 'random':
        eval_testset_random(test_path, formula_res_path, nbest, input_seq_len, correct_res_save_path, acc_res_save_path)
    else:
        eval_testset(test_path, formula_res_path, nbest, input_seq_len, correct_res_save_path, acc_res_save_path)


if __name__ == '__main__':
    parser2 = argparse.ArgumentParser(description='eval_test_acc')

    # parser2.add_argument('--gpuid', type=int, default=0, help='使用的gpuid')

    parser2.add_argument('--gpuid', type=int, default=2, help='使用的gpuid')
    parser2.add_argument('--test_type', type=str, default="base", help='选择待测试的数据及类型，有三种：easy,sign,base')
    parser2.add_argument('--max_degree', type=int, default=-1, help='公式的最大递推度，会影响到词表的选择，进而影响bin文件的选择')
    parser2.add_argument('--input_seq_len', type=int, default=25, help='测试集输入序列的项数')
    parser2.add_argument('--beam_size', type=int, default=32, help='解码束搜索的宽度')
    parser2.add_argument('--nbest', type=int, default=32, help='输出的候选解数量')
    parser2.add_argument('--model_path', type=str,
                         default="/home/skm21/fairseq-0.12.0/checkpoints/4500w_combine_train_36w/iter50/model0/checkpoint_best.pt",
                         help='待测试模型路径')

    parser2.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser2.add_argument('--max_tokens', type=int, default=4096, help='批次中最大tokens数量')

    parser2.add_argument('--data_type', type=str, default='oeis',
                         help='测试数据类型为两类，一类是random: 随机生成的1w条,第二类是oeis: 测试OEIS测试集')
    parser2.add_argument('--vocab_type', type=str, default='vocab4', help='词表类型有三种：vocab2,vocab3,vocab4')

    args2 = parser2.parse_args()

    eval_test(args2)
