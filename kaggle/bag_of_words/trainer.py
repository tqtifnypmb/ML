import cnn
import argparse
import os

def parse_params(args_parser):
    args_parser.add_argument(
        '--job-dir',
        required=True
    )

    args_parser.add_argument(
        '--train-file',
        required=True
    )

    args_parser.add_argument(
        '--output-file',
        required=True
    )

    args_parser.add_argument(
        '--batch-size',
        type=int,
        required=True
    )

    args_parser.add_argument(
        '--output-filters',
        type=int,
        required=True
    )

    args_parser.add_argument(
        '--test-file',
        required=True
    )

    args_parser.add_argument(
        '--max-len',
        type=int,
        required=True,
    )

    args_parser.add_argument(
        '--num-epoches',
        type=int,
        required=True
    )

    return args_parser.parse_args()

data_set = {
    'train': 'data/labeledTrainData.tsv',
    'test': 'data/testData.tsv'
}

truncated_data_set = {
    'train': 'data/labeledTrainData_truncated.tsv',
    'test': 'data/testData_truncated.tsv'
}

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    paras = parse_params(args_parser)

    output = os.path.join(paras.job_dir, paras.output_file)
    cnn.main(paras.train_file,
             paras.test_file,
             output,
             paras.job_dir,
             paras.max_len,
             paras.batch_size,
             paras.num_epoches,
             paras.output_filters)