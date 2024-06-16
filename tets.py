import argparse
import pickle


def save_args(args, filename='args.pkl'):  # 保存参数
    with open(filename, 'wb') as f:
        pickle.dump(args, f)


def load_args(filename='args.pkl'):  # 加载参数
    with open(filename, 'rb') as f:
        return pickle.load(f)


parser = argparse.ArgumentParser(usage="参数", description="help info.")

## 基础参数
parser.add_argument('--sta-iter-id', type=int, default=1, help='开始迭代的id,默认从1开始')
parser.add_argument('--end-iter-id', type=int, default=5, help='迭代结束的id,默认以5结束')
parser.add_argument('--model-type', type=str, default="test_demo", help='迭代的名称，每个迭代最好设置一个不同的名称，防止检查点被覆盖')
parser.add_argument('--is-combine', type=bool, default=True, help='是否采用联合训练的方式，联合训练在每次迭代时同时训练两个模型，详细情况参见大论文第四章')

args = parser.parse_args()

print("sta_iter_id:", args.sta_iter_id)
print("sta_iter_id:", type(args.sta_iter_id))

print("end_iter_id:", args.end_iter_id)
print("end_iter_id:", type(args.end_iter_id))

print("model-type:", args.model_type)
print("model-type:", type(args.model_type))

print("is-combine:", args.is_combine)
print("is-combine:", type(args.is_combine))
print(args)
print(type(args))

# 保存参数到文件
save_args(args)
args2 = load_args()
print(args2)
print(type(args2))
