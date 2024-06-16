import csv

# with open("/home/skm21/fairseq-0.12.0/data_oeis/oeis_data_all_keyword.csv",'r',encoding='utf-8')as f:
#     with open("/home/skm21/fairseq-0.12.0/data_oeis/oeis_data_all_keyword2.csv",'w',encoding='utf-8')as f2:
#         reader=csv.reader(f)
#         writer=csv.writer(f2)
#         for row in reader:
#             row[1]=''
#             writer.writerow(row)
import json

with open('/home/skm21/fairseq-0.12.0/checkpoints/4500w_combine_train_36w/iter30/result/final_res.json', 'r',
          encoding='utf-8') as f:
    dic = json.load(f)
    print(len(dic))
