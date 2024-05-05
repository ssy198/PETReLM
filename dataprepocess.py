import os
import random
import numpy as np
import binascii
from scapy.all import rdpcap
import json
from scapy.layers.inet import IP


if __name__ == '__main__':
    dict_from_path = r'CIC_IoT_Dataset2023'
    num_for_each_cat_ = 5000
    random.seed(913354)

    dirs_dict = {}

    for _, from_dirs, _ in os.walk(dict_from_path):
        cat_classify = len(from_dirs)
        print('detected '+str(cat_classify)+' categories. '+str(from_dirs))
        break
    for i in range(len(from_dirs)):
        dirs_dict[from_dirs[i]] = i
    with open(dict_from_path+'\\class.txt', 'w', encoding='utf-8') as f_f:
        f_f.write(json.dumps(dirs_dict))

    with open(dict_from_path+'\\train.jsonl', 'w', encoding='utf-8') as f_t,\
            open(dict_from_path+'\\valid.jsonl', 'w', encoding='utf-8') as f_v,\
            open(dict_from_path+'\\test.jsonl', 'w', encoding='utf-8') as f_s:
        for cat_dir in from_dirs:
            cat_label = str(dirs_dict[cat_dir])
            print('handling', cat_dir, cat_label)
            pcap_files = [f for f in os.listdir(dict_from_path+'/'+cat_dir) if f.endswith('.pcap')]
            packets_from = []
            packets_to = []
            all_len = []
            for pcap_file in pcap_files:
                pcap_path = dict_from_path+'/'+cat_dir+'/'+pcap_file
                packets_from.extend(rdpcap(pcap_path))
            for p in packets_from:
                if IP in p:
                    # delete IP
                    p = p[IP].payload
                else:
                    # print('opps no ip')
                    continue
                word_packet = p.copy()
                words = (binascii.hexlify(bytes(word_packet)))
                all_len.append(len(words.decode())/2)
                words_string = words.decode()[:1026]

                if len(words_string) > 20:
                    words_to = ''
                    sentence = [words_string[i:i + 2] for i in range(0, len(words_string), 2)]
                    for sub_string_index in range(len(sentence)):
                        if sub_string_index != (len(sentence) - 1):
                            merge_word_bigram = sentence[sub_string_index] + sentence[sub_string_index + 1]
                            words_to = words_to+merge_word_bigram+' '
                        else:
                            break
                    packets_to.append(words_to.strip())

            print('from ', len(packets_to))
            random.shuffle(packets_to)
            num_for_each_cat = min(num_for_each_cat_, len(packets_to))
            num_train = int(num_for_each_cat * 0.8)
            num_validation = int(num_for_each_cat * 0.1)
            num_test = int(num_for_each_cat * 0.1)
            print('select', num_for_each_cat, num_train, num_validation, num_test)

            for s in packets_to[:num_train]:
                f_t.write('{\"sentence\":\"')
                f_t.write(s)
                f_t.write('\",\"labels\":'+cat_label+'}\n')

            for s in packets_to[num_train:num_train+num_validation]:
                f_v.write('{\"sentence\":\"')
                f_v.write(s)
                f_v.write('\",\"labels\":'+cat_label+'}\n')

            for s in packets_to[num_train+num_validation:num_train+2*num_validation]:
                f_s.write('{\"sentence\":\"')
                f_s.write(s)
                f_s.write('\",\"labels\":'+cat_label+'}\n')




