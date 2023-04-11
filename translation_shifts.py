import os
import argparse
from tqdm import tqdm
from itertools import product
from collections import defaultdict, Counter
import numpy as np


# Get raw sentence out of GIZA++ tokenization
def clean(sent):
    cleaned = [x.strip() for x in sent.split('({')][:-1]
    cleaned = [x.split()[-1] for x in cleaned][1:]
    return ' '.join(cleaned)


# Get list of translation pairs
def translations(alignment_file):
    transl = open(alignment_file, 'r', encoding='utf8').readlines()
    transl = [transl[i:i+3][1:] for i in range(0, len(transl), 3)]
    transl = [(a.strip(), clean(b)) for (a,b) in transl]
    return transl


# Get counts from alignment file
def alignment_counts(alignment_file, return_sents=False):
    alignment = open(alignment_file, 'r', encoding='utf8').readlines()
    alignment = [alignment[i:i+3] for i in range(0, len(alignment), 3)]
    al_count = defaultdict(lambda:Counter())
    al_sents = defaultdict(lambda:defaultdict(list))
    for (_, tgt, src) in tqdm(alignment, desc='Counting alignments'):
        src_orig = clean(src)
        tgt_indices = {i+1:w for (i,w) in enumerate(tgt.strip().split())}
        src = [x.strip() for x in src.split('})')][:-1]
        src = [x.split('({ ') for x in src]
        src = [x for x in src if len(x)==2]
        src = [(w.strip(), [int(i) for i in n.split() if i.isnumeric()]) for (w,n) in src]
        for w,n in src:
             if [c for c in w if c.isalnum()] and n:
                 t = ' '.join([tgt_indices[i] for i in n])
                 al_count[w][t] += 1
                 if return_sents:
                     al_sents[w][t].append((src_orig, tgt))
    if return_sents:
        return al_count, al_sents
    return al_count


# Get stats about src-tgt counts
def alignment_info(alignments, min_count=1):
    if type(alignments) == str:
        alignments = alignment_counts(alignments)
    alignments = {s:{t:alignments[s][t] for t in alignments[s] if alignments[s][t]>=min_count} for s in alignments}
    alignments = {s:alignments[s] for s in alignments if alignments[s]}
    src_words = len(alignments)
    num_tgts = [len(alignments[w]) for w in alignments]
    tgt_lengths = []
    for w in alignments:
        tgt_lengths += [len(s.split()) for s in alignments[w]]
    al_info = {'min_count': min_count,
               'source_words': src_words,
               'num_tgts_mean': np.average(num_tgts),
               'num_tgts_median': np.median(num_tgts),
               'tgt_len_mean': np.average(tgt_lengths),
               'tgt_len_median': np.median(tgt_lengths),}
    return al_info


# Get one-to-many mappings from count dict
def unit_shifts(alignments, min_count=10, most_common=None, tgt_file=None):
    if type(alignments) == str:
        alignments = alignment_counts(alignments)
    us = []
    for src in tqdm(alignments, desc='Counting unit shifts'):
        for tgt,count in alignments[src].most_common(most_common):
            if count >= min_count and len(tgt.split())>1:
                us.append((src, tgt, count))
    if tgt_file:
        with open(tgt_file, 'w') as f:
            for src, tgt, count in us:
                f.write(src.strip() + ' --- ' + tgt.strip() + ' ' + str(count) + '\n')
    return us


# Get word order shifts
def word_order_shifts(alignment_file, al_count_dict=None, min_count_word=10, min_count_shift=10, tgt_file=None, return_sents=False):
    alignment = open(alignment_file, 'r', encoding='utf8').readlines()
    alignment = [alignment[i:i+3] for i in range(0, len(alignment), 3)]
    wo_shift_count = defaultdict(lambda:Counter())
    wo_shift_sents = defaultdict(lambda:defaultdict(list)) if return_sents else None
    
    for (_, tgt, src) in tqdm(alignment, desc='Counting word-order shifts'):
        tgt_indices = {i+1:w for (i,w) in enumerate(tgt.strip().split())}
        src = [x.strip() for x in src.split('})')][:-1]
        src = [x.split('({ ') for x in src]
        src_orig = ' '.join([l[0].replace('({', '').strip() for l in src[1:]])
        src = [x for x in src if len(x)==2]
        src = [(w.strip(), [int(i) for i in n.split() if i.isnumeric()]) for (w,n) in src]
        src = [(i, w, tgt_ind) for i,(w, tgt_ind) in enumerate(src)]
        src_pairs = [(l1[1:], l2[1:]) for (l1,l2) in list(product(src, repeat=2)) if l1[0]<l2[0]]
        wo_shifts = [(l1, l2) for (l1, l2) in src_pairs if max(l1[1]) > min(l2[1])]
        wo_shifts = [(l1, l2) for (l1, l2) in wo_shifts if l1[0] != 'NULL' and
                     [c for c in l1[0] if c.isalnum()] and [c for c in l2[0] if c.isalnum()]]
        
        for l1, l2 in wo_shifts:           
            src_1, src_2 = l1[0], l2[0]
            count_enough = True
            if al_count_dict:
                tgt_1 = ' '.join([tgt_indices[i] for i in l1[1]]).strip()
                tgt_2 = ' '.join([tgt_indices[i] for i in l2[1]]).strip()
                if min(al_count_dict[src_1][tgt_1], al_count_dict[src_2][tgt_2]) < min_count_word:
                    count_enough = False
            
            if not count_enough:
                continue
            
            src_str = ' '.join([l1[0], l2[0]]).strip()
            tgt_str = ' '.join([tgt_indices[i] for i in sorted(l1[1] + l2[1])]).strip()
            
            wo_shift_count[src_str][tgt_str] += 1
            if return_sents:
                wo_shift_sents[src_str][tgt_str].append((src_orig, tgt.strip()))
            
    wo_shift_ls = []
    for src in wo_shift_count:
        for tgt in [t for t in wo_shift_count[src] if wo_shift_count[src][t]>=min_count_shift]:
            wo_shift_ls.append((src, tgt, wo_shift_count[src][tgt]))
    
    if tgt_file:
        with open(tgt_file, 'w') as f:
            for src, tgt, count in wo_shift_ls:
                f.write(src.strip() + ' --- ' + tgt.strip() + ' ' + str(count) + '\n')
    
    if return_sents:
        return wo_shift_ls, wo_shift_sents
    return wo_shift_ls
        

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--src', help='Source alignment file created by GIZA++', default='../tatoeba/en_fi-fi_en/fi_en.A3.final')
    arg_parser.add_argument('--tgt', help='Target filepath', default='results/shifts.txt')
    arg_parser.add_argument('--type', help='Type of translation shift: word_order/unit', default='unit')
    arg_parser.add_argument('--min_count', type=int, help='Minimum target count per source word', default=10)

    args = arg_parser.parse_args()
    
    if args.tgt and not os.path.exists(os.path.dirname(args.tgt)):
        os.makedirs(os.path.dirname(args.tgt))
    
    al_counts = alignment_counts(alignment_file=args.src)
    
    shift = 'word-order' if args.type in ['word_order', 'word-order'] else 'unit'
    
    if shift == 'word-order':
        shifts = word_order_shifts(alignment_file=args.src,
                                   al_count_dict=al_counts,
                                   min_count_word=args.min_count,
                                   min_count_shift=args.min_count,
                                   tgt_file=args.tgt)
    else:
        shifts = unit_shifts(alignments=al_counts,
                             min_count=args.min_count,
                             tgt_file=args.tgt)
    
    print('\nOutput written to:', args.tgt)

if __name__=='__main__':
    main()