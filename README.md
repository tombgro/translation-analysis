# translation-analysis
Automatic analysis of translation shifts derived from parallel corpora aligned with the GIZA++ software.

## Preparation

Run alignment on a parallel corpus with the GIZA++ software:
https://www2.statmt.org/moses/giza/GIZA++.html

Original source for GIZA++:
Franz Josef Och, Hermann Ney. "A Systematic Comparison of Various Statistical Alignment Models", Computational Linguistics, volume 29, number 1, pp. 19-51 March 2003.

The final Vitebri alignment file (ending with "A3.final") is used as the argument for translation_shifts.py.

## Use

The file translation_shifts.py contains functions that take the alignment file as input and output information about the translations of source words.

- **translations(alignment_file):** list of translation pairs in the alignment data
- **alignment_counts(alignment_file):** dict from each source word to Counter of its translations
- **alignment_info(alignment_file OR output of alignment_counts):** statistics of translations
- **unit_shifts(alignment_file OR output of alignment_counts):** list of multi-word translations of a single source word (plus counts)
- **word_order_shifts(alignment_file):** list of translations between word-pairs with the linear order shifted (plus counts)

Use from terminal / command-line:
`python translation_shifts.py --src <path_to_alignment_file> --type <unit/word_order> --tgt <path_to_target_file>`
