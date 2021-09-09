# Toxic Spans Detection (SemEval 2021 Task 5)

The Toxic Spans Detection task concerns the evaluation of systems that detect the spans that make a text toxic, when detecting such spans is possible. Moderation is crucial to promoting healthy online discussions. Although several toxicity (a.k.a. abusive language) detection datasets (Wulczyn et al., 2017; Borkan et al., 2019) and models (Schmidt and Wiegand, 2017; Pavlopoulos et al., 2017b; Zampieri et al., 2019) have been released, most of them classify whole comments or documents, and do not identify the spans that make a text toxic. But highlighting such toxic spans can assist human moderators (e.g., news portals moderators) who often deal with lengthy comments, and who prefer attribution instead of just a system-generated unexplained toxicity score per post. The evaluation of systems that could accurately locate toxic spans within a text is thus a crucial step towards successful semi-automated moderation.

We received 479 individual participation requests, 92 team formations, and 1,449 submissions. 91 teams submitted valid predictions (1,385 valid submissions in total) and were scored; out of these, only 36 submitted system descriptions. The best performing system achieved 70.83% F1. Please read our [task overview paper](https://aclanthology.org/2021.semeval-1.6.pdf) for more information about the task, data, evaluation, and performance of the participating systems. If you want to cite this work, please use the following:
```
@inproceedings{pavlopoulos-etal-2021-semeval,
    title = "{S}em{E}val-2021 Task 5: Toxic Spans Detection",
    author = "Pavlopoulos, John  and Sorensen, Jeffrey  and Laugier, L{\'e}o and Androutsopoulos, Ion",
    booktitle = "Proceedings of the 15th International Workshop on Semantic Evaluation (SemEval-2021)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.semeval-1.6",
    doi = "10.18653/v1/2021.semeval-1.6",
    pages = "59--69",
}
```
* In this repository you will find a notebook with code to prepare a valid submission.
* Evaluation code and baseline methods are included.
* The trial, train and test data that were used in the 2021 SemEval challenge are also included.
