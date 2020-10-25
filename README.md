# AsamiAsami

### Website
https://asamiasami.in/

### Demo APIs
https://rapidapi.com/user/asamiasami2020

### Github Main Repo
https://github.com/swapniljadhav1921/asamiasami

### Motivation
For many years we had  performance & accuracy issues with multi-lingual unsupervised models  ... especially for Indian Languages.   For example, in Google's Multilingual BERT Indian language's content percentage is <10%. Similarly, for GPT-3 which is the latest in the bunch has <7% content in other than English language. Over the years, we experienced through experiments that more the data & accurate the data, better the model ... irrespective of how big the model is. Original attention model by Vaswani with more data & hyper-parameter tuning held up very well against state-of-the-art models like BERT, GPT-2. minIndicBERT is the results of the same experimentation. Hope to introduce more & more APIs for Indian Languages in coming months. 



## indicTranslation
* API Location => https://github.com/swapniljadhav1921/asamiasami/tree/main/indicTranslation
* Hindi-2-English Translation Model
    * Model Location => https://github.com/swapniljadhav1921/asamiasami/tree/main/indicTranslation/hi_en_t2t_v3
    * Try out direct api here (same code as shared) => https://rapidapi.com/asamiasami2020/api/indictranslator-hindi-2-english/details
    * Start your local api => `python app.py 0.0.0.0 PORT_NUM ./hi_en_t2t_v3/token_data/ ./hi_en_t2t_v3/checkpoints/checkpoint8.pt ./sentencepiece.bpe.model hi &`
    * BLEU Score Benchmark on Tatoeba DataSet
        * indicTranslation : 49.07 [73.45, 55.02, 42.81, 33.51]
        * Google : 47.44 [73.16, 53.63, 40.88, 31.58]
        * HuggingFace : 40.04
* English-2-Hindi Translation Model
    * Model Location => https://github.com/swapniljadhav1921/asamiasami/tree/main/indicTranslation/en_hi_t2t_v3
    * Try out direct api here (same code as shared) => https://rapidapi.com/asamiasami2020/api/indictranslator-english-2-hindi/details
    * Start your local api => `python app.py 0.0.0.0 PORT_NUM ./en_hi_t2t_v3/token_data/ ./en_hi_t2t_v3/checkpoints/checkpoint9.pt ./sentencepiece.bpe.model en &`
    * BLEU Score Benchmark on Tatoeba DataSet
        * indicTranslation : 28.82 [61.37, 38.43, 25.62, 17.71]
        * Google : 23.51 [51.52, 29.38, 18.07, 11.16]
        * HuggingFace : 16.1
* Next Steps
    * `app.py` has the most basic code written. You can extend and make it more usable with more options.


## minIndicBERT
* API Location => https://github.com/swapniljadhav1921/asamiasami/tree/main/minIndicBERT
* Sample Code Location => https://github.com/swapniljadhav1921/asamiasami/blob/main/minIndicBERT/minIndicBERT_sample_code.py


## minIndicLanguageDetector


## minIndicNSFWDetector 

