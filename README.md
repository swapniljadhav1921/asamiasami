# AsamiAsami

### Website
https://asamiasami.in/

### Demo APIs
https://rapidapi.com/user/asamiasami2020

### Github Main Repo
https://github.com/swapniljadhav1921/asamiasami

Google's Multilingual BERT Indian language's content percentage is <10%. Similarly, for GPT-3 which is the latest in the bunch has <7% content in other than English language. Over the years through experiments we observed that more the data & accurate the data, better the model ... irrespective of how big the model is. Original attention model by Vaswani with more data & hyper-parameter tuning held up very well against state-of-the-art models like BERT, GPT-2. minIndicBERT is the results of the same experimentation.


## Requirements
* Python >=3.6
* torch >=1.4
* sentencepiece >=0.1.83
* fairseq (https://github.com/pytorch/fairseq#requirements-and-installation)
* Flask >=1.0

## indicTranslation
* API Location => https://github.com/swapniljadhav1921/asamiasami/tree/main/indicTranslation
* Hindi-2-English Translation Model
    * Model Location => https://github.com/swapniljadhav1921/asamiasami/tree/main/indicTranslation/hi_en_t2t_v3
    * Try out direct api here (same code as shared) => https://rapidapi.com/asamiasami2020/api/indictranslator-hindi-2-english/details
    * Start your local api & Use
        * `python app.py 0.0.0.0 8888 ./hi_en_t2t_v3/token_data/ ./hi_en_t2t_v3/checkpoints/checkpoint8.pt ./sentencepiece.bpe.model hi &`
        * POST api url => `http://0.0.0.0:8888/get_translation`
        * raw/json body => 
        ```
           {
	             "text": "अपने करियर के 17 लिस्ट ए के मैचों में तारकाई ने 32.52 की औसत से 553 रन बनाए थे. इसके अलावा उन्होंने 33 टी-20 में उन्होंने 127.50 की स्ट्राइक रेट से 700 रन बनाए हैं. अफगानिस्तान के  
                दिग्गज स्पिनर राशिद खान ने भी ट्वीट कर अपना रिएक्शन दिया है. अफगानिस्तान क्रिकेट के लिए तारकाई उभरते हुए क्रिकेटर थे, टी-20 में तारकाई ने 4 अर्धशतक भी जमाए थे. तारकाई ने 2014 में इंटरनेशनल 
                क्रिकेट में डेब्यू किया था, अपने करियर में तराकाई ने 24 फर्स्ट क्लास मैच खेले और 2030 रन बनाए, फर्स्ट क्लास क्रिकेट में उनके नाम 6 शतक और 10 अर्धशतक शामिल है."
           }
        ```
        * Output =>
        ```
           {
                "outtext": "tarakai scored 553 runs in 17 list a matches of his career at an average of 32.52. apart from this, he has scored 700 runs in 33 t20is at a strike rate of 
                127.50. afghanistan's legendary spinner rashid khan has also tweeted his reaction. tarakai was a rising cricketer for afghanistan cricket, tarakai also scored 4 half-
                centuries in t20is. tarakai made his international debut in 2014, tarakai played 24 first class matches in his career and scored 2030 runs, first class cricket 
                includes 6 centuries and 10 half-centuries to his name."
           }
        ```
    * BLEU Score Benchmark on Tatoeba DataSet
        * indicTranslation : 49.07 [73.45, 55.02, 42.81, 33.51]
        * Google : 47.44 [73.16, 53.63, 40.88, 31.58]
        * HuggingFace : 40.04
* English-2-Hindi Translation Model
    * Model Location => https://github.com/swapniljadhav1921/asamiasami/tree/main/indicTranslation/en_hi_t2t_v3
    * Try out direct api here (same code as shared) => https://rapidapi.com/asamiasami2020/api/indictranslator-english-2-hindi/details
    * Start your local api & Use
        * `python app.py 0.0.0.0 8899 ./en_hi_t2t_v3/token_data/ ./en_hi_t2t_v3/checkpoints/checkpoint9.pt ./sentencepiece.bpe.model en &`
        * POST api url => `http://0.0.0.0:8899/get_translation`
        * raw/json body => 
        ```
           {
	             "text": "Google News is a news aggregator service developed by Google. It presents a continuous flow of articles organized from thousands of publishers and magazines. 
                Google News is available as an app on Android, iOS, and the Web. Google released a beta version in September 2002 and the official app in January 2006."
           }
        ```
        * Output =>
        ```
           {
                "outtext": "गूगल न्यूज गूगल द्वारा विकसित एक न्यूज एग्रीगेटर सर्विस है। यह हजारों प्रकाशकों और पत्रिकाओं से आयोजित लेखों का निरंतर प्रवाह प्रस्तुत करता है। गूगल न्यूज एंड्रॉयड, आईओएस, और वेब पर 
                ऐप के तौर पर उपलब्ध है। गूगल ने सितंबर 2002 में बीटा वर्जन और जनवरी 2006 में ऑफिशियल एप जारी किया था।",
           }
       ```
    * BLEU Score Benchmark on Tatoeba DataSet
        * indicTranslation : 28.82 [61.37, 38.43, 25.62, 17.71]
        * Google : 23.51 [51.52, 29.38, 18.07, 11.16]
        * HuggingFace : 16.1
* Next Steps
    * `app.py` has the most basic code written. You can extend and make it more usable with more options.
    * Command line model load & use


## minIndicBERT
* RoBERTa model & Sentence Tokenizer trained with just 4 encoders on 12+ Indian language data
* Input needs 512 tokens, sentence tokenizer has ~66k dictionary of tokens across 12+languages & transliterated text.
* Data Source - Scrapped Websites, Wikipedia, Opus http://opus.nlpl.eu/
* API Location => https://github.com/swapniljadhav1921/asamiasami/tree/main/minIndicBERT
* How to start API => `bash rerun.sh PORT_NUM`
* Live api can be tested here => https://rapidapi.com/asamiasami2020/api/indicbert/details
* Sample Code Location => https://github.com/swapniljadhav1921/asamiasami/blob/main/minIndicBERT/minIndicBERT_sample_code.py
* Model Traning Command (More details here => https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.pretraining.md )
```
CUDA_VISIBLE_DEVICES=0 fairseq-train --fp16 $DATA_DIR --task masked_lm --criterion masked_lm  --arch roberta_base --encoder-layers 4 --encoder-embed-dim 512 --encoder-ffn-embed-dim 1024 --encoder-attention-heads 8 --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 --skip-invalid-size-inputs-valid-test
```

### Process to Finetune 
* More Details Here => https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.pretraining.md
```
TOTAL_NUM_UPDATES=1000000
WARMUP_UPDATES=5000
LR=1e-05
HEAD_NAME=GIVE_SOME_UNIQ_NAME ### Later to be used in python code 
NUM_CLASSES=2
MAX_SENTENCES=64
ROBERTA_PATH=/minIndicBERT/model/path/*.pt

cd fairseq_installation_path

CUDA_VISIBLE_DEVICES=0 python train.py /path/bin_data/ --restore-file $ROBERTA_PATH --max-positions 512 --max-sentences $MAX_SENTENCES  --max-tokens 32768 --task sentence_prediction --reset-optimizer --reset-dataloader --reset-meters --required-batch-size-multiple 1 --init-token 0 --separator-token 2 --arch roberta_base --encoder-layers 4 --encoder-embed-dim 512 --encoder-ffn-embed-dim 1024 --encoder-attention-heads 8 --criterion sentence_prediction --classification-head-name $HEAD_NAME --num-classes $NUM_CLASSES --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 --clip-norm 0.0 --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 --max-epoch 16 --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric --find-unused-parameters  --update-freq 8 --skip-invalid-size-inputs-valid-test
```

## minIndicLanguageDetector
* RoBERTa model finetuned over minIndicBERT base model to detect language of a given text
* Input needs 512 tokens, sentence tokenizer has ~66k dictionary of tokens across 12+languages & transliterated text.
* Data Source - Scrapped Websites, Wikipedia, Opus http://opus.nlpl.eu/
* API Location => https://github.com/swapniljadhav1921/asamiasami/tree/main/minIndicLanguageDetector
* How to start API => `bash rerun.sh PORT_NUM`
* Live api can be tested here => https://rapidapi.com/asamiasami2020/api/indicbert-language-detection/details
* Sample Code Location => https://github.com/swapniljadhav1921/asamiasami/blob/main/minIndicLanguageDetector/minIndicBERT_Language_detection_sample_code.py


## minIndicNSFWDetector 
* RoBERTa model finetuned over minIndicBERT base model to detect if given text is safe or not-safe for work.
* Input needs 512 tokens, sentence tokenizer has ~66k dictionary of tokens across 12+languages & transliterated text.
* Data Source - Scrapped Websites, Wikipedia, Opus http://opus.nlpl.eu/
* API Location => https://github.com/swapniljadhav1921/asamiasami/tree/main/minIndicNSFWDetector
* How to start API => `bash rerun.sh PORT_NUM`
* Live api can be tested here => https://rapidapi.com/asamiasami2020/api/indicbert-nsfwdetection/details
* Sample Code Location => https://github.com/swapniljadhav1921/asamiasami/blob/main/minIndicNSFWDetector/minIndicBERT_NSFW_detection_sample_code.py



