# NLP Models For Indian Languages

Google's Multilingual BERT is trained on Indian language's content having contribution <10%. Similarly, for GPT-3 which is the latest in the bunch has <7% content in other than English language. Over the years through experiments we observed that more the data & accurate the data, better the model ... irrespective of how big the model is. Original attention model by Vaswani with more data & hyper-parameter tuning held up very well against state-of-the-art models like BERT, GPT-2. minIndicBERT is the results of the same experimentation and trained only on Indian Languages specifically.

### Languages Supported
'english', 'gujarati', 'nepali', 'malayalam', 'kannada', 'marathi', 'hindi', 'bangla', 'tamil', 'telugu', 'punjabi', 'urdu', 'oriya'



## Machine Instances Used
* aws T4 single gpu instance - 16gb gpu
* gtx 1070 - 8gb gpu
* Ubuntu 16.04, tested on cuda 10.0

## Data
* indicTranslation - Opus http://opus.nlpl.eu/ + augmented data
* minIndicBERT - Wikipedia dumps and free datasets found on github, reviews/comments web scrapped
* minIndicLanguageDetector - reviews/comments web scrapped, used transliteration to augment data
* minIndicNSFWDetector - free datasets available for slangs


## Installation

### Requirements
* Python >=3.6
* torch >=1.4
* sentencepiece >=0.1.83
* Flask >=1.0
* gdown
* nltk
```
import nltk
nltk.download('punkt')
```
* indic-nlp-library

### Install Fairseq
This particular commit of fairseq is the best compatible for this project. Later commits produce errors.
```
gdown https://drive.google.com/uc?id=19Dw1WMRKyDOBxzmvbU_Gc9WgdZuMVt_h
tar -xzvf fairseq.tar.gz
cd fairseq
pip install --editable ./
cd ..
```

### Install git LFS
```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
```

### Install AsamiAsami
```
git clone https://github.com/swapniljadhav1921/asamiasami.git
cd asamiasami
```

For more details please check `asasmiasami.py` which has simple code interface.
You can set `gpu` or `cpu` in class construction variable `run_option`. 

## indicTranslation
* Trained at sentence level. Process in sample api code => text -> split in sentences -> translation.
* English CASELESS text is used. It improves the model performance manyfold.
* Hindi-2-English Translation Model
    * Code Sample
    ```
    from asamiasami import Hi2EnTranslator
    hi2EnObj = Hi2EnTranslator()
    hi2EnObj.getTranslation("फर्स्ट क्लास क्रिकेट में उनके नाम 6 शतक और 10 अर्धशतक शामिल है")
    ```
    * BLEU Score Benchmark on Tatoeba DataSet
        * indicTranslation : ![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) `49.07 [73.45, 55.02, 42.81, 33.51]`
        * Google : 47.44 [73.16, 53.63, 40.88, 31.58]
        * HuggingFace : 40.04
    * Example
        ```
        Text => अपने करियर के 17 लिस्ट ए के मैचों में तारकाई ने 32.52 की औसत से 553 रन बनाए थे. इसके अलावा उन्होंने 33 टी-20 में उन्होंने 127.50 की स्ट्राइक रेट से 700 रन बनाए हैं. अफगानिस्तान के  
        दिग्गज स्पिनर राशिद खान ने भी ट्वीट कर अपना रिएक्शन दिया है. अफगानिस्तान क्रिकेट के लिए तारकाई उभरते हुए क्रिकेटर थे, टी-20 में तारकाई ने 4 अर्धशतक भी जमाए थे. तारकाई ने 2014 में इंटरनेशनल 
        क्रिकेट में डेब्यू किया था, अपने करियर में तराकाई ने 24 फर्स्ट क्लास मैच खेले और 2030 रन बनाए, फर्स्ट क्लास क्रिकेट में उनके नाम 6 शतक और 10 अर्धशतक शामिल है.
        
        Translation => tarakai scored 553 runs in 17 list a matches of his career at an average of 32.52. apart from this, he has scored 700 runs in 33 t20is at a strike rate of 
        127.50. afghanistan's legendary spinner rashid khan has also tweeted his reaction. tarakai was a rising cricketer for afghanistan cricket, tarakai also scored 4 half-
        centuries in t20is. tarakai made his international debut in 2014, tarakai played 24 first class matches in his career and scored 2030 runs, first class cricket 
        includes 6 centuries and 10 half-centuries to his name.
        ```
    
* English-2-Hindi Translation Model    
    * Code Sample
    ```
    from asamiasami import En2HiTranslator
    en2hiObj = En2HiTranslator()
    en2hiObj.getTranslation("Over the last three months, the spread of the pandemic has shifted from cities")
    ```
    * BLEU Score Benchmark on Tatoeba DataSet
        * indicTranslation : ![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+)`28.82 [61.37, 38.43, 25.62, 17.71]`
        * Google : 23.51 [51.52, 29.38, 18.07, 11.16]
        * HuggingFace : 16.1
    * Example
        ```
        Text => Over the last three months, the spread of the pandemic has shifted from cities towards rural areas, potentially threatening the agriculture sector which has been the 
        one bright spot in the economy so far. At the end of June, 80% of the districts with more than 1,000 confirmed cases of COVID-19 were urban, while only 20% were rural. By the 
        end of September, the ratio had morphed. Of districts with over 1,000 cases, 53% are now rural, according to data analysed by CRISIL Research.
        
        Translation => पिछले तीन महीनों में महामारी का प्रसार शहरों से ग्रामीण क्षेत्रों की ओर स्थानांतरित हो गया है, संभावित रूप से कृषि क्षेत्र को खतरा है जो अब तक अर्थव्यवस्था में एक उज्ज्वल स्थान रहा है। जून के अंत 
        में कोविद 19 के 1,000 से अधिक पुष्ट मामलों वाले 80 फीसदी जिले शहरी थे, जबकि सिर्फ 20 फीसदी ग्रामीण थे। सितंबर के अंत तक अनुपात मॉर्फ्ड हो गया था। 1,000 से अधिक मामलों वाले जिलों में से 53% अब ग्रामीण 
        हैं, जो कि क्रिस्पिल रिसर्च द्वारा विश्लेषण किए गए आंकड़ों के अनुसार हैं।
        ```


## minIndicBERT
* RoBERTa model & Sentence Tokenizer trained with just 4 encoders on 12+ Indian language data
* Input needs 512 tokens, sentence tokenizer has ~66k dictionary of tokens across 12+languages & transliterated text.
* Data Source - Scrapped Websites, Wikipedia, Opus http://opus.nlpl.eu/
* Code Sample For Vector Generation
    ```
    from asamiasami import minIndicBERT
    model = minIndicBERT(run_option="gpu")
    model.getVector("Some Text For Which Vector To Be Generated")
    ```
* Code Sample For Token Generation
    ```
    from asamiasami import minTokenizer
    model = minTokenizer("./indicTranslation/sentencepiece.bpe.model")
    model.getTokens("Sample text to get some tokens")
    ```

### Process to Finetune 
* More Details Here => https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.pretraining.md
* Input Raw data for these models is tokenised text. SentencePiece is used for the same. So in case you are following above link for data preparation ... convert train,test,valid text files to tokenised-text files. 
* More Info here => https://github.com/google/sentencepiece
* Command to encode file => `spm_encode --model=<model_file> --output_format=piece < input.txt > output.txt`
```
TOTAL_NUM_UPDATES=1000000
WARMUP_UPDATES=5000
LR=1e-05
HEAD_NAME=GIVE_SOME_UNIQ_NAME ### Later to be used in python code 
NUM_CLASSES=2
MAX_SENTENCES=64
ROBERTA_PATH=/minIndicBERT/model/path/*.pt

cd fairseq_installation_path

CUDA_VISIBLE_DEVICES=0 python train.py /path/bin_data/ --restore-file $ROBERTA_PATH --max-positions 512 --max-sentences $MAX_SENTENCES  --max-tokens 32768 --task sentence_prediction 
--reset-optimizer --reset-dataloader --reset-meters --required-batch-size-multiple 1 --init-token 0 --separator-token 2 --arch roberta_base --encoder-layers 4 --encoder-embed-dim 512 
--encoder-ffn-embed-dim 1024 --encoder-attention-heads 8 --criterion sentence_prediction --classification-head-name $HEAD_NAME --num-classes $NUM_CLASSES --dropout 0.1 --attention-
dropout 0.1 --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 --clip-norm 0.0 --lr-scheduler polynomial_decay --lr $LR --total-num-update 
$TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 --max-epoch 16 --best-checkpoint-metric accuracy 
--maximize-best-checkpoint-metric --find-unused-parameters  --update-freq 8 --skip-invalid-size-inputs-valid-test
```

## minIndicLanguageDetector
* RoBERTa model finetuned over minIndicBERT base model to detect language of a given text
* Input needs 512 tokens, sentence tokenizer has ~66k dictionary of tokens across 12+languages & transliterated text.
* Code Sample
    ```
    from asamiasami import minIndicLanguageDetector
    model = minIndicLanguageDetector(run_option="gpu")
    model.getLanguage("Sample Text For Which Language To Be Detected")
    ```

## minIndicNSFWDetector 
* RoBERTa model finetuned over minIndicBERT base model to detect if given text is safe or not-safe for work.
* Input needs 512 tokens, sentence tokenizer has ~66k dictionary of tokens across 12+languages & transliterated text.
* Code Sample
    ```
    from asamiasami import minIndicNSFWDetector
    model = minIndicNSFWDetector(run_option="gpu")
    model.getNSFWClass("Text for which safe unsafe to be detected")
    ```



