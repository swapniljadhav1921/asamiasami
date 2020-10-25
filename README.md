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
* API Location => https://github.com/swapniljadhav1921/asamiasami/tree/main/minIndicBERT
* Sample Code Location => https://github.com/swapniljadhav1921/asamiasami/blob/main/minIndicBERT/minIndicBERT_sample_code.py


## minIndicLanguageDetector


## minIndicNSFWDetector 

