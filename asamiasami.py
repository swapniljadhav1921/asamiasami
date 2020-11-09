import os
from fairseq.models.roberta import RobertaModel
import numpy as np
import sentencepiece as spm
import gdown
from utils import get_translation, Generator


class minTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def getTokens(self, text):
        return self.sp.encode_as_pieces(text.strip().lower())


class minIndicBERT:
    def __init__(self, model_folder="./minIndicBERT", model_name="minIndicBERT.pt", run_option="gpu"):
        self.model_path = "%s/%s" % (model_folder, model_name)
        self.run_option = run_option
        if not os.path.exists(self.model_path):
            url = "https://drive.google.com/uc?id=1h65he9aH1Oeiz0JqF2Zvtwo5Iv2_3ozb"
            gdown.download(url, self.model_path, quiet=False)
        self.model = RobertaModel.from_pretrained(model_folder, checkpoint_file=model_name, data_name_or_path="./token_data", bpe="sentencepiece")
        self.model.eval()
        if run_option == "gpu":
            self.model.cuda()

    def getVector(self, text):
        tokens = self.model.encode(text)
        if self.run_option == "gpu":
            last_layer_features = self.model.extract_features(tokens[:511]).detach().cpu().numpy()
        else:
            last_layer_features = self.model.extract_features(tokens[:511]).detach().numpy()
        avg_feature = np.mean(last_layer_features[0], axis=0)
        avg_feature = avg_feature / np.linalg.norm(avg_feature)
        return avg_feature

class minIndicLanguageDetector:
    def __init__(self, model_folder="./minIndicLanguageDetector", model_name="minIndicLanguageDetection.pt", run_option="gpu"):
        self.model_path = "%s/%s" % (model_folder, model_name)
        self.run_option = run_option
        if not os.path.exists(self.model_path):
            url = "https://drive.google.com/uc?id=1YIoOfSrOWIsCxxJe5ko7M3Ys0gn8N43K"
            gdown.download(url, self.model_path, quiet=False)
        self.model = RobertaModel.from_pretrained(model_folder, checkpoint_file=model_name, data_name_or_path="./bin_data", bpe="sentencepiece")
        self.model.eval()
        if run_option == "gpu":
            self.model.cuda()
        self.label_fn = lambda label: self.model.task.label_dictionary.string([label + self.model.task.label_dictionary.nspecial])
        self.lang_arr = ['english', 'gujarati', 'nepali', 'malayalam', 'kannada', 'marathi', 'hindi', 'bangla', 'tamil', 'telugu', 'punjabi', 'urdu', 'oriya']

    def getLanguage(self, text):
        tokens = self.model.encode(text)
        ### lang_detect is HEAD_NAME used in fine-tuning process
        if self.run_option == "gpu":
            pred_arr = self.model.predict('lang_detect', tokens[:511]).detach().cpu().numpy().tolist()[0]
        else:
            pred_arr = self.model.predict('lang_detect', tokens[:511]).detach().numpy().tolist()[0]
        pred_min = np.min(pred_arr)
        if pred_min < 0:
            pred_min = pred_min * -1.0
            pred_arr = [x+pred_min for x in pred_arr]
        pred_sum = np.sum(pred_arr)
        pred_arr = [float("{:.4f}".format(x*1.0/pred_sum)) for x in pred_arr]
        possible_classes = dict(zip(self.lang_arr, pred_arr))
        return possible_classes


class minIndicNSFWDetector:
    def __init__(self, model_folder="./minIndicNSFWDetector", model_name="minIndicTextNSFW.pt", run_option="gpu"):
        self.model_path = "%s/%s" % (model_folder, model_name)
        self.run_option = run_option
        if not os.path.exists(self.model_path):
            url = "https://drive.google.com/uc?id=1uukQSEIKOdMQ2ydc_nerygxN5mmDyi5Y"
            gdown.download(url, self.model_path, quiet=False)
        self.model = RobertaModel.from_pretrained(model_folder, checkpoint_file=model_name, data_name_or_path="./bin_data", bpe="sentencepiece")
        self.model.eval()
        if run_option == "gpu":
            self.model.cuda()
        self.label_fn = lambda label: self.model.task.label_dictionary.string([label + self.model.task.label_dictionary.nspecial])
        self.nsfw_arr = ['sfw', 'nsfw']

    def getNSFWClass(self, text):
        tokens = self.model.encode(text)
        ### nsfw_detect is HEAD_NAME used in fine-tuning process
        if self.run_option == "gpu":
            pred_arr = self.model.predict('nsfw_detect', tokens[:511]).detach().cpu().numpy().tolist()[0]
        else:
            pred_arr = self.model.predict('nsfw_detect', tokens[:511]).detach().numpy().tolist()[0]
        pred_min = np.min(pred_arr)
        if pred_min < 0:
            pred_min = pred_min * -1.0
            pred_arr = [x+pred_min for x in pred_arr]
        pred_sum = np.sum(pred_arr)
        pred_arr = [float("{:.4f}".format(x*1.0/pred_sum)) for x in pred_arr]
        possible_classes = dict(zip(self.nsfw_arr, pred_arr))
        return possible_classes


class En2HiTranslator:
    def __init__(self, data_token_path="./indicTranslation/en_hi_t2t_v3/token_data/", model_path="./indicTranslation/en_hi_t2t_v3/en2hiTranslation.pt", tokenizer_path="./indicTranslation/sentencepiece.bpe.model"):
        if not os.path.exists(model_path):
            url = "https://drive.google.com/uc?id=1uWmlwYxISz5CB33BQQim1OulKbKHqj-W"
            gdown.download(url, model_path, quiet=False)
        self.gen = Generator(data_token_path, model_path)
        self.tokObj = minTokenizer(tokenizer_path)

    def getTranslation(self, text):
        return get_translation(self.gen, self.tokObj.sp, text, "en")


class Hi2EnTranslator:
    def __init__(self, data_token_path="./indicTranslation/hi_en_t2t_v3/token_data/", model_path="./indicTranslation/hi_en_t2t_v3/hi2enTranslation.pt", tokenizer_path="./indicTranslation/sentencepiece.bpe.model"):
        if not os.path.exists(model_path):
            url = "https://drive.google.com/uc?id=1rrySQx5FJ-IxCPiLxJFYU5_IfwAyeEGI"
            gdown.download(url, model_path, quiet=False)
        self.gen = Generator(data_token_path, model_path)
        self.tokObj = minTokenizer(tokenizer_path)

    def getTranslation(self, text):
        return get_translation(self.gen, self.tokObj.sp, text, "hi")
