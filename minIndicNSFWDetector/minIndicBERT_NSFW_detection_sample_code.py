from fairseq.models.roberta import RobertaModel
import numpy as np

roberta = RobertaModel.from_pretrained("/data/public_share/asamiasami/minIndicNSFWDetector", checkpoint_file="nsfw_text_model.pt", data_name_or_path="./bin_data", bpe="sentencepiece")
roberta.eval()
roberta.cuda()

label_fn = lambda label: roberta.task.label_dictionary.string([label + roberta.task.label_dictionary.nspecial])
cls_arr = ['sfw', 'nsfw']

text = """चिराग पासवान ने ट्वीट करके दावा किया कि चुनाव के बाद बिहार में नीतीश मुक्त सरकार बनेगी. उन्होंने कहा कि जनता नीतीश कुमार के राज से परेशान हो चुकी है. बीते 15 सालों में बिहार को बेरोजगारी, गरीबी, अशिक्षा के अलावा कुछ नहीं मिला."""
tokens = roberta.encode(text)
pred_arr = roberta.predict('nsfw_detect', tokens[:511]).detach().cpu().numpy().tolist()[0]
pred_min = np.min(pred_arr)
if pred_min < 0:
    pred_min = pred_min * -1.0
    pred_arr = [x+pred_min for x in pred_arr]
pred_sum = np.sum(pred_arr)
pred_arr = [float("{:.4f}".format(x*1.0/pred_sum)) for x in pred_arr]
possible_classes = dict(zip(cls_arr, pred_arr))
print("Class Scores => ", possible_classes)
