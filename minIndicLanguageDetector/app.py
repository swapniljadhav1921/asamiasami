from flask import Flask, render_template, request, jsonify
import sys, re, traceback, gc, json
from datetime import datetime
from fairseq.models.roberta import RobertaModel
import emoji
import numpy as np


### App Initialization
app = Flask(__name__)

model_path = sys.argv[3].strip()
checkpoint_file = sys.argv[4].strip()
data_name_or_path = sys.argv[5].strip()
bpe = sys.argv[6].strip()
roberta = RobertaModel.from_pretrained(model_path, checkpoint_file=checkpoint_file, data_name_or_path=data_name_or_path, bpe=bpe)
roberta.eval()
roberta.cuda()

label_fn = lambda label: roberta.task.label_dictionary.string([label + roberta.task.label_dictionary.nspecial])
lang_arr = ['english', 'gujarati', 'nepali', 'malayalam', 'kannada', 'marathi', 'hindi', 'bangla', 'tamil', 'telugu', 'punjabi', 'urdu', 'oriya']

def clean_text(text):
    text = emoji.demojize(str(text))
    text = text.lower().replace("  ", " ").replace("  ", " ").replace("  ", " ").strip()
    chstr = "[!@#$%^&*()[]{};:,./<>?\|`~-=_+]"
    for ch in chstr:
        text = text.replace(ch, " ")
    text = text.replace("  ", " ").replace("  ", " ").replace("  ", " ").strip()
    text = re.sub("\s+", " ", text)
    text = " ".join(text.split(' ')[:512])
    return text


###
### roberta language detection
###
@app.route('/get_language_scores', methods=['POST'])
def get_language_scores():
    start = datetime.now()
    print("INFO :: %s :: Request Started" % datetime.now())
    request_info = request.json
    out_json = {"api_output" : []}
    for tmap in [request_info]:
        responce_json = {}
        try:
            try:
                ogtext = tmap['text']
                ogid = tmap['id']
                text = clean_text(ogtext)
                responce_json = {"id":str(ogid)}
            except:
                traceback.print_exc()
                print("ERROR :: %s :: Error in input json" % datetime.now())
                ogtext = tmap.get('text', "")
                ogid = tmap.get('id', "")
                responce_json = {"id":str(ogid), "text":ogtext}
                responce_json["error_msg"] = "Error in input json id or text"
                out_json['api_output'].append(responce_json)
                continue
            print("INFO :: %s :: Json Received" % datetime.now())
            try:
                ### RoBERTa Language Classification Process ###
                tokens = roberta.encode(text)
                print("INFO :: %s :: RoBERTa encoding done" % datetime.now())
                pred_arr = roberta.predict('lang_detect', tokens[:511]).detach().cpu().numpy().tolist()[0] ### GPU
                print("INFO :: %s :: RoBERTa prediction done" % datetime.now())
                pred_min = np.min(pred_arr)
                if pred_min < 0:
                    pred_min = pred_min * -1.0
                    pred_arr = [x+pred_min for x in pred_arr]
                pred_sum = np.sum(pred_arr)
                pred_arr = [float("{:.4f}".format(x*1.0/pred_sum)) for x in pred_arr]
                possible_classes = dict(zip(lang_arr, pred_arr))
                responce_json['language_score_map'] = possible_classes
            except:
                traceback.print_exc()
                print("ERROR :: %s :: Error in RoBERTa Language Classification process" % datetime.now())
                responce_json["error_msg"] = "Error in RoBERTa Language Classification process"
                out_json['api_output'].append(responce_json)
                continue
        except Exception as e:
            print("ERROR :: %s :: %s - %s" % (datetime.now(), traceback.format_exc(), e))
            responce_json["error_msg"] = "%s : %s" % (traceback.format_exc(), e)
        out_json['api_output'].append(responce_json)
    end = datetime.now()
    print("INFO :: %s :: Processing Done" % datetime.now())
    out_json['api_time'] = "%s seconds" % (end-start).total_seconds()
    print("INFO :: %s :: API Time - %s" % (datetime.now(), (end-start).total_seconds()))
    return jsonify(out_json)

@app.route('/healthcheck', methods=['POST', 'GET'])
def healthcheck():
    gc.collect()
    return jsonify({"message":"success"})


if __name__ == '__main__':
    app_host = sys.argv[1].strip()
    app_port = sys.argv[2].strip()
    app.run(use_reloader = False, debug = True, host = app_host, port = app_port)
