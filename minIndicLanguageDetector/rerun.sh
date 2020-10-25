port=$1

sudo kill -9 `sudo lsof -t -i:$port`

cd /home/ubuntu/demo_apis/indiebertLanguage/

nohup python -u app.py 0.0.0.0 $port /home/ubuntu/demo_apis/indiebertLanguage/ lang_clf_model.pt ./bin_data sentencepiece > nohup.out &

echo "indiebert-language-detection process started/restarted"
