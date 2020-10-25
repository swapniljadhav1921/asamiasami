port=$1

sudo kill -9 `sudo lsof -t -i:$port`

cd /home/ubuntu/demo_apis/indiebertNSFW/

nohup python -u app.py 0.0.0.0 $port /home/ubuntu/demo_apis/indiebertNSFW/ nsfw_text_model.pt ./bin_data sentencepiece NSFWFinal.txt > nohup.out &

echo "indiebert-nsfw-detection process started/restarted"
