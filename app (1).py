from flask import Flask, jsonify, request
import openai
import os
import yaml
from utilis import processamento_text

# Set Keys aplication
with open('./config.yaml', 'r') as config_file:
  config = yaml.safe_load(config_file)
  
os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY'] 
os.environ['PINECONE_API_KEY'] = config['PINECONE_API_KEY'] 

app = Flask(__name__)

# completa
@app.route('/questions', methods=['GET'])
def generate_questions():
  data = request.get_json()
  
  transcribe = data.get('transcribe')
    
  if not transcribe or not isinstance(transcribe, str):
      return jsonify({'error': 'Invalid or missing transcribe data'}), 400
    
  
  chain = processamento_text('questions')
  
  response = chain.invoke(transcribe)
  
  if response.startswith('[Pergunta]:'):
    # retornar perguntas caso necessario 
    return jsonify({'as_questions': True, 'content': response})
  
  # criar folha de requisitos
  return jsonify({'as_questions': False, 'content': response})
    
# completa
@app.route('/requirement', methods=['GET'])
def get_requirement():
  data = request.get_json()
  
  transcribe = data.get('transcribe')
  
  if not transcribe or not isinstance(transcribe, str):
      return jsonify({'error': 'Invalid or missing transcribe data'}), 400
    
  
  chain = processamento_text('requirement')
  
  response = chain.invoke(transcribe)
  
  return jsonify({'content': response})

# completa
@app.route('/responses', methods=['POST'])
def get_responses():
  data = request.get_json()
  
  responses = data.get('transcribe')
  
  if not responses or not isinstance(responses, str):
      return jsonify({'error': 'Invalid or missing transcribe data'}), 400
    
  
  chain = processamento_text('responses')
  
  response = chain.invoke(responses)
  
  return jsonify({'content': response})

# completa
@app.route('/ata', methods=['GET'])
def create_ata():
  data = request.get_json()
  
  transcribe = data.get('transcribe')
    
  if not transcribe or not isinstance(transcribe, str):
      return jsonify({'error': 'Invalid or missing transcribe data'}), 400
  
  chain = processamento_text('ATA')
  
  response = chain.invoke(transcribe)
  
  return jsonify({'content': response})

#init api
app.run(port=8080, debug=True)
