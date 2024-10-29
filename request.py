import os
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

# Nome do arquivo JSON onde as informações serão salvas
JSON_FILE = 'rosto_imagem.json'

# Cria o arquivo JSON se não existir
if not os.path.exists(JSON_FILE):
    with open(JSON_FILE, 'w') as json_file:
        json.dump([], json_file)  # Inicializa com uma lista vazia

@app.route('/upload', methods=['POST'])
def upload():
    # Verifica se os dados foram enviados como JSON
    if not request.is_json:
        return jsonify({"error": "Expected JSON data"}), 400

    # Recebe o JSON enviado na requisição
    data = request.get_json()

    # Salva o JSON no arquivo especificado
    try:
        # Lê os dados existentes do arquivo JSON
        with open(JSON_FILE, 'r') as json_file:
            existing_data = json.load(json_file)

        # Adiciona os novos dados à lista existente

        new_entry = {
            "time": data["time"],
            "base64_image": data["base64_image"]
        }

        existing_data.append(new_entry)  

        # Salva a lista atualizada no arquivo JSON
        with open(JSON_FILE, 'w') as json_file:
            json.dump(existing_data, json_file, indent=4)
    except IOError as e:
        return jsonify({"error": f"Failed to save JSON: {e}"}), 500

    return jsonify({"status": "success"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)