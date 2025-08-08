from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/api/v1/hackrx/run", methods=["POST"])
def hackrx_webhook():
    data = request.json  # Get JSON payload
    print("Incoming request:", data)

    # TODO: Add your challenge solution logic here
    result = {"status": "success", "answer": "Webhook is alive!"}

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
