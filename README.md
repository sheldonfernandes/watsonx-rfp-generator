# watsonx-rfp-generator
A chat assistant to help you understand RFPs and build a response tailor to your organizations capabilities

## Installation

1. Create a .env file containing the following keys

```
WATSONX_AI_KEY=<Watsonx.ai key>
WATSONX_AI_API=https://us-south.ml.cloud.ibm.com
WATSONX_AI_PROJECT_ID=<Watsonx.ai Project ID>
```

2. Create virtual environemt by running following commands
```
python -m venv watsonx-rfp-generator
source watsonx-rfp-generator/bin/activate
```

3. Install dependencies

```
pip install -r requirements.txt
```

4. Start the API server
```
chainlit run app.py -w
```