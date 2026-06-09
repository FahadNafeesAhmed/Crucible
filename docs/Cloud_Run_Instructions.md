# Google Cloud Agent Builder Setup

To completely finish the Hackathon requirement, you need to migrate the Reflector Agent out of our Python script and into **Google Cloud Vertex AI Agent Builder**. 

Follow these steps exactly:

## Step 1: Deploy the MCP Wrapper to Cloud Run
Because Google Cloud Agent Tools communicate via REST API (and cannot run a local Node.js process via stdio), we have wrapped our MCP connection into a tiny FastAPI script (`mcp_cloud_run_wrapper.py`).

1. Open your terminal in the `crucible` directory.
2. Ensure you have the `gcloud` CLI installed and authenticated.
3. Run this command to deploy it:
   ```bash
   gcloud run deploy mcp-wrapper \
     --source . \
     --set-env-vars PHOENIX_API_KEY=YOUR_KEY_HERE \
     --allow-unauthenticated
   ```
4. Copy the resulting **Service URL** (e.g. `https://mcp-wrapper-12345.a.run.app`).

## Step 2: Configure the Tool in Agent Builder
1. Go to the [Vertex AI Agent Builder Console](https://console.cloud.google.com/gen-app-builder).
2. Create a new App -> Select **Agent**.
3. Name it "Crucible-Reflector".
4. Go to the **Tools** tab and click **Create Tool**.
5. Select **OpenAPI**.
6. Open the `openapi.yaml` file in this project. Replace `https://[YOUR-CLOUD-RUN-URL]` with the URL you got from Step 1.
7. Paste the contents of `openapi.yaml` into the tool configuration and save it. Name the tool `Phoenix_MCP_Tool`.

## Step 3: Configure the Agent Persona
1. Go to the **Agents** tab.
2. Under **Instructions**, paste the exact Persona prompt we use in our Python script:
   ```text
   You are the Reflector. Your job is to analyze the traces of the Detector's recent failures and write a highly specific rule to prevent it from failing again.
   
   1. Use your Phoenix_MCP_Tool to fetch the latest traces.
   2. Analyze the 'input_value' (the review) and the 'output_value' (the wrong verdict).
   3. Identify the Forger's specific deception strategy.
   4. Output EXACTLY ONE new rule.
   ```
3. Under **Tools**, check the box next to `Phoenix_MCP_Tool` to give the agent access to it.
4. Save the Agent!

## Step 4: Update the Code
Now that the agent is living in Google Cloud:
1. Copy the **Agent ID** and **Project ID** from the console.
2. Add them to your `.env` file:
   ```env
   GOOGLE_CLOUD_PROJECT=your-project-id
   GOOGLE_CLOUD_AGENT_ID=your-agent-id
   ```
3. I will rewrite our `eval.py` loop to simply trigger your Cloud Agent instead of running the local Python logic.
