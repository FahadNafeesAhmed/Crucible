import os
from google.cloud import dialogflowcx_v3beta1 as dialogflow
from google.api_core.client_options import ClientOptions

class CloudReflectorClient:
    def __init__(self):
        self.project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "crucible")
        self.location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
        self.agent_id = os.environ.get("GOOGLE_CLOUD_AGENT_ID", "")
        
        # Determine endpoint based on location
        client_options = None
        if self.location != "global":
            client_options = ClientOptions(api_endpoint=f"{self.location}-dialogflow.googleapis.com")
            
        self.client = dialogflow.SessionsClient(client_options=client_options)
        self.session_id = "crucible-eval-loop"
        self.session_path = self.client.session_path(
            self.project_id, self.location, self.agent_id, self.session_id
        )

    def generate_new_rule(self, failed_reviews: list) -> str:
        """
        Calls the Google Cloud Agent Builder (Vertex AI Agents) to analyze failures.
        The Cloud Agent will autonomously use its OpenAPI MCP Tool to fetch the traces
        and return the newly generated rule.
        """
        if not self.agent_id:
            print("[Warning] GOOGLE_CLOUD_AGENT_ID not set. Falling back to default rule.")
            return "Rule: Look for generic statements."

        text_input = dialogflow.TextInput(text="Analyze the latest failures and generate a new rule.")
        query_input = dialogflow.QueryInput(text=text_input, language_code="en")

        request = dialogflow.DetectIntentRequest(
            session=self.session_path,
            query_input=query_input,
        )

        try:
            response = self.client.detect_intent(request=request)
            # The agent's generated rule is typically in the first text response
            for message in response.query_result.response_messages:
                if message.text:
                    return message.text.text[0]
            
            return "Rule: Agent returned no response."
        except Exception as e:
            print(f"[Cloud Reflector] Error calling Agent Builder: {e}")
            return "Rule: Error calling Cloud Agent."
