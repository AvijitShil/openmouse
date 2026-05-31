import os
import json
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from typing import Optional

class AgentAction(BaseModel):
    action: str = Field(description="Must be exactly 'click', 'type', or 'wait'")
    target: Optional[str] = Field(description="The numeric ID string found inside the bounding box tag, or null if waiting")
    text: Optional[str] = Field(description="The string text value to type, leave blank if clicking")
    reasoning: str = Field(description="A brief sentence explaining why this step is taken")

class BrainController:
    def __init__(self):
        self.client = AsyncOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_API_KEY")
        )
        self.model = os.getenv("NVIDIA_NIM_MODEL", "meta/llama3-70b-instruct")

    async def request_next_step(self, b64_image: str, objective: str) -> dict:
        """Queries the NIM VLM with structured json schemas for speed and output reliability."""
        prompt = (
            f"Objective: {objective}\n\n"
            "Analyze the screenshot containing numerical tags. Identify the target item ID to step closer to the objective. "
            "Respond using the strict JSON schema provided."
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )

            raw_content = response.choices[0].message.content
            return json.loads(raw_content)
        except Exception as e:
            print(f"[Brain Error] Failed parsing response or calling NIM API: {e}")
            return {"action": "wait", "target": None, "reasoning": "Fallback due to API parsing issues."}