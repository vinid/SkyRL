"""
Test environment for DiscoveryBench evaluation that saves predictions and ground truth.
This is a modified version of the allocated_code environment that saves hypothesis data.
"""

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from typing import Any, Tuple, Dict, Optional, List
from skyrl_gym.envs.allocated_code import utils
import re
import json
import os
from omegaconf import DictConfig
import httpx


class AllocatedCodeTestEnv(BaseTextEnv):
    """
    Test environment that saves predicted hypotheses and ground truth for later evaluation.
    """
    
    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        
        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.max_turns = extras["max_turns"] if "max_turns" in extras else 10
        self.query = extras["extra_info"]["question"]
        
        # Store metadata for later evaluation
        self.extra_info = extras.get("extra_info", {})
        self.dataset_name = self.extra_info.get("source", "unknown")
        self.metadata_id = self.extra_info.get("metadata_id", 0)
        self.query_id = self.extra_info.get("query_id", 0)
        self.example_index = self.extra_info.get("index", 0)
        
        # Setup output directory for saving predictions
        # Try multiple possible config paths for test_output_dir
        self.output_dir = (
            env_config.get("test_output_dir") or 
            os.environ.get("SKYRL_TEST_OUTPUT_DIR") or
            "/data/skyrl_test_predictions"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"🔧 [CONFIG] Using output directory: {self.output_dir}")
        print(f"🔧 [CONFIG] Available env_config keys: {list(env_config.keys()) if env_config else 'None'}")
        
        manager_url = env_config.get("manager_url", "http://localhost:5000")
        self.tool_group = AllocatedCodeToolGroup(manager_url)
        
        self.tool_group.allocate_container()
        
        self.chat_history: ConversationType = []

    def reset(self, **kwargs):
        self.turns = 0
        self.chat_history = []
        self.tool_group.allocate_container()
        print(f"🔄 [RESET] Episode {self.example_index}: {self.dataset_name}_{self.metadata_id}_{self.query_id}")
        print(f"📝 [QUERY] {self.query}")
        print(f"🎯 [GROUND_TRUTH] {self.ground_truth}")
        print(f"📁 [OUTPUT_DIR] {self.output_dir}")
        return []

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        self.turns = 0
        self.chat_history = []
        self.tool_group.allocate_container()
        return prompt, {}

    def _parse_action(self, action: str) -> Optional[str]:
        match = None
        if "<python>" in action and "</python>" in action:
            match = re.search(r"<python>(.*?)</python>", action, re.DOTALL)
            if match:
                code = match.group(1)
                if "```python" in code and "```" in code:
                    inner_match = re.search(r"```python(.*?)```", code, re.DOTALL)
                    if inner_match:
                        return inner_match.group(1)
                return code
        return None

    def _extract_hypothesis(self, response: str) -> str:
        """Extract hypothesis from model response."""
        # Look for <answer> tags first
        if '<answer>' in response and '</answer>' in response:
            match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return ''

    def _save_prediction(self, predicted_hypothesis: str, skyrl_reward: float):
        """Save prediction data for later DiscoveryBench evaluation."""
        prediction_data = {
            "example_index": self.example_index,
            "dataset_name": self.dataset_name,
            "metadata_id": self.metadata_id,
            "query_id": self.query_id,
            "query": self.query,
            "predicted_hypothesis": predicted_hypothesis,
            "ground_truth": self.ground_truth,
            "skyrl_reward": skyrl_reward,
            "full_response": "".join([item["content"] for item in self.chat_history]),
            "extra_info": self.extra_info
        }
        
        # Save to JSON file
        filename = f"prediction_{self.example_index}_{self.dataset_name}_{self.metadata_id}_{self.query_id}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        print(f"📝 [SAVE_DETAILS] File: {filename}")
        print(f"📂 [SAVE_PATH] {filepath}")
        print(f"📊 [DATA_SIZE] {len(str(prediction_data))} chars")
        
        try:
            with open(filepath, 'w') as f:
                json.dump(prediction_data, f, indent=2)
            print(f"✅ [SAVE_SUCCESS] Prediction saved successfully!")
        except Exception as e:
            print(f"❌ [SAVE_ERROR] Failed to save prediction: {e}")

    def _get_reward(self, action: str, done: bool) -> float:
        if done:
            chat_history_str = "".join([item["content"] for item in self.chat_history])
            hypothesis = self._extract_hypothesis(chat_history_str)
            # Calculate SkyRL reward (same as original)
            skyrl_reward = utils.compute_llm_score(hypothesis, chat_history_str, self.ground_truth, self.query)
            
            return skyrl_reward
        else:
            return 0

    def _is_done(self, action: str) -> bool:
        max_turns_reached = self.turns >= self.max_turns
        has_answer_tags = "<answer>" in action and "</answer>" in action
        
        print(f"🔍 [DONE_CHECK] Max turns: {max_turns_reached} ({self.turns}/{self.max_turns}), Has answer tags: {has_answer_tags}")
        
        if max_turns_reached:
            print(f"⏰ [MAX_TURNS] Episode ending due to max turns reached")
            return True
        if has_answer_tags:
            print(f"🎯 [ANSWER_FOUND] Episode ending due to answer tags found")
            return True
        return False

    def _postprocess_action(self, action: str) -> str:
        if "</python>" in action:
            return action.split("</python>")[0] + "</python>"
        elif "</answer>" in action:
            return action.split("</answer>")[0] + "</answer>"
        else:
            return action

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        action = self._postprocess_action(action)
        self.chat_history.append({"role": "assistant", "content": action})

        print(f"🚶 [STEP {self.turns}] Episode {self.example_index}")
        if len(action) > 400:
            print(f"🤖 [ACTION] {action[:200]}...{action[-200:]}")
        else:
            print(f"🤖 [ACTION] {action}")

        error = None
        done = self._is_done(action)
        reward = self._get_reward(action, done)
        
        print(f"✅ [DONE] {done}, 🏆 [REWARD] {reward}, 🔄 [TURNS] {self.turns}/{self.max_turns}")

        if done:
            # Extract and save prediction when episode is done
            predicted_hypothesis = self._extract_hypothesis(action)
            print(f"🎯 [PREDICTION] {predicted_hypothesis}")
            print(f"💾 [SAVING] Prediction to {self.output_dir}")
            self._save_prediction(predicted_hypothesis, reward)
            
            self.tool_group.deallocate_container()
            print(f"🏁 [EPISODE_END] Episode {self.example_index} completed in {self.turns} turns")
            return BaseTextEnvStepOutput(
                observations=[], reward=reward, done=done, metadata={}, postprocessed_action=action
            )

        try:
            code = self._parse_action(action)
            if code:
                tool_output = self.tool_group.execute_code(code)
                observation = "\n<information>" + tool_output + "</information>\n"
            else:
                observation = None
        except Exception as e:
            error = str(e)
            observation = None

        if observation:
            new_obs = {"role": "user", "content": observation}
        elif error:
            new_obs = {"role": "user", "content": error}
        else:
            new_obs = None

        info = {
            "tool_group": "AllocatedCodeToolGroup", 
            "tool_name": "python",
            "tool_input": code,
            "allocated_container": self.tool_group.allocated_container,
        }

        if new_obs:
            self.chat_history.append(new_obs)

        return BaseTextEnvStepOutput(
            observations=[new_obs] if new_obs else [],
            reward=reward,
            done=done,
            metadata=info,
            postprocessed_action=action
        )


class AllocatedCodeToolGroup:
    """Same as original AllocatedCodeToolGroup"""
    def __init__(self, manager_url: str = "http://localhost:5000"):
        self.manager_url = manager_url
        self.allocated_container: Optional[int] = None
        self.client = httpx.Client(timeout=30.0)
    
    def allocate_container(self):
        if self.allocated_container is not None:
            return
        
        response = self.client.post(f"{self.manager_url}/allocate")
        response.raise_for_status()
        result = response.json()
        self.allocated_container = result["container_id"]
    
    def deallocate_container(self):
        if self.allocated_container is None:
            return
        
        self.client.post(f"{self.manager_url}/deallocate/{self.allocated_container}")
        self.allocated_container = None
    
    def execute_code(self, code: str) -> str:
        if self.allocated_container is None:
            raise RuntimeError("No container allocated")
        
        response = self.client.post(
            f"{self.manager_url}/session/{self.allocated_container}/execute",
            json={"code": code}
        )
        response.raise_for_status()
        result = response.json()
        output = str(result.get("outputs", []))
        return output
    
    def get_tool_names(self):
        return ["python"]
