from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from typing import Any, Tuple
from skyrl_gym.envs.allocated_code import utils
import re
from typing import Dict, Optional, List
from omegaconf import DictConfig
import httpx


class AllocatedCodeToolGroup:
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


class AllocatedCodeEnv(BaseTextEnv):
    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.max_turns = extras["max_turns"] if "max_turns" in extras else 2

        manager_url = env_config.get("manager_url", "http://localhost:5000")
        self.tool_group = AllocatedCodeToolGroup(manager_url)
        
        self.tool_group.allocate_container()
        
        self.chat_history: ConversationType = []

    def reset(self, **kwargs):
        self.turns = 0
        self.chat_history = []
        self.tool_group.allocate_container()
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

    def _get_reward(self, action: str, done: bool) -> float:
        if done:
            chat_history_str = "".join([item["content"] for item in self.chat_history])
            return utils.compute_score(chat_history_str, self.ground_truth)
        else:
            return 0

    def _is_done(self, action: str) -> bool:
        if self.turns >= self.max_turns:
            return True
        return "<answer>" in action and "</answer>" in action

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

        error = None
        done = self._is_done(action)
        reward = self._get_reward(action, done)

        if done:
            self.tool_group.deallocate_container()
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
            postprocessed_action=action,
        )

    def close(self):
        self.tool_group.deallocate_container()
        super().close() 