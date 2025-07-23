from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from typing import Any, Tuple
from skyrl_gym.envs.allocated_code import utils
import re
from typing import Dict, Optional, List
from omegaconf import DictConfig
import httpx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AllocatedCodeToolGroup:
    def __init__(self, manager_url: str = "http://localhost:5000"):
        self.manager_url = manager_url
        self.allocated_container: Optional[int] = None
        self.client = httpx.Client(timeout=30.0)
    
    def allocate_container(self):
        if self.allocated_container is not None:
            logger.info(f"Container already allocated: {self.allocated_container}")
            return
        
        logger.info(f"Allocating container from {self.manager_url}/allocate")
        response = self.client.post(f"{self.manager_url}/allocate")
        response.raise_for_status()
        result = response.json()
        self.allocated_container = result["container_id"]
        logger.info(f"Allocated container: {self.allocated_container}")
    
    def deallocate_container(self):
        if self.allocated_container is None:
            logger.info("No container to deallocate")
            return
        
        logger.info(f"Deallocating container: {self.allocated_container}")
        self.client.post(f"{self.manager_url}/deallocate/{self.allocated_container}")
        self.allocated_container = None
        logger.info("Container deallocated")
    
    def execute_code(self, code: str) -> str:
        if self.allocated_container is None:
            logger.error("No container allocated for code execution")
            raise RuntimeError("No container allocated")
        
        logger.info(f"Executing code on container {self.allocated_container}: {code[:100]}...")
        response = self.client.post(
            f"{self.manager_url}/session/{self.allocated_container}/execute",
            json={"code": code}
        )
        response.raise_for_status()
        result = response.json()
        output = str(result.get("outputs", []))
        logger.info(f"Code execution result: {output}")
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
        
        # Allocate container immediately since training framework may not call init/reset
        logger.info("=== ENVIRONMENT CONSTRUCTOR ===")
        self.tool_group.allocate_container()
        logger.info("Container allocated in constructor")
        
        self.chat_history: ConversationType = []

    def reset(self, **kwargs):
        logger.info("=== ENVIRONMENT RESET ===")
        self.turns = 0
        self.chat_history = []
        self.tool_group.allocate_container()
        logger.info("Environment reset complete")
        return []  # Return empty observations

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        logger.info("=== ENVIRONMENT INIT ===")
        self.turns = 0
        self.chat_history = []
        self.tool_group.allocate_container()
        logger.info("Environment init complete")
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
        logger.info(f"=== ENVIRONMENT STEP {self.turns + 1} ===")
        logger.info(f"Action: {action[:200]}...")
        
        self.turns += 1
        action = self._postprocess_action(action)
        self.chat_history.append({"role": "assistant", "content": action})

        error = None
        done = self._is_done(action)
        reward = self._get_reward(action, done)
        
        logger.info(f"Done: {done}, Reward: {reward}")

        if done:
            logger.info("Episode finished, deallocating container")
            self.tool_group.deallocate_container()
            return BaseTextEnvStepOutput(
                observations=[], reward=reward, done=done, metadata={}, postprocessed_action=action
            )

        try:
            code = self._parse_action(action)
            logger.info(f"Parsed code: {code}")
            if code:
                logger.info("Executing code...")
                tool_output = self.tool_group.execute_code(code)
                observation = "\n<information>" + tool_output + "</information>\n"
                logger.info(f"Code executed successfully, observation: {observation[:100]}...")
            else:
                logger.info("No code found in action")
                observation = None
        except Exception as e:
            logger.error(f"Code execution error: {str(e)}")
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
        logger.info("=== ENVIRONMENT CLOSE ===")
        self.tool_group.deallocate_container()
        super().close()
        logger.info("Environment closed") 