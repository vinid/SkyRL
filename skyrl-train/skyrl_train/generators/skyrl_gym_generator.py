import asyncio
import copy
from uuid import uuid4
import skyrl_gym
from typing import List, Dict, Any, Optional
import numpy as np

from skyrl_train.generators.base import GeneratorInterface, GeneratorInput, GeneratorOutput
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.base import InferenceEngineInput, ConversationType
from omegaconf import DictConfig
from skyrl_gym.envs.base_text_env import BaseTextEnvStepOutput
from skyrl_train.generators.utils import get_custom_chat_template, get_generation_prompt_ids


class SkyRLGymGenerator(GeneratorInterface):
    def __init__(
        self,
        generator_cfg: DictConfig,
        skyrl_gym_cfg: DictConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
        model_name: str,
    ):
        """
        Args:
            generator_cfg: DictConfig object containing the generator configuration
            inference_engine_client: InferenceEngineClient object for interacting with the inference engines
            tokenizer: tokenizer object for encoding and decoding text
        """
        self.generator_cfg = generator_cfg
        self.skyrl_gym_cfg = skyrl_gym_cfg
        self.inference_engine_client = inference_engine_client
        self.tokenizer = tokenizer
        self.max_turns = generator_cfg.max_turns
        self.batched = generator_cfg.batched
        self.use_conversation_multi_turn = generator_cfg.use_conversation_multi_turn

        # optionally use custom chat template to get loss masks (i.e. for Qwen3)
        self.custom_chat_template = get_custom_chat_template(model_name)
        # get generation prompt ids for the tokenizer if needed
        self.generation_prompt_ids = get_generation_prompt_ids(tokenizer) if self.use_conversation_multi_turn else None

    def _update_engine_input_chat_history(
        self,
        chat_history: ConversationType,
        chat_end_index: int,
        loss_mask: List[int],
        input_ids: List[int],
        output: str,
        new_obs: ConversationType,
    ):
        """
        Update the chat history, loss mask, and input ids given a new model response and observation.

        This function is used if `use_conversation_multi_turn` is True. It assumes that the input to the LLM is formatted as a list of messages, with observations
        stored in user messages.

        For example (using the Qwen 2.5 chat template), a trajectory for multi-turn generation would look like:
        <|im_start|>system
        ...
        <|im_end|>
        <|im_start|>user
                            question goes here
        <|im_end|>
        <|im_start|>assistant
                            turn 1 model response goes here
                            <think>... </think>
                            ...
        <|im_end|>
        <|im_start|>user
                            turn 1 env observation goes here
                            <observation>...</observation>
        <|im_end|>
        ...

        the chat template is applied without tokenization before and after the chat history is appended to
        in order to get new token ids in the chat template format (but without re-tokenizing the entire chat history every turn)

        Args:
            chat_history: ConversationType
            chat_end_index: int
            loss_mask: List[int]
            input_ids: List[int]
            output: str
            new_obs: ConversationType
        Returns:
            chat_history: ConversationType
            chat_end_index: int
            loss_mask: List[int]
            input_ids: List[int]
        """
        # get number of output tokens - only this many tokens should be loss masked to 1
        num_output_tokens = len(self.tokenizer.encode(output, add_special_tokens=False))
        # remove eos token from end of output if it exists, since it will be reapplied by the chat template
        if output.endswith(self.tokenizer.eos_token):
            output = output[: -len(self.tokenizer.eos_token)]

        # Add assistant response to chat history
        chat_history += [{"role": "assistant", "content": output}]

        if self.custom_chat_template:
            chat_end_index += 1
            if len(new_obs) > 0:
                chat_history += new_obs
                chat_end_index += len(new_obs)
            # re-apply whole chat template so length check is correct
            input_ids = self.tokenizer.apply_chat_template(
                chat_history[:chat_end_index], add_generation_prompt=False, tokenize=True
            )
            return chat_history, chat_end_index, loss_mask, input_ids

        # apply chat template without tokenization
        prev = self.tokenizer.apply_chat_template(
            chat_history[:chat_end_index], add_generation_prompt=False, tokenize=False
        )
        curr = self.tokenizer.apply_chat_template(
            chat_history[: chat_end_index + 1], add_generation_prompt=False, tokenize=False
        )

        # entire response including chat template tokens
        new_resp_tokens = self.tokenizer.encode(curr[len(prev) :], add_special_tokens=False)

        # make sure that only the original output tokens are loss masked to 1
        new_loss_mask = [0] * len(self.generation_prompt_ids)  # 0 for generation prompt tokens
        new_loss_mask += [1] * num_output_tokens  # 1 for output tokens generated by model
        new_loss_mask += [0] * (
            len(new_resp_tokens) - len(self.generation_prompt_ids) - num_output_tokens
        )  # 0 for rest of response
        loss_mask += new_loss_mask

        input_ids += new_resp_tokens
        chat_end_index += 1

        # Add observations to chat history
        if len(new_obs) > 0:
            chat_history += new_obs

            # Directly encode the observation content
            for _ in range(len(new_obs)):
                prev = self.tokenizer.apply_chat_template(
                    chat_history[:chat_end_index], add_generation_prompt=False, tokenize=False
                )
                curr = self.tokenizer.apply_chat_template(
                    chat_history[: chat_end_index + 1], add_generation_prompt=False, tokenize=False
                )
                obs_tokens = self.tokenizer.encode(curr[len(prev) :], add_special_tokens=False)
                loss_mask += [0] * len(obs_tokens)
                input_ids += obs_tokens

                chat_end_index += 1

        return chat_history, chat_end_index, loss_mask, input_ids

    def _update_engine_input_token_ids(
        self, output: str, new_obs: ConversationType, loss_mask: List[int], input_ids: List[int]
    ):
        """
        Update the loss mask and input ids given a new model response and observation.

        This function is used if `use_conversation_multi_turn` is False. It assumes that the input to the LLM is a list of token ids
        and that the multi-turn conversation happens in a single assistant message.

        For example (using the Qwen 2.5 chat template), a trajectory for single-turn generation would look like:
        <|im_start|>system
        ...
        <|im_end|>
        <|im_start|>user
                            question goes here
        <|im_end|>
        <|im_start|>assistant
                            turn 1 model response goes here
                            <think>... </think>
                            ...

                            turn 1 env observation goes here
                            <observation>...</observation>

                            turn 2 model response goes here:
                            <think>... </think>
                            ...
        Args:
            output: str
            new_obs: ConversationType
            loss_mask: List[int]
            input_ids: List[int]
        Returns:
            loss_mask: List[int]
            input_ids: List[int]
        """
        # just update raw tokens and loss mask
        new_resp_tokens = self.tokenizer.encode(output, add_special_tokens=False)
        if new_resp_tokens[-1] == self.tokenizer.eos_token_id:
            # remove the eos token since we are continuing the current assistant message
            new_resp_tokens = new_resp_tokens[:-1]
        loss_mask += [1] * len(new_resp_tokens)
        input_ids += new_resp_tokens

        if len(new_obs) > 0:
            for obs in new_obs:
                obs_tokens = self.tokenizer.encode(obs["content"], add_special_tokens=False)
                loss_mask += [0] * len(obs_tokens)
                input_ids += obs_tokens

        return loss_mask, input_ids

    async def agent_loop(
        self,
        prompt: ConversationType,
        env_class: str,
        env_extras: List[Dict[str, Any]],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Multi-turn generation loop that executes a single trajectory.

        Args:
            prompt: ConversationType
            env_extras: List[Dict[str, Any]]
            max_tokens: int
            max_input_length: int
            sampling_params: Optional[Dict[str, Any]]
        Returns:
            response_ids: List[int]
            reward: float
            stop_reason: str
            loss_mask: List[int]
            prompt_token_ids: List[int]
        """

        # Create a new environment instance
        env_extras["max_turns"] = self.max_turns  # TODO(shu): move this to config
        env_config = self.skyrl_gym_cfg.get(env_class, DictConfig({}))
        env = skyrl_gym.make(env_class, env_config=env_config, extras=env_extras)

        trajectory_id = uuid4().hex
        done = False

        # need copy here since the prompt is a list of messages and we are going to modify it
        chat_history = copy.deepcopy(prompt)
        chat_end_index = len(chat_history)

        # Init() returns the first prompt to be given to the model, and optional metadata dict
        chat_history, _ = env.init(chat_history)
        input_ids = self.tokenizer.apply_chat_template(
            chat_history,
            # if we are keeping the chat history in token ids, we have to add the generation prompt to the original prompt
            add_generation_prompt=not self.use_conversation_multi_turn,
            tokenize=True,
        )

        initial_prompt_length = len(input_ids)
        loss_mask = []

        while not done:
            if self.use_conversation_multi_turn:
                engine_input = InferenceEngineInput(
                    prompts=[chat_history], trajectory_ids=[trajectory_id], sampling_params=sampling_params
                )
            else:
                engine_input = InferenceEngineInput(
                    prompt_token_ids=[input_ids], trajectory_ids=[trajectory_id], sampling_params=sampling_params
                )
            engine_output = await self.inference_engine_client.generate(engine_input)
            output = engine_output["responses"][0]
            stop_reason = engine_output["stop_reasons"][0]
            env_step_output: BaseTextEnvStepOutput = env.step(output)
            new_obs = env_step_output["observations"]
            reward = env_step_output["reward"]
            done = env_step_output["done"]

            if env_step_output.get("postprocessed_action", None) is not None:
                output = env_step_output["postprocessed_action"]

            if self.use_conversation_multi_turn:
                chat_history, chat_end_index, loss_mask, input_ids = self._update_engine_input_chat_history(
                    chat_history, chat_end_index, loss_mask, input_ids, output, new_obs
                )
            else:
                loss_mask, input_ids = self._update_engine_input_token_ids(output, new_obs, loss_mask, input_ids)

            if len(input_ids) > max_input_length:
                stop_reason = "length"
                break

        env.close()  # does nothing for now

        prompt_ids = input_ids[:initial_prompt_length]
        if self.custom_chat_template and self.use_conversation_multi_turn:
            response_encodings = self.tokenizer.apply_chat_template(
                chat_history[len(prompt) :],
                chat_template=self.custom_chat_template,
                add_generation_prompt=False,
                return_dict=True,
                return_assistant_tokens_mask=True,
                tokenize=True,
            )
            loss_mask = response_encodings["assistant_masks"]
            response_ids = response_encodings["input_ids"]
        else:
            response_ids = input_ids[initial_prompt_length:]

        if not self.use_conversation_multi_turn:
            # we might need to add the eos token to the response ids
            if response_ids[-1] != self.tokenizer.eos_token_id:
                response_ids.append(self.tokenizer.eos_token_id)
                loss_mask.append(1)

        # need to truncate loss mask correctly for responses that go to max length
        if self.max_turns > 1:
            # max total resp length = max tokens (max length of final turn generation) + max_input_length (max input for any generation turn) - len(original prompt)
            max_response_tokens = max_tokens + max_input_length - initial_prompt_length
        else:
            max_response_tokens = max_tokens

        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]
        if len(response_ids) > max_response_tokens:
            stop_reason = "length"

        return response_ids, reward, stop_reason, loss_mask, prompt_ids

    async def generate_batched(
        self,
        prompts: List[ConversationType],
        env_classes: List[str],
        env_extras: List[Dict[str, Any]],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Optional[Dict[str, Any]] = None,
    ) -> GeneratorOutput:
        """
        Single-turn batched generation (can use the synchronous offline engine)

        Args:
            prompts: List[ConversationType]
            env_classes: List[str]
            env_extras: List[Dict[str, Any]]
            max_tokens: int
            max_input_length: int --> Currently unused as we assume batched is used only for single-turn.
            sampling_params: Optional[Dict[str, Any]]
        Returns:
            GeneratorOutput
        """
        envs = []
        init_prompts = []
        for env_class, env_extra, prompt in zip(env_classes, env_extras, prompts):
            env_extra["max_turns"] = self.max_turns
            env_config = self.skyrl_gym_cfg.get(env_class, DictConfig({}))
            env = skyrl_gym.make(env_class, env_config=env_config, extras=env_extra)
            init_prompt, _ = env.init(prompt)
            init_prompts.append(init_prompt)
            envs.append(env)

        engine_input = InferenceEngineInput(prompts=init_prompts, sampling_params=sampling_params)
        engine_output = await self.inference_engine_client.generate(engine_input)
        responses = engine_output["responses"]
        stop_reasons = engine_output["stop_reasons"]
        truncated_responses = []
        rewards = []
        loss_masks = []

        for response, env in zip(responses, envs):
            # step on function and compute reward
            env_step_output: BaseTextEnvStepOutput = env.step(response)
            reward = env_step_output["reward"]
            rewards.append(reward)

            # if batched then always single turn
            response_ids = self.tokenizer(response)["input_ids"]
            if len(response_ids) > max_tokens:
                response_ids = response_ids[:max_tokens]
            loss_masks.append([1] * len(response_ids))
            truncated_responses.append(response_ids)

            env.close()

        prompt_token_ids = self.tokenizer.apply_chat_template(prompts, add_generation_prompt=True, tokenize=True)
        responses = truncated_responses
        rollout_metrics = self._rollout_metrics(responses, rewards)

        generator_output: GeneratorOutput = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
        }

        return generator_output

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """
        Generate trajectories for the input batch.

        Returns outputs in the same order as the input batch.
        Args:
            input_batch: GeneratorInput
        Returns:
            GeneratorOutput
        """
        prompts = input_batch["prompts"]
        env_classes = input_batch["env_classes"]
        env_extras = input_batch["env_extras"]
        sampling_params = input_batch.get("sampling_params", None)
        max_tokens = self.generator_cfg.sampling_params.max_generate_length
        max_input_length = self.generator_cfg.max_input_length

        if self.batched:
            return await self.generate_batched(
                prompts, env_classes, env_extras, max_tokens, max_input_length, sampling_params
            )

        # Async agent loop to generate trajectories in parallel.
        tasks = []
        for i in range(len(prompts)):
            tasks.append(
                self.agent_loop(
                    prompts[i],
                    env_classes[i],
                    env_extras[i],
                    max_tokens,
                    max_input_length,
                    sampling_params=sampling_params,
                )
            )

        # TODO (erictang000): this is still synchronous RL - come back to this
        # for supporting fully async RL
        all_outputs = await asyncio.gather(*tasks)

        responses = sum([[output[0]] for output in all_outputs], [])
        rewards = sum([[output[1]] for output in all_outputs], [])
        stop_reasons = sum([[output[2]] for output in all_outputs], [])
        loss_masks = sum([[output[3]] for output in all_outputs], [])
        prompt_token_ids = sum([[output[4]] for output in all_outputs], [])

        rollout_metrics = self._rollout_metrics(responses, rewards)
        if self.generator_cfg.zero_reward_on_non_stop:
            # set reward to 0 if the stop reason is not "stop"
            rewards = self._zero_reward_if_not_stop(rewards, stop_reasons)

        generator_output: GeneratorOutput = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
        }

        return generator_output

    def _rollout_metrics(self, responses: List[List[int]], rewards: List[float]):
        num_tokens_arr = np.array([len(response) for response in responses])
        non_zero_rewards_arr = np.array([reward > 0.0 for reward in rewards])
        zero_rewards_arr = np.array([reward == 0.0 for reward in rewards])
        # average tokens for non zero rewards
        avg_tokens_non_zero_rewards = (
            np.mean(num_tokens_arr[non_zero_rewards_arr]) if non_zero_rewards_arr.sum() > 0 else np.zeros(1)
        )
        # average tokens for zero rewards
        avg_tokens_zero_rewards = (
            np.mean(num_tokens_arr[zero_rewards_arr]) if zero_rewards_arr.sum() > 0 else np.zeros(1)
        )

        return {
            "generate/min_num_tokens": np.min(num_tokens_arr).item(),
            "generate/max_num_tokens": np.max(num_tokens_arr).item(),
            "generate/avg_num_tokens": np.mean(num_tokens_arr).item(),
            "generate/std_num_tokens": np.std(num_tokens_arr).item(),
            "generate/avg_tokens_non_zero_rewards": avg_tokens_non_zero_rewards.item(),
            "generate/avg_tokens_zero_rewards": avg_tokens_zero_rewards.item(),
        }

    def _zero_reward_if_not_stop(self, rewards: List[float], stop_reasons: List[str]):
        """Sets the reward to 0 if the stop reason is not "stop".

        This can be useful in cases where the LLM generation was truncated or aborted, but the environment still assigns non-zero reward.
        Often, we have format rewards for the LLM to follow, but in cases where the LLM didn't finish the response,
        we typically don't want to reward it. This is a general setting for all environments.
        """
        for i, stop_reason in enumerate(stop_reasons):
            if stop_reason != "stop":
                rewards[i] = 0.0
        return rewards
