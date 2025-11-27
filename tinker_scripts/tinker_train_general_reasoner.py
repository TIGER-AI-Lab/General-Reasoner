"""
This script is modified from https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/rl_loop.py
"""
import logging
import time
import re

from concurrent.futures import Future

import chz
import datasets
import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)

VERIFIER_PROMPT_TEMPLATE = (
    "User: ### Question: {question}\n\n"
    "### Ground Truth Answer: {ground_truth}\n\n"
    "### Student Answer: {student_answer}\n\n"
    "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
    "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
    "If the student's answer is correct, output \"Final Decision: Yes\". If the student's answer is incorrect, output \"Final Decision: No\". Assistant:"
)

VERIFIER_PASS_TAG = "Final Decision: Yes"


def extract_last_boxed(text: str) -> str:
    """
    Extract the last occurrence of a boxed answer from the input text.
    
    Returns:
        The content inside the last \\boxed{...} or None if not found.
    """
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    matches = list(re.finditer(pattern, text))
    if matches:
        return matches[-1].group(1)
    return None


def extract_last_final_answer(text: str) -> str:
    """
    Try to extract the final answer from the text using several candidate patterns.
    
    Returns:
        The extracted answer as a string, or None if none of the patterns match.
    """
    candidate_patterns = [
        r"Final Answer:\s*((?:[^<]|<[^<])*?)\n",
        r"Final Answer is:\s*((?:[^<]|<[^<])*?)\n",
        r"The answer is:\s*((?:[^<]|<[^<])*?)\n",
        r"Answer:\s*((?:[^<]|<[^<])*?)\n",
        r"Solution:\s*((?:[^<]|<[^<])*?)\n",
        r"The solution is:\s*((?:[^<]|<[^<])*?)\n",
    ]
    
    last_match = None
    last_position = -1
    for pattern in candidate_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            if match.start() > last_position:
                last_position = match.start()
                last_match = match.group(1).strip()

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    for stop_word in stop_words:
        if last_match and last_match.endswith(stop_word):
            last_match = last_match[:-len(stop_word)].strip()
    
    return last_match


def extract_solution(solution_str: str) -> str:
    boxed_answer = extract_last_boxed(solution_str)
    if boxed_answer:
        return boxed_answer
    return extract_last_final_answer(solution_str)

class GeneralVerifier:
    def __init__(self, model_name: str):
        self.llm = LLM(model=model_name, gpu_memory_utilization=0.7)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sampling_params = SamplingParams(temperature=0, max_tokens=2048)

    def _truncate_response(self, response: str) -> str:
        if response is None:
            return ""
        return self.tokenizer.decode(self.tokenizer.encode(response)[-1024:])
        
    
    def verify_batch(self, questions: list[str], ground_truths: list[str], responses: list[str]) -> list[bool]:
        student_answers = [extract_solution(response) for response in responses]
        ground_truths = [self._truncate_response(ground_truth) for ground_truth in ground_truths]
        student_answers = [self._truncate_response(student_answer) for student_answer in student_answers]
        messages = [VERIFIER_PROMPT_TEMPLATE.format(question=question, ground_truth=ground_truth, student_answer=student_answer) for question, ground_truth, student_answer in zip(questions, ground_truths, student_answers)]
        outputs = self.llm.generate(messages, sampling_params=self.sampling_params)
        verifier_responses = [output.outputs[0].text.strip() for output in outputs]
        rewards = []
        for verifier_response, ground_truth, student_answer in zip(verifier_responses, ground_truths, student_answers):
            try:
                if VERIFIER_PASS_TAG in verifier_response:
                    # penalize if student answer and ground truth having too different length
                    student_answer_length = len(self.tokenizer.encode(student_answer))
                    ground_truth_length = len(self.tokenizer.encode(ground_truth))
                    difference = abs(student_answer_length - ground_truth_length)
                    difference = min(difference, 10)
                    rewards.append(1.0 - difference * 0.05)
                else:
                    rewards.append(0.0)
            except Exception as e:
                logger.warning(f"Error verifying batch: {e}, verifier_response: {verifier_response}, ground_truth: {ground_truth}, student_answer: {student_answer}")
                rewards.append(0.0)
        return rewards

@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = "./log"
    model_name: str = "Qwen/Qwen3-8B-Base"
    batch_size: int = 1024
    group_size: int = 8
    learning_rate: float = 4e-5
    max_length: int = 8192
    lora_rank: int = 32
    save_every: int = 20
    max_tokens: int = 8192


def main(config: Config):
    # Setup logging
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

    # Get tokenizer and renderer
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    verifier = GeneralVerifier("TIGER-Lab/general-verifier")

    # Load GSM8K dataset
    logger.info("Loading dataset...")
    dataset = datasets.load_dataset("TIGER-Lab/WebInstruct-verified")
    assert isinstance(dataset, datasets.DatasetDict)
    train_dataset = dataset["train"]

    n_train_batches = len(train_dataset) // config.batch_size

    # Setup training client
    service_client = tinker.ServiceClient(base_url=config.base_url)

    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        training_client = service_client.create_training_client_from_state(
            resume_info["state_path"]
        )
        start_batch = resume_info["batch"]
        logger.info(f"Resuming from batch {start_batch}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_batch = 0

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
    )
    # Optimizer step
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    logger.info(f"Training for {n_train_batches} batches")

    #  Main training loop
    for batch_idx in range(start_batch, n_train_batches):
        # Setup metrics for logging
        t_start = time.time()
        step = batch_idx
        metrics: dict[str, float] = {
            "progress/batch": batch_idx,
            "optim/lr": config.learning_rate,
            "progress/done_frac": (batch_idx + 1) / n_train_batches,
        }

        # Save checkpoint
        if step % config.save_every == 0 and step > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{step:06d}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": batch_idx},
            )

        # Get training batch and convert to datums online
        batch_start = batch_idx * config.batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(train_dataset))
        batch_rows = train_dataset.select(range(batch_start, batch_end))

        sampling_path = training_client.save_weights_for_sampler(name=f"{step:06d}").result().path
        sampling_client = service_client.create_sampling_client(model_path=sampling_path)
        # Set up sampling parameters

        training_datums: list[types.Datum] = []
        batch_rewards: list[float] = []
        batch_futures: list[list[Future[types.SampleResponse]]] = []
        batch_prompts: list[list[int]] = []
        
        # Step 1: Generate all samples
        for question in batch_rows["question"]:
            message = [
                {"role": "user", "content": question + " Please reason step by step, and put your final answer within \\boxed{}."}
            ]
            model_input = renderer.build_generation_prompt(message)
            prompt_tokens = model_input.to_ints()

            # Generate response
            sample_futures: list[Future[types.SampleResponse]] = []
            for _ in range(config.group_size):
                sample_futures.append(
                    sampling_client.sample(
                        prompt=model_input,
                        num_samples=1,
                        sampling_params=sampling_params,
                    )
                )

            batch_futures.append(sample_futures)
            batch_prompts.append(prompt_tokens)

        # Step 2: Collect all responses and prepare for verification
        all_questions: list[str] = []
        all_answers: list[str] = []
        all_responses: list[str] = []
        all_metadata: list[dict] = []  # Store metadata for reconstruction
        
        for sample_futures, prompt_tokens, question, answer in zip(
                batch_futures, batch_prompts, batch_rows["question"], batch_rows["answer"]
        ):
            group_tokens: list[list[int]] = []
            group_logprobs: list[list[float]] = []
            group_ob_lens: list[int] = []
            group_responses: list[str] = []
            
            for future in sample_futures:
                sample_result = future.result()
                sampled_tokens = sample_result.sequences[0].tokens
                sampled_logprobs = sample_result.sequences[0].logprobs
                assert sampled_logprobs is not None

                all_tokens = prompt_tokens + sampled_tokens
                group_tokens.append(all_tokens)
                group_ob_lens.append(len(prompt_tokens) - 1)
                group_logprobs.append(sampled_logprobs)

                parsed_message, _ = renderer.parse_response(sampled_tokens)
                response_content = parsed_message["content"]
                group_responses.append(response_content)
                
                # Add to batch-level lists for verification
                all_questions.append(question)
                all_answers.append(answer)
                all_responses.append(response_content)
            
            # Store metadata for this group
            all_metadata.append({
                "group_tokens": group_tokens,
                "group_logprobs": group_logprobs,
                "group_ob_lens": group_ob_lens,
                "group_size": len(group_responses),
                "question": question,
                "answer": answer
            })
        
        # Step 3: Call verifier once for entire batch
        all_rewards = verifier.verify_batch(all_questions, all_answers, all_responses)
        
        # Step 4: Process rewards and create training datums
        reward_idx = 0
        for metadata in all_metadata:
            group_size = metadata["group_size"]
            group_rewards = all_rewards[reward_idx:reward_idx + group_size]
            reward_idx += group_size
            
            advantages = [
                reward - (sum(group_rewards) / len(group_rewards)) for reward in group_rewards
            ]
            batch_rewards.append(sum(group_rewards) / len(group_rewards))

            # Check if all advantages are zero
            if all(advantage == 0.0 for advantage in advantages):
                # Skip question because all advantages are the same
                continue

            for tokens, logprob, advantage, ob_len in zip(
                metadata["group_tokens"], 
                metadata["group_logprobs"], 
                advantages, 
                metadata["group_ob_lens"]
            ):
                input_tokens = tokens[:-1]
                input_tokens = [int(token) for token in input_tokens]
                target_tokens = tokens[1:]
                all_logprobs = [0.0] * ob_len + logprob
                all_advantages = [0.0] * ob_len + [advantage] * (len(input_tokens) - ob_len)
                assert (
                    len(input_tokens)
                    == len(target_tokens)
                    == len(all_logprobs)
                    == len(all_advantages)
                ), (
                    f"len(input_tokens): {len(input_tokens)}, len(target_tokens): {len(target_tokens)}, len(all_logprobs): {len(all_logprobs)}, len(all_advantages): {len(all_advantages)}"
                )
                datum = types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=input_tokens),
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                        "logprobs": TensorData.from_torch(torch.tensor(all_logprobs)),
                        "advantages": TensorData.from_torch(torch.tensor(all_advantages)),
                    },
                )
                training_datums.append(datum)

        # Training step
        fwd_bwd_future = training_client.forward_backward(
            training_datums, loss_fn="importance_sampling"
        )
        optim_step_future = training_client.optim_step(adam_params)
        _fwd_bwd_result = fwd_bwd_future.result()
        _optim_result = optim_step_future.result()

        # Log metrics[]
        metrics["time/total"] = time.time() - t_start
        metrics["reward/mean"] = sum(batch_rewards) / len(batch_rewards)
        ml_logger.log_metrics(metrics, step=batch_idx)

        # Save final checkpoint
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": n_train_batches},
    )
    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    chz.nested_entrypoint(main)